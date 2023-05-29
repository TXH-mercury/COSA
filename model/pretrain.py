from collections import defaultdict
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
import json
import torch.distributed as dist
from .modeling import COSAModel, COSAPreTrainedModel
import ipdb
import numpy as np
import random
from utils.logger import LOGGER
import yaml
from torchvision.transforms import *
import math
from time import time
from tqdm import tqdm
from utils.misc import NoOp
import os
from utils.distributed import any_broadcast
from utils.distributed import all_gather_list
from utils.distributed import ddp_allgather_with_grads, ddp_allgather, all_gather_with_grad
from apex.normalization.fused_layer_norm import FusedLayerNorm 
from scorer.scorer import Scorer
import copy




def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output



class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)
class Config():
    def __init__(self):
        self.void = 'void'
def must_shuffle(ls):
    ls_old = copy.deepcopy(ls)
    while ls == ls_old:
        random.shuffle(ls)
    return ls
def must_replace_one(ls,bs):
    idx = random.randint(0,len(ls)-1)
    rand = random.randint(0,bs-1)
    while rand==ls[idx] :
        rand = random.randint(0,bs-1)

    ls[idx] = rand 
    return ls

class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))


class COSA(COSAModel):
    """ COSA pretraining """
    def __init__(self, opts):
        super().__init__(opts)
        config = opts
        self.max_generation_len = config.max_generation_len
        self.beam_size  = config.beam_size
        self.beam_size_qa  = config.beam_size_qa
        self.evaluate_ret_text = config.evaluate_ret_text
        self.scst_finetuning = config.scst_finetuning
        self.concatenated_nums = config.concatenated_nums
        self.max_text_tokens = config.max_text_tokens


        self.concatenated_raw_pixels = config.concatenated_raw_pixels
        if self.frozen_vision:
            self.concatenated_raw_pixels = False

    
      
        self.itm_rerank_num = opts.itm_rerank_num
        self.itm_ratio = config.itm_ratio   
        self.itm_head = Match_head(self.multimodal_dim)

        if self.scst_finetuning:
            # self.scorer = Scorer()
            self.init_alpah()

                    
        contra_dim = config.contra_dim   
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.video_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))


        

   



        
    

    
    def process_batch_concatenated(self, batch, concatenated_task):
        raw_sentences = batch['raw_sentences']
        batch_size = len(raw_sentences)
        video_pixels = batch['video_pixels']
        max_length = self.max_text_tokens       

        global_batch = all_gather_list(batch_size)
        if self.concatenated_type == 'random':    
            
            global_batch_size = sum(global_batch)
            concatenated_idxs =  torch.tensor([(random.sample(list(range(global_batch_size)), self.concatenated_nums-1)) for i in range(batch_size)])

        else:
            raise NotImplementedError


        ###### concatenated samples
        if self.concatenated_raw_pixels:
            video_pixels_gather = ddp_allgather(video_pixels)
            video_pixels = torch.cat([video_pixels, *(video_pixels_gather[concatenated_idxs[:,i]] for i in range(self.concatenated_nums-1))],dim=1)
            video_output = self.forward_video_encoder(video_pixels)

        else:

            if 'video_output' in batch:
                video_output = batch['video_output']  
            else:
                video_output = self.forward_video_encoder(video_pixels)

            video_output_gather = ddp_allgather(video_output)  
            video_output = torch.cat([video_output,*(video_output_gather[concatenated_idxs[:,i]] for i in range(self.concatenated_nums-1))],dim=1)


            

        batch['video_output'] = video_output
        video_input = self.get_multimodal_forward_input_video(video_output)
        batch['video_input'] = video_input
        


 

        raw_sentences_gather = [ j for i in all_gather_list(raw_sentences) for j in i]
        raw_sentences = [[raw_sentences[i],*(raw_sentences_gather[j] for j in concatenated_idxs[i])] for i in range(batch_size)]
        raw_sentences = [' '.join(i) for i in raw_sentences]

        max_length = max_length * math.ceil(self.concatenated_nums/2)

        multimodal_tokens = self.multimodal_encoder.tokenizer(raw_sentences,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=max_length,
                                                return_tensors="pt")
        multimodal_tokens = multimodal_tokens.to(video_pixels.device)
        batch['multimodal_tokens'] = multimodal_tokens


        

                    

    def process_batch(self, batch):
        video_pixels = batch['video_pixels']
        video_output = self.forward_video_encoder(video_pixels)
        batch['video_output'] = video_output
        video_input = self.get_multimodal_forward_input_video(video_output)
        batch['video_input'] = video_input
        raw_sentences = batch['raw_sentences']
        max_length = self.max_text_tokens
        if raw_sentences[0] is not None:
            multimodal_tokens = self.multimodal_encoder.tokenizer(raw_sentences,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=max_length,
                                                    return_tensors="pt")
            multimodal_tokens = multimodal_tokens.to(video_pixels.device)
            batch['multimodal_tokens'] = multimodal_tokens


    def forward_batch(self,batch, task, compute_loss=True):

        

        output_ls = []
        if 'ret' in task :
            ret_dict = self.forward_ret(batch, task, compute_loss=compute_loss)
            output_ls.append(ret_dict)
        
        if 'cret' in task:
            cret_dict = self.forward_ret_concatenated(batch, task, compute_loss=compute_loss)
            output_ls.append(cret_dict)

        if 'mlm' in task or  'cmlm' in task:
            mlm_dict = self.forward_mlm(batch, compute_loss=compute_loss)
            output_ls.append(mlm_dict)

        if 'cap' in task or 'ccap' in task:
            cap_dict = self.forward_cap(batch, compute_loss=compute_loss)
            output_ls.append(cap_dict)


        if 'qa' in task:
            qa_dict = self.forward_qa(batch, compute_loss=compute_loss)
            output_ls.append(qa_dict)

    
        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }

        return output_dict


    def forward(self, batch, task, compute_loss=True):

        task = task.split('_')
        origin_task = []
        concatenated_task = []
        for i in task:
            if i in ['cmlm','cret','ccap']:
                concatenated_task.append(i)
            else:
                origin_task.append(i)
        

        output_dict_origin= {}
        output_dict_concatenated = {}
    
        ### forward origin
        if origin_task!=[]:
            self.process_batch(batch)
            output_dict_origin = self.forward_batch(batch, origin_task, compute_loss)
            output_dict_origin = {k:v for k,v in output_dict_origin.items()}

        #### forward_concatenated
        if concatenated_task!=[]:
            self.process_batch_concatenated(batch, concatenated_task)
            output_dict_concatenated = self.forward_batch(batch, concatenated_task, compute_loss)
            output_dict_concatenated = {k:v for k,v in output_dict_concatenated.items()}

        
        if compute_loss:
            output_dict = {}
            for k,v in output_dict_origin.items():
                output_dict[k+'_origin'] = v
            for k,v in output_dict_concatenated.items():
                output_dict[k+'_concatenated'] = v        
            return output_dict

        else:
            return output_dict_origin

    def forward_ret_concatenated(self, batch, task, compute_loss=True):
        assert compute_loss
        loss_dict={}
       
        multimodal_tokens = copy.deepcopy(batch['multimodal_tokens'])
        input_ids = multimodal_tokens.input_ids
        input_ids[:,0] = self.multimodal_encoder.tokenizer.itc_token_id  
        attention_mask = multimodal_tokens.attention_mask
        txt_output = self.forward_multimodal_encoder(input_ids, attention_mask).sequence_output
        txt_output = self.pool_text_for_contra(txt_output)

        feat_t = self.contra_head_t(txt_output) 
        feat_t = F.normalize(feat_t,dim=-1)

        video_output = batch['video_output']
        video_output_pooled = self.pool_video_for_contra(video_output)
        feat_v = self.contra_head_v(video_output_pooled)
        feat_v = F.normalize(feat_v,dim=-1)
        batch['feat_t'] = feat_t
        batch['feat_v'] = feat_v

            
          
        image_feats_all = ddp_allgather(feat_v) 
        text_feat_all = ddp_allgather(feat_t)


        sim_i2t = torch.matmul(feat_v, text_feat_all.permute(1,0))
        sim_i2t = sim_i2t / self.contra_temp
        sim_t2i = torch.matmul(feat_t, image_feats_all.permute(1,0))
        sim_t2i = sim_t2i / self.contra_temp  
        rank = dist.get_rank()
        bs = feat_v.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_v.device)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        
        loss_dict['itc_loss'] = loss_itc

           
       
        input_ids = copy.deepcopy(input_ids)
        input_ids[:,0] = self.multimodal_encoder.tokenizer.itm_token_id  
        input_ids_collate = ddp_allgather(input_ids)
        attention_mask_collate = ddp_allgather(attention_mask)
        image_embeds_world = ddp_allgather(video_output)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

    
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

    
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(input_ids_collate[neg_idx])
            text_atts_neg.append(attention_mask_collate[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        


        batch_size = image_embeds_neg.shape[0]

            

        num = batch_size //3 
        input_ids = input_ids[:num]
        text_ids_neg = text_ids_neg[:num]
        attention_mask = attention_mask[:num]
        text_atts_neg = text_atts_neg[:num]
        video_output = video_output[:num]
        image_embeds_neg = image_embeds_neg[:num]
        batch_size = num
        input_ids = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
        attention_mask = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
        video_output = torch.cat((video_output,image_embeds_neg,video_output),dim=0)
        video_input = self.get_multimodal_forward_input_video(video_output)   
        output = self.forward_multimodal_encoder(input_ids, attention_mask, video_input).sequence_output
        ground_truth = torch.zeros(batch_size*3).long().cuda()
        ground_truth[:batch_size] = 1

        logits = self.itm_head(output[:,0])
        itm_loss_tv = F.cross_entropy(logits,ground_truth)
        loss_dict['itm_loss'] =  self.itm_ratio * itm_loss_tv

    
            

        return loss_dict



    def forward_ret(self, batch, task, compute_loss=True):
        

        multimodal_tokens = copy.deepcopy(batch['multimodal_tokens'])
        input_ids = multimodal_tokens.input_ids

        input_ids[:,0] = self.multimodal_encoder.tokenizer.itc_token_id  
        attention_mask = multimodal_tokens.attention_mask
        txt_output = self.forward_multimodal_encoder(input_ids, attention_mask).sequence_output
        txt_output = self.pool_text_for_contra(txt_output)

        feat_t = self.contra_head_t(txt_output) 
        feat_t = F.normalize(feat_t,dim=-1)

        video_output = batch['video_output']
        video_output_pooled = self.pool_video_for_contra(video_output)
        feat_v = self.contra_head_v(video_output_pooled)
        feat_v = F.normalize(feat_v,dim=-1)
        batch['feat_t'] = feat_t
        batch['feat_v'] = feat_v

        if compute_loss:
            ### itc
            image_feats_all = ddp_allgather(feat_v) 
            text_feat_all = ddp_allgather(feat_t)

      
            sim_i2t = torch.matmul(feat_v, text_feat_all.permute(1,0))
            sim_i2t = sim_i2t / self.contra_temp
            sim_t2i = torch.matmul(feat_t, image_feats_all.permute(1,0))
            sim_t2i = sim_t2i / self.contra_temp  # [batch_size, batch_size*num_gpu]
            rank = dist.get_rank()
            bs = feat_v.size(0)
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_v.device)

            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2
 
            loss_dict={}
            loss_dict['itc_loss'] = loss_itc

        

            
            #### itm
            input_ids = copy.deepcopy(input_ids)
            input_ids[:,0] = self.multimodal_encoder.tokenizer.itm_token_id  
            input_ids_collate = ddp_allgather(input_ids)
            attention_mask_collate = ddp_allgather(attention_mask)
            image_embeds_world = ddp_allgather(video_output)


            with torch.no_grad():
                weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
                weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
                weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

            
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_collate[neg_idx])
                text_atts_neg.append(attention_mask_collate[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)
            batch_size = image_embeds_neg.shape[0]

            
            input_ids = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
            attention_mask = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            video_output = torch.cat((video_output,image_embeds_neg,video_output),dim=0)
            video_input = self.get_multimodal_forward_input_video(video_output)   
            output = self.forward_multimodal_encoder(input_ids, attention_mask, video_input).sequence_output
            ground_truth = torch.zeros(batch_size*3).long().cuda()
            ground_truth[:batch_size] = 1

            logits = self.itm_head(output[:,0])
            itm_loss_tv = F.cross_entropy(logits,ground_truth)
            loss_dict['itm_loss'] =  self.itm_ratio * itm_loss_tv

            return loss_dict

        else:
            evaluation_dict = {}
            evaluation_dict['feat_t'] = feat_t
            evaluation_dict['feat_v'] = feat_v
            evaluation_dict['multimodal_tokens'] = multimodal_tokens
          
            video_input = self.get_multimodal_forward_input_video(video_output) 
            evaluation_dict['video_input'] = video_input
        
            
            
            return evaluation_dict







    def forward_mlm(self, batch, compute_loss=True):

        multimodal_tokens = copy.deepcopy(batch['multimodal_tokens'])
        input_ids = multimodal_tokens['input_ids']
        input_ids[:,0] = self.multimodal_encoder.tokenizer.mlm_token_id 
        attention_mask = multimodal_tokens['attention_mask']
        input_ids, txt_labels = self.text_masker(input_ids, 0.15)
        video_input = batch['video_input']
        
        if compute_loss:
            output = self.forward_multimodal_encoder(input_ids, attention_mask, video_input, labels = txt_labels)
            loss_dict = {'mlm_loss': output.loss}
            return loss_dict
        else:
            evaluation_dict={}
            output = self.forward_multimodal_encoder(input_ids, attention_mask, video_input)
            evaluation_dict['mlm_scores_tv'] = output.logits
            evaluation_dict['txt_labels_mlm'] = txt_labels
            return evaluation_dict






    def forward_cap(self, batch, compute_loss=True):

        if compute_loss:
            if self.scst_finetuning:
                return self.forward_cap_scst(batch)
            multimodal_tokens = copy.deepcopy(batch['multimodal_tokens'])
            input_ids, attention_mask = multimodal_tokens['input_ids'], multimodal_tokens['attention_mask']
            input_ids[:,0] = self.multimodal_encoder.tokenizer.bos_token_id
     
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
           
            sample_num = batch['sample_num']
            video_input = batch['video_input']
            video_input_expand = []
            for i in range(video_input.shape[0]):
                video_input_expand.append( video_input[i:i+1].expand(sample_num[i],-1,-1))
            video_input = torch.cat(video_input_expand,dim=0)

        
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])
            output = self.forward_multimodal_encoder(input_ids, attention_mask, video_input, labels = txt_labels)
            loss_dict = {'caption_loss': output.loss}
            return loss_dict

        else:

            video_input = batch['video_input']
            video_input = video_input.unsqueeze(1).reshape(-1,*video_input.shape[1:])
            batch_size = video_input.shape[0]
            init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
            init_attention_mask = init_input_ids.new_ones(batch_size, 1, 1)
            
            with torch.no_grad():
                outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                    attention_mask=init_attention_mask,
                                                    encoder_hidden_states=video_input,
                                                    max_new_tokens=self.max_generation_len,
                                                    num_beams=self.beam_size,
                                                    eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                    pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id,
                                                    length_penalty=0.6) 
                       
            outputs_newgen = outputs[:,1:]
            captions = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
            evaluation_dict = {'generated_captions_t_v' : captions}
            return evaluation_dict



    def forward_qa(self, batch, compute_loss=True):

     
        raw_questions = batch['raw_questions']
        raw_answers = batch['raw_answers']
        sample_num = batch['sample_num']
        video_input = batch['video_input']

        
        question_tokens = self.multimodal_encoder.tokenizer(raw_questions,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=self.max_text_tokens,
                                                return_tensors="pt")

        question_tokens = question_tokens.to(video_input.device)
        question_tokens_ids, question_tokens_mask = question_tokens['input_ids'], question_tokens['attention_mask']
        question_tokens_ids[:,0] = self.multimodal_encoder.tokenizer.bos_token_id

        if compute_loss:
            answer_tokens = self.multimodal_encoder.tokenizer(raw_answers,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=10,
                                                    return_tensors="pt")
        
       
            answer_tokens = answer_tokens.to(video_input.device)
            answer_tokens_ids, answer_tokens_mask = answer_tokens['input_ids'], answer_tokens['attention_mask']
            answer_tokens_ids[:,0] = self.multimodal_encoder.tokenizer.bos_token_id  
            input_ids, txt_labels = self.text_masker(answer_tokens_ids, 0.99)

            input_ids = torch.cat((question_tokens_ids,input_ids),dim=1)
            attention_mask = torch.cat((question_tokens_mask,answer_tokens_mask),dim=1)

                 

            dummy_labels = (-100*torch.ones_like(question_tokens_ids)).cuda()
            txt_labels = torch.cat((dummy_labels,txt_labels),dim=1)

            #### part-causal attention mask
            question_len = question_tokens_ids.shape[1]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, question_len: seq_len, question_len: seq_len] = torch.tril(attention_mask[:, question_len: seq_len, question_len: seq_len])
            attention_mask[:, :question_len, question_len:seq_len] = 0
           

            output = self.forward_multimodal_encoder(input_ids, attention_mask, video_input, labels = txt_labels)
            loss_dict = {'vqa_loss': output.loss}
            return loss_dict
        
        else:

            video_input_expand = []
            for i in range(video_input.shape[0]):
                video_input_expand.append( video_input[i:i+1].expand(sample_num[i],-1,-1))
            video_input = torch.cat(video_input_expand,dim=0)
            batch_size = video_input.shape[0]


            init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
            init_input_ids = torch.cat((question_tokens['input_ids'],init_input_ids),dim=1)
            question_len = question_tokens['input_ids'].shape[1]
            seq_len = init_input_ids.shape[1]
            attention_mask = question_tokens['attention_mask'].unsqueeze(1).expand(-1, question_len, -1).clone()
            init_attention_mask = self.multimodal_encoder.update_attention_mask(attention_mask)
            slice_size = 50
            answers = []
        
            times = (len(init_input_ids)-1) // slice_size + 1
            for i in range(times):
                with torch.no_grad():
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids[i*slice_size: i*slice_size+slice_size],
                                                                attention_mask=init_attention_mask[i*slice_size: i*slice_size+slice_size],
                                                                encoder_hidden_states=video_input[i*slice_size: i*slice_size+slice_size],
                                                                max_new_tokens=self.max_generation_len,
                                                                num_beams=self.beam_size_qa,
                                                                eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                                pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 
                    
                outputs_newgen = outputs[:,seq_len:]
                answers += self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)

            evaluation_dict = {'generated_answers_t_v' : answers}

            return evaluation_dict



    def process_scst(self,seq):
        N, T = seq.size()
        sents = []
        for n in range(N):
            tokens = []
            for t in range(T):
                ix = seq[n, t].item()
                if ix == self.multimodal_encoder.tokenizer.eos_token_id:
                    #tokens.append(ix)  ### add
                    break
                tokens.append(ix)
            sents.append(tokens)
        return sents

    def forward_cap_scst(self, batch):

 
        loss_dict = {}
        batch_ids = batch['ids']

        video_input = batch['video_input']
        video_input = video_input.unsqueeze(1).reshape(-1,*video_input.shape[1:])
        batch_size = video_input.shape[0]
        init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
        init_attention_mask = init_input_ids.new_ones(batch_size, 1, 1)

        self.eval()
        with torch.no_grad():
            outputs_greedy = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                        attention_mask=init_attention_mask,
                                                        encoder_hidden_states=video_input,
                                                        max_new_tokens=self.max_generation_len,
                                                        num_beams=self.beam_size,
                                                        eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                        pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id)  ### compute  reward baseline

            outputs_greedy = outputs_greedy[:,1:]
        self.train()
                            
        
        outputs_sample, logprobs = self.multimodal_encoder.generate_scst( input_ids=init_input_ids,
                                                    attention_mask=init_attention_mask,
                                                    do_sample = True,
                                                    encoder_hidden_states=video_input,
                                                    max_new_tokens=self.max_generation_len,
                                                    eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                    pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id)        
        outputs_sample = outputs_sample[:,1:]      
    
        outputs_greedy_processed = self.process_scst(outputs_greedy)
        outputs_sample_processed = self.process_scst(outputs_sample)

        reward_greedy = self.scorer(batch_ids, outputs_greedy_processed)
        reward_sample = self.scorer(batch_ids, outputs_sample_processed)

        self.update_alpha(reward_sample, reward_greedy)
        rewards = reward_sample - reward_greedy * self.get_alpha()
        rewards = torch.from_numpy(rewards).float().cuda()
        caption_loss_tv = self.reward_loss(outputs_sample, logprobs, rewards)    
        loss_dict['caption_loss_tv'] = caption_loss_tv
        
        return loss_dict


    def init_alpah(self):
    
        self.alpha_type = 0

        self.total_alpha = 0.7
        self.beta = 1.0
        self.recent_alpha = 0.7
        self.recent_num = 5000
        self.recent_alpha_list = np.linspace(0, 0, self.recent_num)
        self.recent_index = 0

        self.reward_sample_total = 0
        self.reward_greedy_total = 0
        self.reward_num = 0

    def update_alpha(self, rewards_sample, rewards_max):

        sample_mean = rewards_sample.mean()
        greedy_mean = rewards_max.mean()

        # total
        self.reward_sample_total += sample_mean
        self.reward_greedy_total += greedy_mean
        self.reward_num += 1
        self.total_alpha = self.reward_sample_total / self.reward_greedy_total

        # recent num
        self.recent_alpha_list[self.recent_index % self.recent_num] = sample_mean / greedy_mean
        self.recent_index += 1
        self.recent_alpha = np.mean(self.recent_alpha_list[:min(self.recent_index, self.recent_num)])

        reward_sample_avg = self.reward_sample_total / self.reward_num
        reward_greedy_avg = self.reward_greedy_total / self.reward_num


    def get_alpha(self):

        if self.alpha_type == 0:
            temp_alpha = 1.0
        elif self.alpha_type == 1:
            temp_alpha = self.recent_alpha * self.beta
        elif self.alpha_type == 2:
            temp_alpha = self.total_alpha * self.beta
        else:
            raise Exception("Error alpha_type")
      
        return temp_alpha


    def tile(self,x, dim, n_tile):
        init_dim = x.size(dim)
        repeat_idx = [1] * x.dim()
        repeat_idx[dim] = n_tile
        x = x.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(x, dim, order_index.to(x.device))    

    def reward_loss(self, seq, logP, rewards):
        mask = (seq != 0)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss



