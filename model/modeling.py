"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
from builtins import NotImplementedError
import copy
import json
import ipdb
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm 
import random
import numpy as np
from utils.logger import LOGGER
import torch.nn.functional as F
from time import time




class COSAConfig(object):
    def __init__(self,
                 config):
        
        if isinstance(config, dict):
            for key, value in config.items():
                self.__dict__[key] = value

        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `COSAConfig` from a
           Python dictionary of parameters."""
        config = COSAConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `COSAConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class COSAPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self,  *inputs, **kwargs):
        super().__init__()
        # if not isinstance(config, COSAConfig):
        #     raise ValueError(
        #         "Parameter config in `{}(config)` should be an instance of "
        #         "class `COSAConfig`. To create a model from a Google "
        #         "pretrained model use "
        #         "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
        #             self.__class__.__name__, self.__class__.__name__
        #         ))
        # self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=0.02)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, opts, state_dict, *inputs, **kwargs):
        model = cls(opts, *inputs, **kwargs)

        missing_keys,unexpected_keys = model.load_state_dict(state_dict,strict=False)
        if state_dict != {}:
            # print(state_dict)
            LOGGER.info(f"Unexpected keys {unexpected_keys}")
            LOGGER.info(f"missing_keys  {missing_keys}")
        return model


def valid(x):
    return x is not None


class TokenMasker(nn.Module):
    def __init__(self, mask_token = -1, range_start=-1, range_end=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start,range_end]

    def forward(self, tokens, mask_prob):
        tokens = tokens.clone() ### important, must have
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels

    
    def perform_mask(self, tokens, mask_prob):
        
        tokens = np.array(tokens.cpu().numpy())

        ### generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j]!=0 and random.random() < mask_prob:
                        mask_indicator[i][j] = 1
        
        


        labels = -np.ones(tokens.shape, dtype=np.int64) * 100 ### -100 ignore idx for nn.CrossEntropyLoss used in BERT
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                
                if mask_indicator[i][j] == 1 :
                    src_token = tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  ### e-6 have no idea why too much 
                    elif prob < 0.9: 
                        tokens[i][j] = random.choice(list(range(*self.range)))   
                    #tokens[i][j] = self.mask_token
                    labels[i][j] = src_token


        tokens =torch.from_numpy(tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return tokens, labels







class COSAModel(COSAPreTrainedModel):

    def __init__(self, opts):
        super().__init__()
        config = opts
        self.config = config

       
        self.video_encoder_type = config.video_encoder_type
        self.multimodal_encoder_type = config.multimodal_encoder_type
        
        ##### construct vision encoder 
        if self.video_encoder_type.startswith('clip'):
            self.load_clip_model(config) 
        elif self.video_encoder_type.startswith('swin'):
            self.load_swin_model(config)           
        else:
            raise NotImplementedError

      
        ### construct multimodal encoder
        if self.multimodal_encoder_type.startswith('bert'):
            self.load_bert_model(config)
        else:
            raise NotImplementedError
    
        self.video_frame_embedding = nn.Parameter(0.02 * torch.randn(1, opts.train_video_sample_num, self.multimodal_dim))
        self.frozen_vision = config.frozen_vision
        self.hidden_trans_video_multimodal = nn.Sequential(nn.Linear(self.video_dim, self.multimodal_dim),FusedLayerNorm(self.multimodal_dim, eps=1e-12))


    def pool_text_for_contra(self, feature):

        if self.multimodal_encoder_type.startswith('bert'):
            return feature[:,0]
        else:
            return NotImplementedError


    def pool_video_for_contra(self, feature):  #feature b ,n ,x ,c
        #### always use frame_avg  for retrieval
        if self.video_encoder_type.startswith('clip'):
            feature = feature[:,:,0]
        elif self.video_encoder_type.startswith('swin'):
            feature = feature.mean(dim=2)
        feature = torch.mean(feature, dim=1)
        return feature

    def forward_video_encoder(self, video_pixels):   ### b,n,3,h,w
        

        if self.frozen_vision:
            if self.video_encoder_type.startswith('clip'):
                self.clip_model.eval()
            else:
                self.video_encoder.eval()


        b,n,_,h,w = video_pixels.shape
        
        if self.video_encoder_type.startswith('clip'):
            
            video_output = self.clip_model.encode_image(video_pixels.reshape(b*n,3,h,w))
            video_output = video_output.reshape(b,-1,*video_output.shape[-2:])
     
        
        elif self.video_encoder_type.startswith('swin'):
            video_output = self.video_encoder(video_pixels.reshape(b*n,3,h,w))
            video_output = video_output.reshape(b,-1,*video_output.shape[-2:])
    
        else:
            raise NotImplementedError()

        return video_output  #### B , n , x ,C  n = self.frame_num





    def get_multimodal_forward_input_video(self, video_output):

        b,n,x,c = video_output.shape
        video_output = self.hidden_trans_video_multimodal(video_output)  
        # print(self.video_frame_embedding.shape[1])

        if n!=self.video_frame_embedding.shape[1]: #### testing and interpolate
            video_frame_embedding = F.interpolate(self.video_frame_embedding.permute(0,2,1),n,mode='nearest').permute(0,2,1)
        else:
            video_frame_embedding = self.video_frame_embedding
        

        # import ipdb
        # ipdb.set_trace()
        video_output =  video_output + video_frame_embedding.unsqueeze(-2)

        video_output =  video_output.reshape(b,-1,self.multimodal_dim) 


        return video_output



    def forward_multimodal_encoder(self, input_ids, attention_mask, video_input=None, labels=None, position_ids=None):

        if self.multimodal_encoder_type.startswith('bert'):
            return self.multimodal_encoder(input_ids = input_ids,
                                           attention_mask = attention_mask,
                                           encoder_hidden_states=video_input,
                                           labels = labels,
                                           position_ids = position_ids
                                          )



                                     

        
   
    def load_clip_model(self,config):
        from .clip import build_model
        from .clip import Transformer
        from transformers import CLIPTokenizer
        if  self.video_encoder_type == 'clip_vit_base_16':
            clip_weight = torch.jit.load('./pretrained_weights/CLIP/clip-vit-base-16.pt', map_location='cpu')
            self.video_dim = 768
        elif self.video_encoder_type == 'clip_vit_large_14_336px':
            clip_weight = torch.jit.load('./pretrained_weights/CLIP/clip-vit-large-14-336px.pt', map_location='cpu')
            self.video_dim = 1024
        clip_weight = clip_weight.state_dict()

        self.clip_model = build_model(clip_weight, config.video_resolution, config.checkpointing).float()
        
        self.clip_model.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        
        if config.frozen_vision:
            for k,v in self.clip_model.named_parameters():
                if 'visual' in k and not 'ada' in k:
                    v.requires_grad = False 


    

  


    def load_swin_model(self,config):
        from .swin import SwinTransformer
        from .swin_config import get_config

        if self.video_encoder_type.startswith('swin_base_22k_224'):
            swin_config = get_config('./pretrained_weights/SWIN/swin_base_patch4_window7_224_22k.yaml')
            swin_weight = torch.load('./pretrained_weights/SWIN/swin_base_patch4_window7_224_22k.pth', map_location='cpu')['model']
            self.video_dim=1024

        else:
            raise NotImplementedError

        # elif self.video_encoder_type.startswith('swin_large_22k_224'):
        #     swin_config = get_config('./pretrained_weights/SWIN/swin_large_patch4_window7_224_22k.yaml')
        #     swin_weight = torch.load('./pretrained_weights/SWIN/swin_large_patch4_window7_224_22k.pth', map_location='cpu')['model']
        #     self.video_dim=1536

        model_type = swin_config.MODEL.TYPE
        if swin_config.FUSED_LAYERNORM:
            try:
                import apex as amp
                layernorm = amp.normalization.FusedLayerNorm
            except:
                layernorm = None
                print("To use FusedLayerNorm, please install apex.")
        else:
            import torch.nn as nn
            layernorm = nn.LayerNorm
        
        self.video_encoder = SwinTransformer(img_size=swin_config.DATA.IMG_SIZE,
                                patch_size=swin_config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=swin_config.MODEL.SWIN.IN_CHANS,
                                num_classes=swin_config.MODEL.NUM_CLASSES,
                                embed_dim=swin_config.MODEL.SWIN.EMBED_DIM,
                                depths=swin_config.MODEL.SWIN.DEPTHS,
                                num_heads=swin_config.MODEL.SWIN.NUM_HEADS,
                                window_size=swin_config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=swin_config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=swin_config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=swin_config.MODEL.SWIN.QK_SCALE,
                                drop_rate=swin_config.MODEL.DROP_RATE,
                                drop_path_rate=swin_config.MODEL.DROP_PATH_RATE,
                                ape=swin_config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=swin_config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=swin_config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=swin_config.FUSED_WINDOW_PROCESS)

        
        missing_keys, unexpected_keys = self.video_encoder.load_state_dict(swin_weight,strict=False)

        del(swin_weight)
        #LOGGER.info(f'missing_keys in video encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in video encoder: {unexpected_keys}')

        if config.frozen_vision:
            for k,v in self.video_encoder.named_parameters():
                v.requires_grad = False 
        
   

      
    def load_bert_model(self, config):
        
        from model.bert import BertForMaskedLM, BertConfig
        if self.multimodal_encoder_type == 'bert_base_uncased':
            # bertconfig = BertConfig.from_pretrained("bert-base-uncased")
            # bertconfig.add_cross_attention = True
            # bertconfig.is_decoder = True
            # self.multimodal_encoder = BertForMaskedLM.from_pretrained("bert-base-uncased",config = bertconfig )
            # self.multimodal_encoder.save_pretrained('./pretrained_weights/BERT/bert-base-uncased-crossattn')
            self.multimodal_encoder = BertForMaskedLM.from_pretrained('./pretrained_weights/BERT/bert-base-uncased-crossattn')
            self.multimodal_dim = 768

        elif self.multimodal_encoder_type == 'bert_large_uncased':
            # bertconfig = BertConfig.from_pretrained("bert-large-uncased")
            # bertconfig.add_cross_attention = True
            # bertconfig.is_decoder = True
            # self.multimodal_encoder = BertForMaskedLM.from_pretrained("bert-large-uncased",config = bertconfig )
            # self.multimodal_encoder.save_pretrained('./pretrained_weights/BERT/bert-large-uncased-crossattn')
            self.multimodal_encoder = BertForMaskedLM.from_pretrained('./pretrained_weights/BERT/bert-large-uncased-crossattn')
            self.multimodal_dim = 1024

        else:
            raise NotImplementedError()
        
      

        if config.checkpointing:
            self.multimodal_encoder._set_gradient_checkpointing(self.multimodal_encoder.bert.encoder, True)

        from transformers import BertTokenizer
        # self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.multimodal_encoder.tokenizer.save_pretrained('./pretrained_weights/BERT/tokenizer')

        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/BERT/tokenizer')
        self.multimodal_encoder.tokenizer.cls_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.bos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.eos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.multimodal_encoder.tokenizer.pad_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.multimodal_encoder.tokenizer.mask_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.multimodal_encoder.tokenizer.itm_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.mlm_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.itc_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]

        self.text_mask_token = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.text_masker = TokenMasker(mask_token = self.text_mask_token, range_start=106, range_end = 30522)


      
        
    

def trans(x):
    return torch.from_numpy(x)


