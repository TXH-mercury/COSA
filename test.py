from builtins import ValueError
import json
import math
import os
from time import time
import torch
from torch.nn import functional as F
import torch.distributed as dist
from utils.logger import LOGGER
from utils.distributed import ddp_allgather, all_gather_list

from utils.misc import NoOp
from cococaption.pycocoevalcap.eval import COCOEvalCap
from cococaption.pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm 


def validate(model, val_dataloaders, opts, global_step, total_step):

    eval_log = {}
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        val_log = validate_single(model, loader, task.split('--')[0], opts, global_step, total_step,task.split('--')[1])
        eval_log[task] = val_log
    model.train()
    return eval_log


@torch.no_grad()
def validate_single(model, val_loader, task, opts, global_step, total_step,dset_name):
    LOGGER.info("start running {} validation...".format(task))

    task = task.split('_')

    output_ls = []

    if 'ret' in task:
        ret_dict = validate_ret(model, val_loader, opts, global_step)
        output_ls.append(ret_dict)
    if 'cap' in task:
        cap_dict = validate_cap(model, val_loader, opts, global_step, dset_name)
        output_ls.append(cap_dict)
    if 'qa' in task:
        qa_dict = validate_qa(model, val_loader, opts, global_step, dset_name)
        output_ls.append(qa_dict)

    output_dict = {k:v for dic in output_ls for k,v in dic.items() }
    return output_dict

@torch.no_grad()
def validate_qa(model, eval_loader, opts, global_step,dset_name):
    st = time()
    val_log = {}

    # task = task_str.split('%')[1:]
    output_dir = os.path.join(opts.output_dir,f'predict_answers')
    os.makedirs(output_dir,exist_ok=True)
    

    groundtruth_answers=[]
    generated_answers_t_v = []


    
    if dist.get_rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()


    answer_tokens = None
    
    submit_file_tv = []
        
    for batch in eval_loader:
        ids = batch['ids']
        groundtruth_answers += batch['raw_answers']
        # if global_step !=0:
        batch['global_step'] = global_step
        evaluation_dict = model(batch, 'qa', compute_loss=False)

        answers = evaluation_dict['generated_answers_t_v']
        generated_answers_t_v += answers
        if eval_loader.dataset.make_submission:
            if eval_loader.dataset.name.startswith('vqav2'):
                for i in range(len(batch['question_ids'])):
                    submit_file_tv.append({'question_id':batch['question_ids'][i],'answer':answers[i]})
            else:
                raise NotImplementedError




        pbar.update(1)
        

    pbar.close()

    groundtruth_answers = [i for j in all_gather_list(groundtruth_answers)  for i in j]
    if dist.get_rank()==0:
        json.dump(groundtruth_answers,open(os.path.join(output_dir,f'step{global_step}_gt_{dset_name}.json'),'w'))
    total_num = len(groundtruth_answers)
    #assert len(all_groundtruth_answers) == total_num
    LOGGER.info('total {} questions has been tested'.format(total_num))

   
    generated_answers_t_v = [i for j in all_gather_list(generated_answers_t_v)  for i in j]
    if dist.get_rank()==0:
        json.dump(generated_answers_t_v,open(os.path.join(output_dir,f'step{global_step}_tv_pred_{dset_name}.json'),'w'))
    submit_files_t_v = [i for j in all_gather_list(submit_file_tv)  for i in j]
    if dist.get_rank()==0:
        json.dump(submit_files_t_v,open(os.path.join(output_dir,f'step{global_step}_tv_pred_submited_{dset_name}.json'),'w'))
    accurate_num = sum([generated_answers_t_v[i] == groundtruth_answers[i] for i in range(total_num)])
    accuracy = accurate_num / total_num
    val_log['tv'] = {'accuracy':round(accuracy*100,2)} 
    
    return val_log




@torch.no_grad()
def validate_cap(model, eval_loader, opts, global_step,dset_name):
    st = time()
    val_log = {}   
    result_folder = os.path.join(opts.output_dir, f'results_test_{dset_name}')
    os.makedirs(result_folder, exist_ok=True)
    generated_captions_t_v = []
    if dist.get_rank() == 0:
        # pbar = tqdm(total=len(eval_loader))
        pbar = tqdm()
    else:
        pbar = NoOp()

    for batch in eval_loader:
        ids = batch['ids']
        evaluation_dict = model(batch, 'cap', compute_loss=False)
        sents = evaluation_dict['generated_captions_t_v']       
        for  i in range(len(sents)):
            generated_captions_t_v.append({'video_id':ids[i], 'caption': sents[i]})
        pbar.update(1)
    annfile_path = eval_loader.dataset.annfile
    pbar.close()

    results = [i for j in all_gather_list(generated_captions_t_v)  for i in j]
    if dist.get_rank()==0:
        val_log['tv'] = compute_metric_cap(results, annfile_path) 
        json.dump(results,open(os.path.join(result_folder, 'step_{}_tv.json'.format(global_step)), 'w'))
   
    return val_log






@torch.no_grad()
def validate_ret(model, val_loader, opts, global_step):
    val_log = {}
    feat_t = []
    feat_v = []
    ids = []
    ids_txt = []
    input_ids = []
    attention_mask = []
    video_input =[]

    for i, batch in enumerate(val_loader):
        evaluation_dict= model(batch, 'ret', compute_loss=False)
        feat_t.append(evaluation_dict['feat_t'])
        feat_v.append(evaluation_dict['feat_v'])
        input_ids.append(evaluation_dict['multimodal_tokens']['input_ids'])
        attention_mask.append(evaluation_dict['multimodal_tokens']['attention_mask'])
        ids += batch['ids']
        if batch['ids_txt'] is  None:
            ids_txt  += batch['ids']
        else:
            ids_txt  += batch['ids_txt']

        
        video_input.append(evaluation_dict['video_input'])

    
    ids = [j for i in all_gather_list(ids) for j in i]
    ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
    

    if feat_t[0] is not None:
        feat_t = torch.cat(feat_t, dim = 0)
        feat_t = ddp_allgather(feat_t).half()
    if feat_v[0] is not None:
        feat_v = torch.cat(feat_v, dim = 0)
        feat_v = ddp_allgather(feat_v).half()
   

    if input_ids[0] is not None:  
        input_ids = torch.cat([i for i in input_ids],dim=0)
        input_ids = ddp_allgather(input_ids)
        attention_mask = torch.cat([i for i in attention_mask],dim=0)
        attention_mask = ddp_allgather(attention_mask)


        
    score_matrix_t_v = torch.matmul(feat_t,feat_v.permute(1,0))
    log = compute_metric_ret(score_matrix_t_v, ids, ids_txt, model)
    log = {k.replace('forward','video').replace('backward','txt') : v for k,v in log.items()}
    val_log['t_v'] = log





    #### itm


    video_input = torch.cat(video_input, dim = 0).half()
    top_k = get_model_attr(model,'itm_rerank_num')
    idxs = score_matrix_t_v.topk(top_k,dim=1)[1]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    nums = score_matrix_t_v.shape[0]//world_size +1
    
    score_matrix_t_v_new = torch.zeros_like(score_matrix_t_v)
    idxs_new = torch.zeros_like(score_matrix_t_v_new).long()
    for i in range(len(idxs)):
        for j in idxs[i]:
            idxs_new[i][j] = 1
    
    cur_length = video_input.shape[0]
    length_ls = all_gather_list(cur_length)
    start = 0
    start_ls = []
    end_ls = []
    for l in range(len(length_ls)):
        start_ls.append(start)
        end_ls.append(start+length_ls[l])
        start = start+length_ls[l]
    
    cur_score_matrix_t_v = score_matrix_t_v[:,start_ls[rank]:end_ls[rank]]
    cur_score_matrix_t_v_new = score_matrix_t_v_new[:,start_ls[rank]:end_ls[rank]]
    cur_idxs_new = idxs_new[:,start_ls[rank]:end_ls[rank]]

    if dist.get_rank() == 0:
        pbar = tqdm(total=cur_length)
    else:
        pbar = NoOp()
    for i in range(cur_length):
        if sum(cur_idxs_new[:,i] == 1) == 0:
            continue
        cur_scores = []
        cur_input_ids = input_ids[(cur_idxs_new[:,i] == 1)]
        cur_attention_mask = attention_mask[(cur_idxs_new[:,i] == 1)]
        

        cur_video_input = video_input[i].unsqueeze(0).expand(cur_input_ids.shape[0],-1,-1)
        total_len = len(cur_video_input)
        small_batch=25
        times = total_len//small_batch if total_len%small_batch==0 else total_len//small_batch+1

        for k in range(times):

            slice_input_ids = cur_input_ids[k*small_batch:(k+1)*small_batch]
            slice_attention_mask = cur_attention_mask[k*small_batch:(k+1)*small_batch]

          
            slice_video_input = cur_video_input[k*small_batch:(k+1)*small_batch]
            slice_output = get_model_attr(model,'forward_multimodal_encoder')(slice_input_ids, slice_attention_mask, slice_video_input).sequence_output
            
            slice_scores = F.softmax(get_model_attr(model,'itm_head')(slice_output[:,0]),dim=1)[:,1]
            cur_scores.append(slice_scores)
        cur_scores = torch.cat(cur_scores,dim=0)

        cur_score_matrix_t_v_new[:,i][(cur_idxs_new[:,i] == 1)] = cur_scores
        pbar.update(1)
    pbar.close()
    
    score_matrix_t_v = ddp_allgather(cur_score_matrix_t_v_new.T.contiguous()).T

    log = compute_metric_ret(score_matrix_t_v, ids, ids_txt, model)
    log = {k.replace('forward','video').replace('backward','txt') : v for k,v in log.items()}
    val_log['t_v_itm'] = log



    return val_log



def get_model_attr(model, attr_name):

    
    if hasattr(model,'module') and hasattr(model.module,attr_name):
        return getattr(model.module, attr_name)

    elif hasattr(model,attr_name):
        return getattr(model, attr_name)
    
    else:
        return ValueError

    




def compute_metric_ret(score_matrix, ids, ids_txt, model):
    assert score_matrix.shape == (len(ids_txt),len(ids))
    indice_matrix = score_matrix.sort(dim=-1,descending=True)[1].tolist()
    rank = []
    for i in range(len(ids_txt)):
        gt_indice = ids.index(ids_txt[i])
        rank.append(indice_matrix[i].index(gt_indice))
    
    rank = torch.tensor(rank).to(score_matrix)
    
    vr_r1 = (rank < 1).sum().item() / len(ids_txt)
    vr_r5 = (rank < 5).sum().item() / len(ids_txt)
    vr_r10 = (rank < 10).sum().item() / len(ids_txt)
    v_medianR = torch.median(rank).item() +1
    v_meanR = torch.mean(rank).item() +1

   
    if get_model_attr(model,'evaluate_ret_text'):
       
        indice_matrix = score_matrix.sort(dim=0,descending=True)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1), 
                    'backward_r1': round(tr_r1*100,1),
                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1) 
   }
    
    else:
        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)}
    return eval_log





def compute_metric_cap(results, annfile_path, process=True):
    coco = COCO(annfile_path)
    cocoRes = coco.loadRes(results)
    cocoEval = COCOEvalCap(coco, cocoRes, process)
    cocoEval.evaluate()
    metric = cocoEval.eval
    metric = {k: round(v*100,2)  for k,v in metric.items()}
    return metric
