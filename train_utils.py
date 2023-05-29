import os
from numpy import short
import torch 
import json
import argparse
from data import  VideoMapper
from data.data import COSADataset, cosa_collate
from optim.misc import build_optimizer
from utils.misc import NoOp, parse_with_config, set_random_seed
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from tqdm import tqdm 
from utils.save import ModelSaver
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed import DistributedSampler_wopadding
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from data import PrefetchLoader
from collections import defaultdict
from apex import amp
import torch.nn.functional as F
from optim import get_lr_sched
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from time import time
# from data.data_webvid_web import WebvidFrameDataset
from data import  MetaLoader, PrefetchLoader , AccumMetaLoader
from test import validate
from easydict import EasyDict as edict
from scorer.scorer import Scorer
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


#torch.autograd.set_detect_anomaly(True)



def initialize(opts):
    if not os.path.exists(opts.output_dir):
        os.makedirs(os.path.join(opts.output_dir, 'log'), exist_ok=True)
        os.makedirs(os.path.join(opts.output_dir, 'ckpt'), exist_ok=True)
    local_rank = opts.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl') 
    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if dist.get_rank() == 0:
        TB_LOGGER.create(os.path.join(opts.output_dir, 'log'))
        add_log_to_file(os.path.join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
    if opts.test_video_sample_num != -1:
        for d_cfg in opts.data_cfg.val:
            d_cfg.video_sample_num = opts.test_video_sample_num

    if opts.train_video_sample_num != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.video_sample_num = opts.train_video_sample_num

    if opts.test_batch_size != -1:
        for d_cfg in opts.data_cfg.val:
            d_cfg.batch_size = opts.test_batch_size
    
    if opts.train_batch_size != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.batch_size = opts.train_batch_size



    if opts.train_task != '':
        for d_cfg in opts.data_cfg.train:
            d_cfg.task = opts.train_task


    if opts.test_task != '':
        assert len(opts.data_cfg.val)==1
        opts.data_cfg.val[0].task = opts.test_task

    if opts.video_transforms !='none':
        assert len(opts.data_cfg.train)==1
        assert len(opts.data_cfg.val)==1
        opts.data_cfg.train[0]['datasets'][0]['video_transforms'] = opts.video_transforms
        opts.data_cfg.val[0]['video_transforms'] = opts.video_transforms




    if opts.train_epoch != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.epoch = opts.train_epoch
    if opts.train_steps != -1:
        for d_cfg in opts.data_cfg.train:
            d_cfg.steps = opts.train_steps

    
  

def load_from_pretrained_dir(opts, input_args):

    checkpoint_dir = os.path.os.path.join(opts.pretrain_dir,'ckpt')
    if opts.pretrain_step is not None:
        step = opts.pretrain_step
    else:
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
        
    checkpoint_name = 'model_step_'+str(step)+'.pt'
    ckpt_file = os.path.os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_file, map_location = 'cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    LOGGER.info(f'load_from_pretrained: {ckpt_file}')

    pretrain_cfg = edict(json.load(open(os.path.join(opts.pretrain_dir,'log','hps.json'))))
    ### cover model_cfg 
    cover_cfg=["video_encoder_type", "multimodal_encoder_type"]


    for k in cover_cfg:
        if k in pretrain_cfg and not k in input_args:
            setattr(opts,k,pretrain_cfg[k])

 

    pretrain_embed = checkpoint['video_frame_embedding']
    if pretrain_embed.shape[1]!=opts.train_video_sample_num:
        if pretrain_embed.shape[1] == 32: ### old
            pretrain_embed = pretrain_embed[:,:pretrain_cfg.video_sample_num]
        pretrain_embed = F.interpolate(pretrain_embed.permute(0,2,1),opts.train_video_sample_num,mode='nearest').permute(0,2,1)
        checkpoint['video_frame_embedding'] = pretrain_embed

   

    if opts.video_resolution != pretrain_cfg['video_resolution']:
        if opts.video_encoder_type.startswith('clip'):
            vision_width = checkpoint["clip_model.visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = checkpoint["clip_model.visual.conv1.weight"].shape[-1]
            
            grid_size = round((checkpoint["clip_model.visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
            src  = checkpoint["clip_model.visual.positional_embedding"]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = opts.video_resolution // vision_patch_size
            src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
            src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
            tgt = torch.cat((src_cls,src_oth),dim=0)
            checkpoint["clip_model.visual.positional_embedding"] = tgt
  
      
         



        else:
            pass

    
    return checkpoint


def load_from_resume(opts):
    ckpt_dir = os.path.join(opts.output_dir,'ckpt')
    previous_optimizer_state = [i  for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
    steps = [i.split('.pt')[0].split('_')[-1] for i in  previous_optimizer_state] 
    steps = [ int(i) for i in steps]
    steps.sort()
    previous_step = steps[-1]
    previous_optimizer_state = f'optimizer_step_{previous_step}.pt'
    previous_model_state = f'model_step_{previous_step}.pt'
    previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
    previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
    previous_model_state = os.path.join(ckpt_dir, previous_model_state)
    
    assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
    LOGGER.info("choose previous model: {}".format(previous_model_state))
    LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    previous_model_state = torch.load(previous_model_state,map_location='cpu')
    previous_optimizer_state = torch.load(previous_optimizer_state,map_location='cpu')
    return previous_model_state, previous_optimizer_state, previous_step


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')



def cover(opts, config, x, y):
    if getattr(opts, x) is not None:
        config[y] =  getattr(opts, x)



def set_parallel_optimizer_and_apex(model, opts, checkpoint_optim):
    device = torch.device("cuda", opts.local_rank)
    model.to(device)

    ### initialize optimizer
    optimizer = build_optimizer(model, opts)
    optimizer.zero_grad()

    ### apex initialize 

    # if opts.amp=='apex':
    model, optimizer = amp.initialize(model, optimizer, enabled=opts.fp16, opt_level='O2')

    if checkpoint_optim:
        optimizer.load_state_dict(checkpoint_optim)
        del(checkpoint_optim)


    if not opts.checkpointing:
        model = DDP(model, device_ids=[opts.local_rank], output_device=opts.local_rank, find_unused_parameters=True)
    else:
        pass
    model.train()

    LOGGER.info(f"  basic_lr : {optimizer.basic_lr}")
    LOGGER.info(f"  clip_lr_visual : {optimizer.clip_lr_visual}")
    LOGGER.info(f"  clip_lr_text : {optimizer.clip_lr_text}")
    LOGGER.info(f"  new_lr : {optimizer.new_lr}")
    LOGGER.info(f"  new_params_name: {optimizer.new_params_name}")

    return model, optimizer

def zero_shot_evaluation(model, test_loader, opts):
    eval_log = validate(model, test_loader, opts, global_step=0, total_step=opts.num_train_steps)
    if dist.get_rank()==0:  
        for task_name, val_log in eval_log.items():
            for eval_name, metric in val_log.items():
                eval_name = task_name +'_' +eval_name 
                # TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                #                 for k, v in metric.items() if not isinstance(v,str)})
                LOGGER.info(f"====-zero-shot evaluation--{eval_name}====== beam-size = {opts.beam_size} ==\n")
                LOGGER.info(metric)


def get_best_name(eval_name, metric):
    if eval_name.startswith('cap'):
        return 'CIDEr'
    elif eval_name.startswith('qa'):
        return 'accuracy'
    elif eval_name.startswith('ret'):
        if 'video_r1' in metric:
            return 'video_r1'

    elif eval_name.startswith('pt'):
        return None 
        
    else:
        raise NotImplementedError




def conduct_train(model, optimizer, train_loader, val_loaders, LOGGER, TB_LOGGER, opts, start_step=0, verbose_time=False):
  
    if dist.get_rank() == 0:
        pbar = tqdm(total=opts.num_train_steps, initial=start_step)
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpt'),remove_before_ckpt=opts.remove_before_ckpt)
    else:
        pbar = NoOp()
        model_saver = NoOp()
        
    loss_moving_averagetors ={}
    metric_logger_dict = defaultdict(dict)
    global_step = start_step
    n_gpu = dist.get_world_size()
    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)
    LOGGER.info(f"  Optim : {opts.optim}")
    LOGGER.info(f"  Scheduler : {opts.scheduler}")
    LOGGER.info(f"  Grad_norm : {opts.grad_norm}")
    LOGGER.info(f"  Warmup_ratio : {opts.warmup_ratio}")
    LOGGER.info(f"  Weight_decay: {opts.weight_decay}")
    best_indicator = {}

 
    ### training 
    for step, (name, batch) in enumerate(train_loader):
      
        ndata = train_loader.ndata
        task = name.split('--')[0]
        loss_dict = model(batch, task=task, compute_loss=True)
        loss = sum(list(loss_dict.values()))
        loss_dict['total_loss'] = loss
        loss_dict = {k:v.item() for k,v in loss_dict.items()}

        if opts.dataset_mix_type =='accum' :
            loss = loss / ndata
            delay_unscale = (step+1) % ndata != 0
        else:
            delay_unscale = False

        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
            
            scaled_loss.backward()
            if opts.checkpointing:
                works = []
                for p in model.parameters():
                # to speed it up, you can also organize grads to larger buckets to make allreduce more efficient
                    if p.grad is not None:
                        works.append(dist.all_reduce(p.grad, async_op=True))
                for work in works:
                    work.wait()

        if not name in loss_moving_averagetors:
                ### first time initialize 
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()
        ####accumulate loss

        for k,v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)
    
            
        if (opts.dataset_mix_type =='accum' and (step + 1) % ndata == 0) or opts.dataset_mix_type in ['round-robin','random']:
            global_step += 1
            # learning rate scheduling
            lr_ratio = get_lr_sched(global_step, opts)

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['init_lr'] * lr_ratio
                
            TB_LOGGER.add_scalar('lr_ratio', lr_ratio, global_step)
            TB_LOGGER.log_scaler_dict({name: averagetor.val
                                    for name, averagetor in loss_moving_averagetors.items()
                                    if averagetor.val is not None})

            if global_step % 200 == 0:    
                LOGGER.info({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()})                                   
            
            # update model params
            if opts.grad_norm != -1:
                grad_norm = clip_grad_norm_(amp.master_params(optimizer), opts.grad_norm)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)



        if (global_step+1) % opts.valid_steps == 0:
            eval_log = validate(model, val_loaders, opts, global_step, opts.num_train_steps)

            if dist.get_rank() == 0:
                for task_name, val_log in eval_log.items():
                    for eval_name, metric in val_log.items():
                        eval_name = task_name +'_' +eval_name 
                        metric_logger_dict[eval_name][str(global_step)] = metric
                        TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                                            for k, v in metric.items() if not isinstance(v,str)})
                        LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--======= beam-size = {opts.beam_size} =====\n")
                        LOGGER.info(metric)
                        best_name = get_best_name(eval_name, metric)
                        if best_name is not None:
                            if ('best_step' not in metric_logger_dict[eval_name]) or \
                                    (metric[best_name] >= metric_logger_dict[eval_name]['best_value']):
                                metric_logger_dict[eval_name]['best_step'] = global_step
                                metric_logger_dict[eval_name]['best_value'] = metric[best_name]
                                best_indicator[eval_name] = True 
                            else:
                                best_indicator[eval_name] = False 
                            best_step = metric_logger_dict[eval_name]['best_step']
                            LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}===== beam-size = {opts.beam_size} ===\n")
                            LOGGER.info(metric_logger_dict[eval_name][str(best_step)])          
                
                model_saver.save(model, global_step, optimizer,best_indicator, opts.save_best)
        TB_LOGGER.step()

        if global_step >= opts.num_train_steps:
            break
    pbar.close()









def compute_video_sample_num(opts):
    data_cfg = opts.data_cfg.train
    video_sample_num_ls=[]
    for d_cfg in data_cfg: 
        video_sample_num = d_cfg.get('video_sample_num',1)
        video_sample_num_ls.append(video_sample_num * opts.concatenated_nums)
    
    opts.train_video_sample_num = max(video_sample_num_ls)

    assert opts.train_video_sample_num  > 0
    
def create_train_dataloaders(opts, tokenizer):
    data_cfg = opts.data_cfg.train
    dataloaders = []
    dataloaders_dict={}
    train_steps = []
    loader_names = []
    scorer = None
    for d_cfg in data_cfg:
       
        dataset_ls = [] 
        use_sampler = True
        name = d_cfg['name']
      
        assert d_cfg['datatype'] in ['video','image']
        data_type = d_cfg['datatype'] + '_' + name
        video_mapper = None 
        task = d_cfg['task'].split('_') 
        video_path = d_cfg['video']
        video_sample_num = d_cfg['video_sample_num'] if data_type.startswith('video') else 1
        video_transforms =  d_cfg.get('video_transforms','none')
        data_format =  getattr(d_cfg,'data_format', 'frame')
        video_mapper = VideoMapper(video_path, opts, data_type, video_sample_num, video_transforms, data_format)
        dataset = COSADataset(d_cfg['txt'], video_mapper, opts, training=True)
        collate_fn = cosa_collate
        dataset.data_type = data_type
        LOGGER.info("Create Dataset {} Success".format(name))
        dataset_ls.append(dataset)


        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 

        if 'steps' in d_cfg:
            train_steps.append(d_cfg['steps'])
        elif 'epoch' in d_cfg:
            epoch = d_cfg['epoch']
            train_steps.append(int((len(dataset) // batch_size) * epoch))
        else:
            assert opts.dataset_mix_type in ['accum','round-robin']
            train_steps.append(None)
        loader = build_dataloader(dataset, collate_fn, True, batch_size, n_workers, use_sampler)
        dataloaders.append(loader)
        loader_names.append(f'{task}--{name}')


        if opts.scst_finetuning:   #### create scorer for scst finetuning, must only have one train dataset.
            assert len(data_cfg) == 1 
            scorer = Scorer(d_cfg['txt'], tokenizer)



    for i in range(len(dataloaders)):
        ratio = train_steps[i]
        dataloaders_dict[loader_names[i]] = (dataloaders[i], ratio)

    n_gpu = dist.get_world_size()
    for name, (loader, ratio) in dataloaders_dict.items():
        LOGGER.info(f" loader {name} , ratio {ratio} , bs_pergpu {loader.batch_size}, n_workers {loader.num_workers}" )

    if opts.dataset_mix_type == 'random' :
        meta_loader = MetaLoader(dataloaders_dict,
                                accum_steps=opts.gradient_accumulation_steps,
                                distributed=n_gpu > 1)
        
        if opts.num_train_steps == 0:
            total_train_steps = sum(train_steps)
            opts.num_train_steps = total_train_steps
    elif opts.dataset_mix_type in ['accum','round-robin']:
        assert opts.gradient_accumulation_steps == 1
        meta_loader = AccumMetaLoader(dataloaders_dict,
                                distributed=n_gpu > 1)
        
        
        
    meta_loader = PrefetchLoader(meta_loader)
    meta_loader.ndata = len(dataloaders_dict)
    opts.valid_steps = opts.num_train_steps // opts.valid_freq -1
    
 
    
    return meta_loader, scorer


def create_val_dataloaders(opts):
    data_cfg = opts.data_cfg.val
    dataloaders = {}
    
    for d_cfg in data_cfg:
        name = d_cfg['name']
        assert d_cfg['datatype'] in ['video','image']
        data_type = d_cfg['datatype'] + '_' + name
        

        
        use_sampler = True
        



        # task_short = [i.split('%')[1:] for i in d_cfg['task'].replace('pt_','').split('_')]
        # task_short = [j for i in task_short for j in i]
        # task_short = ''.join(task_short)

        task = d_cfg['task'].split('_') 
        video_path = d_cfg['video']
        video_sample_num = d_cfg['video_sample_num'] if data_type.startswith('video') else 1
        video_transforms =  d_cfg.get('video_transforms','none')
        data_format =  d_cfg.get('data_format','frame')
        video_mapper = VideoMapper(video_path, opts, data_type, video_sample_num, video_transforms, data_format)
        dataset = COSADataset(d_cfg['txt'], video_mapper, opts, training=False)
        collate_fn = cosa_collate
    

        if 'qa' in task:
            dataset.make_submission = d_cfg.get('make_submission', False) #### for vqav2

        if 'cap' in task:
            dataset.annfile = d_cfg['annfile']



        dataset.data_type = data_type
        dataset.name = name
        LOGGER.info("Create Dataset {} Success".format(name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 
        loader = build_dataloader(dataset, collate_fn, False, batch_size, n_workers, use_sampler)
        task_name = f'{task}--{name}'
        dataloaders[task_name] = PrefetchLoader(loader)

    return dataloaders


def build_dataloader(dataset, collate_fn, is_train, batch_size, n_workers=None, use_sampler=True):
    batch_size = batch_size // dist.get_world_size()
    if use_sampler:
        if is_train:
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedSampler_wopadding(dataset)
        loader = DataLoader(dataset, sampler = sampler, batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train)
    else:

        loader = DataLoader(dataset,  batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train)    

    return loader

def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_resolution", default=224, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output_dir", default='output/', type=str)
    parser.add_argument("--pretrain_step", default=None, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--clip_lr", default=5e-7, type=float)
    parser.add_argument("--clip_lr_text", default=5e-7, type=float)
    parser.add_argument("--optim", default='adam', choices=['adam', 'adamax', 'adamw'])
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+')
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--grad_norm", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument('--resume', action = 'store_true', help='use txt out')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument('--config')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--scheduler', type=str, default='warmup_linear')
    parser.add_argument("--concatenated_nums", type=int, default=1)
    parser.add_argument("--concatenated_type", type=str, default='random')
    parser.add_argument("--max_generation_len", type=int, default=40)
    parser.add_argument("--concatenated_raw_pixels", type=str2bool, default=True)
    parser.add_argument("--amp", type=str, default='apex')
    parser.add_argument("--train_id", type=str, default='')
    parser.add_argument("--test_id", type=str, default='')
    parser.add_argument("--train_task", type=str, default='')
    parser.add_argument("--test_task", type=str, default='')
    parser.add_argument("--test_batch_size", type=int, default=-1)
    parser.add_argument("--max_text_tokens", type=int, default=40)
    parser.add_argument("--train_batch_size", type=int, default=-1)
    parser.add_argument("--checkpointing", type=str2bool, default=False)
    parser.add_argument("--frozen_vision", type=str2bool, default=False)
    parser.add_argument("--scst_finetuning", type=str2bool, default=False)
    parser.add_argument("--itm_rerank_num", type=int, default=50)
    parser.add_argument("--itm_ratio", type=float, default=1.0)
    parser.add_argument("--save_best", type=str2bool, default=False)
    parser.add_argument("--train_epoch", type=float, default=-1)
    parser.add_argument("--train_steps", type=int, default=-1)
    parser.add_argument("--train_video_sample_num", type=int, default=-1)
    parser.add_argument("--test_video_sample_num", type=int, default=-1)
    parser.add_argument('--video_encoder_type', type=str, default='clip_vit_base_16')
    parser.add_argument('--video_transforms', type=str, default='none')
    parser.add_argument('--multimodal_encoder_type', type=str, default='bert_base_uncased')
    parser.add_argument('--num_train_steps', type=int, default=0)
    parser.add_argument('--pretrain_dir', type=str, default=None)          
    parser.add_argument('--dual_softmax', type=str2bool, default=False)
    parser.add_argument('--evaluate_ret_text', type=str2bool, default=False)
    parser.add_argument('--first_eval', type=str2bool, default=True)
    parser.add_argument('--remove_before_ckpt', type=str2bool, default=True)
    parser.add_argument('--dataset_mix_type', type=str, default='random')
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--new_params_name', type=str, default=[], nargs='+') 
    parser.add_argument('--new_lr', type=float, default=0.0)  
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--beam_size_qa', type=int, default=1)
    parser.add_argument('--contra_dim', type=int, default=512)
    
    args, input_args = parse_with_config(parser)

    return args, input_args