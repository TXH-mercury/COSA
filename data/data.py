"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""

from cProfile import label
import json
from toolz.sandbox import unzip
import torch
from torch.utils.data import Dataset

from torchvision.transforms.transforms import *
from torchvision import transforms
import random
from os.path import join 

import os

from PIL import Image
from utils.logger import LOGGER
import ipdb
import matplotlib.pyplot as plt
import string 
from time import time
from typing import List, Tuple, Optional, Dict
from torch import Tensor
import torch.nn.functional as F
punctuation = string.punctuation
import numpy as np
from torchvision.transforms import functional as transform_F

import io
import pickle

class VideoMapper(object):
    def __init__(self, video_dir, opts, data_type = 'video', sample_num = 4, video_transforms='none',data_format='frame'):
        self.video_dir = video_dir
        self.datatype = data_type
        self.frame_syncaug = True
        self.training = True
        self.sample_num = sample_num 
        self.data_format = data_format
        
        self.resolution = opts.video_resolution

        if opts.video_encoder_type.startswith('clip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
        else:       
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
        
        LOGGER.info(f'{data_type} mean : {self.mean}')
        LOGGER.info(f'{data_type} std : {self.std}')      
        
        self.video_transforms = video_transforms
        if video_transforms == 'none':
            self.train_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
                
            self.test_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
        elif video_transforms == 'crop_flip':
            self.train_transforms = transforms.Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                        RandomHorizontalFlip(),
                                                        Normalize(self.mean,self.std)])

            self.test_transforms = transforms.Compose([Resize(self.resolution),
                                    CenterCrop(self.resolution),
                                    Normalize(self.mean,self.std)])
                                    
        else:
            raise NotImplementedError

        LOGGER.info(f'{data_type} video_transforms : {video_transforms} ')    
            
    def read(self, id_):
      
        if  self.datatype.startswith('video'):

            video_pixels = []        
            sample_num = self.sample_num
            
            try:

                
                if self.data_format == 'frame':
  

                    frame_path = os.path.join(self.video_dir, str(id_))
                    frames = os.listdir(frame_path)
                    frames.sort()   ### ['img_0001.jpg','img_0002.jpg',...]
                
                    sample_num = self.sample_num
                    frames_splited = split(frames,sample_num)    
                    if self.training:
                        sample_idx = [random.choice(i) for i in frames_splited]
                    else:
                        sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]
                    for i in range(sample_num):
                        frame = Image.open(os.path.join(frame_path,sample_idx[i]))
                        frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                        video_pixels.append(frame.unsqueeze(0))


                    video_pixels = torch.cat(video_pixels,dim=0)   ### nX3xHxW
                    if self.training:
                        video_pixels = self.train_transforms(video_pixels)    
                    else:
                        video_pixels = self.test_transforms(video_pixels)     
                    return video_pixels


                else:
                    raise NotImplementedError

            except Exception as e:
                print(e)
                print(id_)
                return None



        elif self.datatype.startswith('image'):
            
            try:
              
                img_path = os.path.join(self.video_dir, id_)
                if not os.path.exists(img_path):
                    img_path += '.jpg'
                if not os.path.exists(img_path):
                    img_path =  img_path.replace('.jpg','.JPEG')

                img = Image.open(img_path)
                img = img.convert('RGB')  #### convert 1-channel gray image and 4-channel CMYK image to RGB image
                img = transforms.ToTensor()(img)
                if self.training:    
                    img = self.train_transforms(img)
                else:
                    img = self.test_transforms(img)
                img = img.unsqueeze(0)
                return img

            except Exception as e:
                print(e)
                return None

        else:
            raise NotImplementedError()

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]




class COSADataset(Dataset):
    def __init__(self, desc, video_mapper, opts, training):
        
        self.video_mapper = video_mapper
        if self.video_mapper is not None:
            self.video_mapper.training = training 
        self.annos = json.load(open(desc))
 

        self.idx = list(range(len(self.annos)))
        if self.video_mapper is not None:
            self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        else:
            self.dataset_name = 'none'
        self.training = training
     
        
        
    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
  
        anno = self.annos[i]
        id_ = anno['video_id']
        raw_sentences = [None]
        question_id = [None]
        question = [None]
        answer = [None]
        if 'desc' in anno or 'caption' in anno:
            raw_sentences = anno['desc'] if 'desc' in anno else anno['caption'] 
            if isinstance(raw_sentences, list):
                pass ## testing
            else:
                raw_sentences=[raw_sentences]
            num_samples = len(raw_sentences)
        if 'question' in anno:
            if self.training:

                question = [anno['question']]
                if isinstance(anno['answer'],list): #### vqav2
                    answer = [random.choice(anno['answer'])]
                else:
                    answer = [anno['answer']]

            else:
                question = anno['question']
                answer = anno['answer']
                if 'question_id' in anno:
                    question_id = anno['question_id']
                if isinstance(answer[0],list):
                    answer = [random.choice(ans) for ans in answer]          
            num_samples = len(question)
        id_txt = [id_] * num_samples
        video_pixels = self.video_mapper.read(id_)
        if video_pixels is None: ###wrong img/video and needs to resample 
            resample_idx = random.choice(self.idx)
            LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
            return self.__getitem__(resample_idx)
        return id_, raw_sentences, video_pixels, id_txt, num_samples, question, answer, question_id




def cosa_collate(inputs):
    

    (ids, raw_sentences, video_pixels, ids_txt, num_samples, questions, answers, question_ids) = map(list, unzip(inputs))

    ids_txt = [ j  for i in ids_txt for j in i]
    raw_sentences = [ j  for i in raw_sentences for j in i]
    questions = [ j  for i in questions for j in i]
    answers = [ j  for i in answers for j in i]
    question_ids = [ j  for i in question_ids for j in i]
    video_pixels = torch.stack(video_pixels, dim=0)
    batch =   {'ids': ids,
             'raw_sentences': raw_sentences,
             'video_pixels': video_pixels,
             'ids_txt': ids_txt,
             'sample_num': num_samples,
             'raw_questions': questions,
             'raw_answers': answers,
             'question_ids': question_ids}
    
    return batch

    
