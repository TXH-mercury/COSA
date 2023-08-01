# COSA: Concatenated Sample Pretrained Vision-Language Foundation Model

<div align=center><img src=COSA-sample.png width="75%" height="75%"></div>

This is the official repository of COSA which provide training and testing code, as well as pretraining checkpoints. 


## Building Environment
COSA is implemented based on Pytorch. We use pytorch-1.9.0 and cuda-11.1. Other version could be also compatible.

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- apex is needed. 
```
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- setup packages.
```
sh preinstall.sh
```
## Download Checkpoints
- [pretrained_weights](https://drive.google.com/file/d/1o3bOw1-tirRK0G7yqEWBPjfmUMUth-DI/view?usp=sharing) (BERT,CLIP,Swin).

Put pretrained_weights dir under main path. (COSA/pretrained_weights)

- [COSA-base-swin-5m](https://drive.google.com/file/d/1jaKFGbVE-BW3x5JUjRHbRqhVaXIy8q8s/view?usp=sharing).
- [COSA-base-swin-17m](https://drive.google.com/file/d/15LACWjLKD_Y7DCdvNRhqdc5MnwUBmcT7/view?usp=sharing).
- [COSA-large-clip-417m](https://drive.google.com/file/d/114taD5SwhQ5NQdEtIRDh-HDdybsfP0EU/view?usp=sharing).

Put them  under the output dir. (COSA/output/COSA-base-swin-5m)

## Prepare Datasets
COSA is pretrained and finetuned on multiple vision-language datasets. 
e.g. PRETRAIN: CC3M, WebVid-2.5M, CC4M, CC12M, LAION...
FiNETUNE: MSRVTT, MSVD, DiDeMo, LSMDC, ActivityNet, VATEX, YouCook2, TVC, TGIF, MSCOCO, Flickr30K, VQAV2...

The processed datasets folder is available at [here](https://drive.google.com/file/d/1fnd2rNFzgI7Pi-u3N0e5Z3zZjXP48988/view?usp=sharing), please download it and put it under the main directory (COSA/datasets). With regards to the vision part, you need to download raw images or videos of those datasets and extract frames (by default for fast traing, you can also use tools such as decord or av for online processing, and skip the this step but need to modify the dataset code).

using utils/extract_frame_and_wav_multiprocess.py  to extract frames.



- [Download all (pretrained_weight, all ckpts and datasets)](https://drive.google.com/drive/folders/1pNdr1D4S4cQ3-VKzcl3rkk_5232bMvgS?usp=share_link)

## Finetune  Model
- finetune retrieval tasks
```
sh scripts/finetune_ret.sh $pretrain_path(output/COSA-base-swin-5m)
```
- finetune captioning tasks
```
sh scripts/finetune_cap.sh $pretrain_path(output/COSA-base-swin-5m)
```
- finetune QA tasks
```
sh scripts/finetune_qa.sh $pretrain_path(output/COSA-base-swin-5m)
```
The finetuning output path will be the subdir of $pretrain_path

## Test Model
For example, the cmd for finetuning retrieval model  in scripts/finetune_ret.sh is as follows:

```
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 16 \
--learning_rate 2e-5 \
--config ./config/retrieval-msrvtt.json \
--pretrain_dir $output_dir \
--save_best true \
--checkpointing true \
--output_dir $output_dir/retrieval-msrvtt \
```

if you want to test model, just add following two rows to the cmd:
```
--zero_shot \
--checkpoint $checkpoint_save_path(.pt)
```
## Pretrain Model
```
sh scripts/pretrain_base_swin_5m.sh
```

## Citation

If you find this code useful for your research, please consider citing:


```
@article{chen2023cosa,
  title={COSA: Concatenated Sample Pretrained Vision-Language Foundation Model},
  author={Chen, Sihan and He, Xingjian and Li, Handong and Jin, Xiaojie and Feng, Jiashi and Liu, Jing},
  journal={arXiv preprint arXiv:2306.09085},
  year={2023}
}
```


## License

MIT -->
