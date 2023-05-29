output_dir=$1





#### caption

python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 16 \
--learning_rate 3e-5 \
--checkpointing true \
--config ./config/caption-youcook.json \
--pretrain_dir $output_dir \
--save_best true \
--output_dir $output_dir/caption-youcook \

python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8  \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 8 \
--learning_rate 1e-5 \
--checkpointing true \
--config ./config/caption-msvd.json \
--pretrain_dir $output_dir \
--save_best true \
--output_dir  $output_dir/caption-msvd \







python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9634 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 8 \
--learning_rate 2e-5 \
--train_batch_size 128 \
--train_epoch 10 \
--checkpointing true \
--save_best true \
--config ./config/caption-msrvtt.json \
--pretrain_dir $output_dir \
--output_dir  $output_dir/caption-msrvtt \

python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 8 \
--learning_rate 3e-5 \
--checkpointing true \
--train_epoch 40 \
--save_best true \
--config ./config/caption-tv.json \
--pretrain_dir $output_dir \
--output_dir $output_dir/caption-tv \




python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 20 \
--learning_rate 2e-5 \
--checkpointing true \
--config ./config/caption-vatex.json \
--pretrain_dir $output_dir \
--output_dir  $output_dir/caption-vatex \
--save_best true \



####scst for vatex 

python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 20 \
--learning_rate 7e-6 \
--checkpointing true \
--config ./config/caption-vatex.json \
--pretrain_dir $output_dir \
--fp16 false \
--scst_finetuning true \
--train_epoch 5 \
--output_dir  $output_dir/caption-vatex/scst \
--save_best true \
--first_eval true \
#--checkpoint output_dir/caption-vatex/ckpt/best.pth



python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--learning_rate 1e-5 \
--config ./config/caption-mscoco.json \
--pretrain_dir $output_dir \
--video_resolution 480 \
--output_dir $output_dir/caption-mscoco \
--checkpointing false \
--save_best true \

#### scst for coco


python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--learning_rate 2.5e-6 \
--config ./config/caption-mscoco.json \
--pretrain_dir $output_dir \
--video_resolution 480 \
--fp16 false \
--scst_finetuning true \
--train_epoch 2.5 \
--output_dir $output_dir/caption-mscoco/scst \
--checkpointing false \
--save_best true \
