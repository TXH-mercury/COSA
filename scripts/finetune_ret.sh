output_dir=$1




#### retrieval
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



python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 16 \
--learning_rate 2e-5 \
--checkpointing true \
--config ./config/retrieval-vatex.json \
--pretrain_dir $output_dir \
--save_best true \
--output_dir $output_dir/retrieval-vatex \








python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 32 \     ##### if out of memory ,set 8 when training and run an additional test with 32
--learning_rate 2e-5 \
--checkpointing true \
--first_eval false \
--config ./config/retrieval-lsmdc.json \   
--pretrain_dir $output_dir \
--save_best true \
--output_dir $output_dir/retrieval-lsmdc \






python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 32 \
--learning_rate 2e-5 \
--checkpointing true \
--config ./config/retrieval-didemo.json \
--pretrain_dir $output_dir \
--save_best true \
--output_dir $output_dir/retrieval-didemo \



python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 32 \   ##### if out of memory ,set 8 when training and run an additional test with 32
--learning_rate 2e-5 \
--checkpointing true \
--config ./config/retrieval-activitynet.json \
--pretrain_dir $output_dir \
--output_dir $output_dir/retrieval-activitynet \
--save_best true \





python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--learning_rate 1e-5 \
--config ./config/retrieval-flickr.json \
--train_batch_size 256 \
--train_epoch 5 \
--pretrain_dir $output_dir \
--output_dir  $output_dir/retrieval-flickr  \
--video_resolution 384 \
--checkpointing true \
#--checkpoint $output_dir/retrieval-coco/ckpt/best.pt

