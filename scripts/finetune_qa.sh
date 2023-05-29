output_dir=$1




#### QA
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--learning_rate 2e-5 \
--train_video_sample_num 4 \
--test_video_sample_num 4 \
--checkpointing true \
--first_eval false \
--save_best true \
--pretrain_dir $output_dir \
--config ./config/VQA-tgif-frame.json \
--output_dir  $output_dir/VQA-tgif \

python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 8 \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--config ./config/VQA-msvd.json \
--pretrain_dir $output_dir \
--save_best true \
--output_dir $output_dir/VQA-msvd \



python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./train.py \
--train_video_sample_num 8 \
--test_video_sample_num 8 \
--learning_rate 2e-5 \
--checkpointing true \
--beam_size 1 \
--first_eval false \
--config ./config/VQA-msrvtt.json \
--save_best true \
--pretrain_dir $output_dir \
--output_dir $output_dir/vqa-msrvtt \







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
--save_best true \
--config ./config/VQA-activitynet.json \
--first_eval false \
--pretrain_dir $output_dir \
--output_dir $output_dir/VQA-activitynet \

python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9824 \
./train.py \
--learning_rate 2e-5 \
--config ./config/VQA-vqav2_3129.json \
--pretrain_dir $output_dir \
--first_eval false \
--video_resolution 384 \
--valid_freq 1 \
--output_dir $output_dir/VQA-vqav2_3129 \



