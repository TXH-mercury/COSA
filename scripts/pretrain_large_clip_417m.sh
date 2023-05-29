output_dir=./output/$config_name

# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./train.py \
# --config ./config/$config_name.json \
# --output_dir $output_dir \
# --checkpointing true 

sh scripts/finetune_ret.sh $output_dir
sh scripts/finetune_cap.sh $output_dir
sh scripts/finetune_qa.sh $output_dir