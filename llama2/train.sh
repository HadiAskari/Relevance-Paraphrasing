CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 create_paraphrase.py\

torchrun --nproc_per_node 2 inference_generate_summary.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 8000 --max_batch_size 1
