export WANDB_API_KEY=""
export WANDB_PROJECT=""
export HF_TOKEN=

accelerate launch ./train_vlm.py \
    --dataset_name JMandy/nuscenes_qa_mini_day \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --logging_steps 1 \
    --optim adamw_bnb_8bit \
    --completion_only_loss true \
    --gradient_checkpointing true \
    --lr_scheduler_type 'cosine' \
    --learning_rate 5e-5 \
    --output_dir ./output/qwen3_vl_8b_instruct_lora \
    --save_strategy epoch \
    --dtype bfloat16 \
    --report_to wandb \
    --run_name qwen3_vl_8b_instruct_lora \
    --attn_implementation kernels-community/flash-attn3 \
    # --use_peft \
    # --lora_target_modules all-linear \
    # --lora_modules_to_save "embed_tokens" "lm_head" "patch_embed" "pos_embed" \
    # --lora_r 256 \
    # --lora_alpha 128 


