source activate
conda activate newqiu
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/fusion_bench:$PYTHONPATH

#################################################### Sequential-ft #################################################################

# continual ft
fusion_bench \
    fabric.loggers.root_dir=outputs/continual_finetune \
    fabric.loggers.name=vit-b-16-continual_finetune \
    fabric.loggers.version=0 \
    method=classification/continual_clip_finetune \
    method.shuffle_order=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_continual_finetune \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16

# convert seq ft MTIL
datasets=(aircraft caltech101 cifar100 dtd eurosat oxford_flowers102 food101 mnist oxford-iiit-pet stanford-cars sun397)

for i in {0..10}; do
    task=${datasets[$i]}
    echo "Processing $i: $task"
    python fusion_bench/scripts/clip/convert_checkpoint.py \
        --checkpoint outputs/continual_finetune/vit-b-16-continual_finetune/version_0/checkpoints/task=${i}_step=3999.ckpt \
        --output model_convert/clip-vit-base-patch16_${task}_seq \
        --model openai/clip-vit-base-patch16
done

# seq magmax MTIL
fusion_bench \
    fabric.loggers.root_dir=outputs/magmax \
    fabric.loggers.name=vit-b-16-MTIL_seq \
    fabric.loggers.version=0 \
    method=mingle/magmax \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11_seq \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16



# seq mingle MTIL
fusion_bench \
    fabric.loggers.root_dir=outputs/mingle \
    fabric.loggers.name=vit-b-16-MTIL_seq \
    fabric.loggers.version=0 \
    method=mingle/mingle_seq \
    method.max_steps=500 \
    method.gamma=0 \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11_seq \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16


#################################################### Independent-ft #################################################################


# finetune aircraft

fusion_bench \
    fabric.loggers.root_dir=model_save \
    fabric.loggers.name=clip-vit-base-patch16_aircraft \
    method=clip_finetune \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_aircraft \
    taskpool=CLIPVisionModelTaskPool/clip-vit-single-task_aircraft \
    taskpool.base_model=openai/clip-vit-base-patch16


fusion_bench \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_aircraft \
    taskpool=CLIPVisionModelTaskPool/clip-vit-single-task_aircraft \
    taskpool.base_model=openai/clip-vit-base-patch16 


#convert model
python fusion_bench/scripts/clip/convert_checkpoint.py \
    --checkpoint model_save/clip-vit-base-patch16_aircraft/version_0/checkpoints/step=3999.ckpt \
    --output model_convert/clip-vit-base-patch16_aircraft \
    --model openai/clip-vit-base-patch16



# finetune caltech101
fusion_bench \
    fabric.loggers.root_dir=model_save \
    fabric.loggers.name=clip-vit-base-patch16_caltech101 \
    method=clip_finetune \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_caltech101 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-single-task_caltech101 \
    taskpool.base_model=openai/clip-vit-base-patch16

fusion_bench \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_caltech101 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-single-task_caltech101 \
    taskpool.base_model=openai/clip-vit-base-patch16

#convert model
python fusion_bench/scripts/clip/convert_checkpoint.py \
    --checkpoint model_save/clip-vit-base-patch16_caltech101/version_0/checkpoints/step=3999.ckpt \
    --output model_convert/clip-vit-base-patch16_caltech101 \
    --model openai/clip-vit-base-patch16


# test MTIL

fusion_bench \
    fabric.loggers.root_dir=outputs/weight_average \
    fabric.loggers.name=vit-b-16-MTIL \
    fabric.loggers.version=0 \
    method=mingle/weight_average \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16


fusion_bench \
    fabric.loggers.root_dir=outputs/continual_task_arithmetic \
    fabric.loggers.name=vit-b-16-MTIL \
    fabric.loggers.version=0 \
    method=mingle/task_arithmetic \
    method.scaling_factor=0.1 \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16


fusion_bench \
    fabric.loggers.root_dir=outputs/continual_ties_merging \
    fabric.loggers.name=vit-b-16-MTIL \
    fabric.loggers.version=0 \
    method=mingle/ties_merging \
    method.scaling_factor=0.1 \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16


fusion_bench \
    fabric.loggers.root_dir=outputs/magmax \
    fabric.loggers.name=vit-b-16-MTIL \
    fabric.loggers.version=0 \
    method=mingle/magmax \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16


fusion_bench \
    fabric.loggers.root_dir=outputs/opcm \
    fabric.loggers.name=vit-b-16-MTIL \
    fabric.loggers.version=0 \
    method=mingle/opcm \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16



fusion_bench \
    fabric.loggers.root_dir=outputs/mingle \
    fabric.loggers.name=vit-b-16-MTIL \
    fabric.loggers.version=0 \
    method=mingle/mingle \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    method.max_steps=50 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_MTIL11 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_MTIL11 \
    taskpool.base_model=openai/clip-vit-base-patch16

