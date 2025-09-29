source activate
conda activate newqiu
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/fusion_bench:$PYTHONPATH


# clip-vit-base-patch32, 8 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.3 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.3 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32 
done


# clip-vit-base-patch32, 14 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32 
done

# clip-vit-base-patch32, 20 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32 
done




#######################################################################################################################################



# clip-vit-base-patch16, 8 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done



for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.3 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.3 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done



for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16 
done


# clip-vit-base-patch16, 14 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16 
done

# clip-vit-base-patch16, 20 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done



for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16 
done



#######################################################################################################################################



# clip-vit-large-patch14, 8 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done



for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.3 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.3 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14 
done


# clip-vit-large-patch14, 14 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14 
done

# clip-vit-large-patch14, 20 tasks

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/weight_average \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/weight_average \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/continual_ties_merging \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/ties_merging \
        method.scaling_factor=0.1 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/consensus_ta \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/consensus_ta \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/mingle_star \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=mingle/mingle \
        method.lora_layer="[attn,fc1]" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true\
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14 
done


