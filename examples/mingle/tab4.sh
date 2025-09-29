source activate
conda activate newqiu
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/fusion_bench:$PYTHONPATH



for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
    for version in {0..0}; do
        fusion_bench \
        fabric.loggers.root_dir=outputs/c_adamerging \
        fabric.loggers.name=vit-b-32_${corruption} \
        fabric.loggers.version=${version} \
        method.seed=$((42 + version)) \
        method=mingle/c_adamerging \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
        taskpool=clip-vit-base-patch32_robustness_corrupted \
        taskpool.base_model=openai/clip-vit-base-patch32 \
        taskpool.corruption=${corruption}
    done
done

for version in {0..0}; do
    fusion_bench \
    fabric.loggers.root_dir=outputs/c_adamerging \
    fabric.loggers.name=vit-b-32_clean \
    fabric.loggers.version=${version} \
    method.seed=$((42 + version)) \
    method=mingle/c_adamerging \
    method.shuffle_order=true \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
    taskpool=clip-vit-base-patch32_robustness_corrupted \
    taskpool.base_model=openai/clip-vit-base-patch32 \
    taskpool.corruption=test
done



for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
    for version in {0..0}; do
        fusion_bench \
        fabric.loggers.root_dir=outputs/c_wemoe \
        fabric.loggers.name=vit-b-32_${corruption} \
        fabric.loggers.version=${version} \
        method.seed=$((42 + version)) \
        method=mingle/c_wemoe \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
        taskpool=clip-vit-base-patch32_robustness_corrupted \
        taskpool.base_model=openai/clip-vit-base-patch32 \
        taskpool.corruption=${corruption}
    done
done

for version in {0..0}; do
    fusion_bench \
    fabric.loggers.root_dir=outputs/c_wemoe \
    fabric.loggers.name=vit-b-32_clean \
    fabric.loggers.version=${version} \
    method.seed=$((42 + version)) \
    method=mingle/c_womoe \
    method.shuffle_order=true \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
    taskpool=clip-vit-base-patch32_robustness_corrupted \
    taskpool.base_model=openai/clip-vit-base-patch32 \
    taskpool.corruption=test
done


for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
    for version in {0..0}; do
        fusion_bench \
        fabric.loggers.root_dir=outputs/continual_task_arithmetic \
        fabric.loggers.name=vit-b-32_${corruption} \
        fabric.loggers.version=${version} \
        method.seed=$((42 + version)) \
        method=mingle/task_arithmetic \
        method.scaling_factor=0.3 \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
        taskpool=clip-vit-base-patch32_robustness_corrupted \
        taskpool.base_model=openai/clip-vit-base-patch32 \
        taskpool.corruption=${corruption}
    done
done

for version in {0..0}; do
    fusion_bench \
    fabric.loggers.root_dir=outputs/continual_task_arithmetic \
    fabric.loggers.name=vit-b-32_clean \
    fabric.loggers.version=${version} \
    method.seed=$((42 + version)) \
    method=mingle/task_arithmetic \
    method.scaling_factor=0.3 \
    method.shuffle_order=true \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
    taskpool=clip-vit-base-patch32_robustness_corrupted \
    taskpool.base_model=openai/clip-vit-base-patch32 \
    taskpool.corruption=test
done


for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
    for version in {0..0}; do
        fusion_bench \
        fabric.loggers.root_dir=outputs/magmax \
        fabric.loggers.name=vit-b-32_${corruption} \
        fabric.loggers.version=${version} \
        method.seed=$((42 + version)) \
        method=mingle/magmax \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
        taskpool=clip-vit-base-patch32_robustness_corrupted \
        taskpool.base_model=openai/clip-vit-base-patch32 \
        taskpool.corruption=${corruption}
    done
done

for version in {0..0}; do
    fusion_bench \
    fabric.loggers.root_dir=outputs/magmax \
    fabric.loggers.name=vit-b-32_clean \
    fabric.loggers.version=${version} \
    method.seed=$((42 + version)) \
    method=mingle/magmax \
    method.shuffle_order=true \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
    taskpool=clip-vit-base-patch32_robustness_corrupted \
    taskpool.base_model=openai/clip-vit-base-patch32 \
    taskpool.corruption=test
done


for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
    for version in {0..0}; do
        fusion_bench \
        fabric.loggers.root_dir=outputs/opcm \
        fabric.loggers.name=vit-b-32_${corruption} \
        fabric.loggers.version=${version} \
        method.seed=$((42 + version)) \
        method=mingle/opcm \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
        taskpool=clip-vit-base-patch32_robustness_corrupted \
        taskpool.base_model=openai/clip-vit-base-patch32 \
        taskpool.corruption=${corruption}
    done
done

for version in {0..0}; do
    fusion_bench \
    fabric.loggers.root_dir=outputs/opcm \
    fabric.loggers.name=vit-b-32_clean \
    fabric.loggers.version=${version} \
    method.seed=$((42 + version)) \
    method=mingle/opcm \
    method.shuffle_order=true \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
    taskpool=clip-vit-base-patch32_robustness_corrupted \
    taskpool.base_model=openai/clip-vit-base-patch32 \
    taskpool.corruption=test
done



for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
    for version in {0..0}; do
        fusion_bench \
        fabric.loggers.root_dir=outputs/mingle \
        fabric.loggers.name=vit-b-32_${corruption} \
        fabric.loggers.version=${version} \
        method.seed=$((42 + version)) \
        method=mingle/mingle \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
        taskpool=clip-vit-base-patch32_robustness_corrupted \
        taskpool.base_model=openai/clip-vit-base-patch32 \
        taskpool.corruption=${corruption}
    done
done

for version in {0..0}; do
    fusion_bench \
    fabric.loggers.root_dir=outputs/mingle \
    fabric.loggers.name=vit-b-32_clean \
    fabric.loggers.version=${version} \
    method.seed=$((42 + version)) \
    method=mingle/mingle \
    method.shuffle_order=true \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
    taskpool=clip-vit-base-patch32_robustness_corrupted \
    taskpool.base_model=openai/clip-vit-base-patch32 \
    taskpool.corruption=test
done




