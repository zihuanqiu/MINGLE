source activate
conda activate newqiu
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/fusion_bench:$PYTHONPATH


fusion_bench \
    fabric.loggers.root_dir=outputs/ta_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=mingle/task_arithmetic_nlp \
    method.shuffle_order=false \
    method.scaling_factor=0.3 \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true\
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 


fusion_bench \
    fabric.loggers.root_dir=outputs/ties_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=mingle/ties_merging_nlp \
    method.shuffle_order=false \
    method.scaling_factor=0.3 \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true\
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 


fusion_bench \
    fabric.loggers.root_dir=outputs/opcm_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=mingle/opcm_nlp \
    method.shuffle_order=false \
    method.scaling_factor=0.3 \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true\
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 


fusion_bench \
    fabric.loggers.root_dir=outputs/c_adamerging_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=mingle/c_adamerging_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true\
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 



fusion_bench \
    fabric.loggers.root_dir=outputs/c_lora_wemoe_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=mingle/c_wemoe_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true\
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 



fusion_bench \
    fabric.loggers.root_dir=outputs/mingle_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=mingle/mingle_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 
