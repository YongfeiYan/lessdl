
#!/bin/bash
set -x 

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./
lr=5e-4
src_lang=de
tgt_lang=en
exp_dir=local/data/exp/iwslt14-$src_lang-$tgt_lang
mkdir -p $exp_dir
# 保存运行脚本
cp "$0" $exp_dir
# run 
python -u scripts/train.py \
    --exp-dir $exp_dir \
    --dataset translation_dataset --src-language $src_lang --tgt-language $tgt_lang \
    --data-dir local/data/iwslt14/data-converted-en-de-raw \
    --arch transformer_iwslt_de_en --dropout 0.3 \
    --max-batch-tokens 4096 --num-workers 0 --max-samples-in-memory 100000000 --sort-key '_size' --epochs 200 \
    --log-every-n-batches 500 --grad-norm 0 \
    --optimizer fairseq_adam,lr=$lr,weight_decay=0.0001,beta1=0.9,beta2=0.98 \
    --lr-scheduler inverse_sqrt,warmup_updates=4000,warmup_end_lr=$lr \
    --loss label_smoothed_cross_entropy --label-smoothing 0.1 \
    &> $exp_dir/train.log &

sleep 2
echo 'exp_dir :' $exp_dir
echo 'log file:' $exp_dir/train.log
# tail -f $exp_dir/train.log
echo 'Runing jobs'
jobs
disown -a
