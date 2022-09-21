
#!/bin/bash
set -x 

if [ $# -ne 5 ]; then 
    echo "args: data_dir save_dir cuda_device src_lang tgt_lang"
    exit 1
fi

export PYTHONPATH=./
data_dir="$1"
exp_dir="$2"
export CUDA_VISIBLE_DEVICES="$3"
src_lang="$4"
tgt_lang="$5"
lr=5e-4
mkdir -p $exp_dir
# keep script
cp "$0" $exp_dir
# run 
python -u scripts/train.py \
    --exp-dir $exp_dir \
    --dataset translation_dataset --src-language $src_lang --tgt-language $tgt_lang \
    --data-dir "$data_dir" \
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
echo 'Runing jobs'
jobs
disown -a
