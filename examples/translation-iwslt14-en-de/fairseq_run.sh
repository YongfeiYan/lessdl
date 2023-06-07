#!/bin/bash
set -e

if [ $# -ne 4 ]; then 
    echo 'args: datadir cuda_device src_lang tgt_lang'
    exit 1
fi

# Training args: 
train="on"
evaluate="on"
src_lang="$3"
tgt_lang="$4"
epochs=200
patience=10
norm=0.0
lr=5e-4

# 
source scripts/env.sh
save_dir=local/data/exp/iwslt14/fairseq-$src_lang-$tgt_lang
root_dir=${WORK_DIR}/examples/translation-iwslt14-en-de
BLEU=$root_dir/multi-bleu.perl
prepro_dir="$1/data-bin-${src_lang}-${tgt_lang}-raw"
export CUDA_VISIBLE_DEVICES="$2"
TEXT="$1"/iwslt14.tokenized.de-en

# Training
if [ "$train" = "on" ]; then
    mkdir -p $save_dir
    echo "copy scripts $0 to $save_dir"
    cp "$0" $save_dir/
    echo 'begin to run.'
    python -u $FAIRSEQ_CLI/train.py  \
        $prepro_dir --dataset-impl raw \
        --save-dir $save_dir --log-format tqdm --seed 14 \
        --max-epoch $epochs --no-epoch-checkpoints --num-workers 0 --patience $patience \
        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm $norm \
        --lr $lr --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --max-tokens 4096 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        &> $save_dir/train.log
        # --max-sentences 40 \
        # --lr $lr \
else
    echo "no training"
fi


# Evaluate
if [ "$evaluate" = "on" ]; then
    echo 'evaluating ...'
    python -u $FAIRSEQ_CLI/generate.py \
        $prepro_dir --dataset-impl raw \
        --path $save_dir/checkpoint_best.pt \
        --beam 5 --remove-bpe \
        &> $save_dir/evaluate.log
    bash $root_dir/parse-fairseq-gene-res.sh $save_dir/evaluate.log $save_dir/res-eval
    # $BLEU $save_dir/res-eval.tgt < $save_dir/res-eval.hyp
    sed 's/@@ //g' $TEXT/test.$tgt_lang | sed 's/ @@//g' > $save_dir/test.$tgt_lang.tok
    $BLEU $save_dir/test.$tgt_lang.tok < $save_dir/res-eval.hyp &>> $save_dir/res.bleu  
    tail -n 2 $save_dir/evaluate.log >> $save_dir/res.bleu 
    echo 'BLEU results:'
    tail -n 3 $save_dir/res.bleu
else
    echo 'no evaluation'
fi

echo 'save_dir' $save_dir
