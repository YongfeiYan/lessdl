
# 

if [ $# -ne 5 ]; then 
    echo 'args: data_dir exp_dir cuda_device src_lang tgt_lang'
    exit 
fi 

data_dir="$1"
exp_dir="$2"
export CUDA_VISIBLE_DEVICES="$3"
src_lang="$4"
tgt_lang="$5"
export PYTHONPATH=./
# keep scripts
cp "$0" $exp_dir

echo 'beam search ...'
python -u scripts/beam_search_predict.py \
    --exp-dir $exp_dir --predictor beam_search --beam-size 10 \
    &> $exp_dir/eval.log

echo 'tail of eval.log'
tail $exp_dir/eval.log

BLEU=examples/translation-iwslt14-en-de/multi-bleu.perl

echo 'calculating bleu ...'
cat $exp_dir/eval.log | egrep 'beam-0' | cut -d '-' -f 2- | sort -t '-' -nk1 | cut -d $'\t' -f 2 > $exp_dir/predict.$tgt_lang
sed 's/@@ //g' $data_dir/test.$tgt_lang | sed 's/ @@//g' > $exp_dir/test.$tgt_lang.tok
sed 's/@@ //g' $exp_dir/predict.$tgt_lang | sed 's/ @@//g' > $exp_dir/predict.$tgt_lang.tok
$BLEU $exp_dir/test.$tgt_lang.tok < $exp_dir/predict.$tgt_lang.tok > $exp_dir/bleu-res.$tgt_lang
echo 'BLEU results:'
tail -n 3 $exp_dir/bleu-res.$tgt_lang
