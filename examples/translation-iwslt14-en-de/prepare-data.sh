
if [ $# -ne 1 ]; then 
    echo 'Args: save_dir'
    echo 'Example: local/iwslt14'
    exit 1 
fi 

# 
work_dir=$(realpath $(dirname "$0")/../..)
source $work_dir/scripts/env.sh
save_dir="$(realpath $1)"
script="$work_dir/pkgs/fairseq/examples/translation/prepare-iwslt14.sh"
TEXT=$save_dir/iwslt14.tokenized.de-en
echo 'work_dir:' $work_dir
echo 'script:' $script

# Step 1, download iwslt14 using fairseq
mkdir -p $save_dir
cd $save_dir
if [ ! -d $TEXT ]; then 
    echo 'downloading ...'
    bash $script &> download.log
    echo 'download log path:' $save_dir/download.log 
else
    echo "$TEXT exists, no downloading"
fi
ls -l . 

# Step 2, preprocess/binarize the data
echo "Binarize data at $TEXT ..."
for src_lang in en de; do 
    if [ $src_lang = en ]; then 
        tgt_lang=de
    else
        tgt_lang=en
    fi 
    echo 'process src_lang:' $src_lang 'tgt_lang:' $tgt_lang
    prepro_dir=data-bin-$src_lang-$tgt_lang-raw
    mkdir -p $prepro_dir
    # python -u $FAIRSEQ_CLI/preprocess.py --source-lang $src_lang --target-lang $tgt_lang --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir $prepro_dir --workers 6 --dataset-impl raw
    if [ $src_lang = en ]; then 
        convert_dir=data-converted-$src_lang-$tgt_lang-raw
        mkdir -p $convert_dir
        echo 'convert dataset to' $convert_dir
        python -u $WORK_DIR/scripts/fairseq-raw-to-simpledl.py $prepro_dir $convert_dir
    fi 
done
ls -l . 
echo 'fairseq datasets and converted dirs'
ls -lh $save_dir/data-*-raw
