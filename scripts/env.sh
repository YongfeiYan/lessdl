
echo 'env.sh:' ${BASH_SOURCE}
WORK_DIR=$(realpath $(dirname "${BASH_SOURCE}")/..)
FAIRSEQ_CLI=$WORK_DIR/pkgs/fairseq/fairseq_cli
PYTHONPATH=$PYTHONPATH:$WORK_DIR:$WORK_DIR/pkgs/fairseq
BLEU=$WORK_DIR/examples/translation-iwslt14-en-de/multi-bleu.perl
