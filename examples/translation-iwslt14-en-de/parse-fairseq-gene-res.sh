
if [ $# -ne 2 ]; then
    echo 'args: results-path save-prefix'
    exit
fi

echo "results: $1"
echo "save prefix: $2"

egrep 'S-' "$1" | cut -d '-' -f 2- | sort -nk1 | cut -f2 > $2.src
egrep 'T-' "$1" | cut -d '-' -f 2- | sort -nk1 | cut -f2 > $2.tgt
egrep 'H-' "$1" | cut -d '-' -f 2- | sort -nk1 | cut -f3 > $2.hyp
# egrep 'T-' "$1" | tr '-' $'\t' | sort -nk2 | cut -f3 > $2.tgt
# egrep 'H-' "$1" | tr '-' $'\t' | sort -nk2 | cut -f4 > $2.hyp

echo 'save to:'
ls -lh $2.{src,tgt,hyp}
