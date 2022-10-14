

work_dir=$(realpath $(dirname "$0")/..)
echo 'work_dir:' $work_dir
pkg_dir=$work_dir/pkgs 
mkdir -p $pkg_dir
cd $pkg_dir

# fairseq 
echo 'intall fairseq ...'
if [ ! -d fairseq ]; then 
    git clone https://github.com/facebookresearch/fairseq.git 
else 
    echo 'fairseq exits'
fi 
cd fairseq
pip install --editable ./

# other pip libs 
pip install fastBPE sacremoses subword_nmt tensorboardX overrides omegaconf
