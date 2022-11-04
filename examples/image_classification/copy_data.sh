
# chose one image each class to form debugging dataset
src_dir=local/data/imagenet 
dst_dir=local/data/imagenet-test
mkdir -p $dst_dir 
for t in train val; do 
    for d in `ls $src_dir/$t`; do 
        mkdir -p $dst_dir/$t/$d
        for f in `ls $src_dir/$t/$d | head -n 1`; do 
            echo "cp $src_dir/$t/$d/$f $dst_dir/$t/$d"
            cp $src_dir/$t/$d/$f $dst_dir/$t/$d
        done 
    done 
done
