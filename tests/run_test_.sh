
set -e 
source scripts/env.sh 

# Single test case 
# bash tests/run_test_.sh test_file 
# Test all case
# bash tests/run_test_.sh 

if [ $# -eq 1 ]; then 
    echo 'Test file:' $1
    PYTHONPATH=$PYTHONPATH python -u "$1"
else 
    for f in tests/*_test_.py; do 
        echo 'Test file: '$f
        echo $PYTHONPATH
        PYTHONPATH=$PYTHONPATH python -u "$f"
    done
fi 
echo 'Success'
