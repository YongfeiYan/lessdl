# NeRF Models
This is a implementation of [NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields), which is in the procedure of simdltk and supports Multi-GPU training.
It is based on the code of [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch), and adds implementation of Dataset/NeRFModel and several changes of GPU training. 


## Requirements 
```
pip install -r examples/nerf/requirements.txt
```

## Data 
```
mkdir -p local && cd local
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
```

## Train 
Train on a single GPU
```bash 
bash examples/nerf/run_train.sh
```
Train on 8 GPUs
```bash
bash examples/nerf/run_traindpp.sh
```

## Results 
### Generated example
![Lego](logs/lego_res.gif)
### Training statistics
I tested the NeRF using Tesla V100, the training states of Lego datset are listed in the following

|       | Train Loss | Eval Loss | Epochs | Time/Epoch |
| :--:  | :--:       |    :--:   | :--:   |  :--:      |
| 1 GPU |  0.0026    |  0.0030   | 15     |  40min     |    
| 4 GPUs|  0.0031    |  0.0032   | 30     |  10min     |
| 8 GPUs|  0.0032    |  0.0033   | 60     |  5min      |


## Reference
- [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch)
- [NeRF Tensorflow](https://github.com/bmild/nerf)

