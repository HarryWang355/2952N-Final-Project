# 2952N-Final-Project

We base our implementation on NSFF [codebase](https://github.com/zhengqili/Neural-Scene-Flow-Fields) and a third-party [implementation](https://github.com/ashawkey/torch-ngp) of INGP encoding.

## Preparing the environment

Due to the implementation details the code requires CUDA 11.X and a GPU with compute capability 7+.

First, it is required to compile and install INGP CUDA kernels. Step into ```nsff_exp/gridencoder``` and run

```
python setup.py build_ext --inplace
pip install .
```

Next, install the required packages

```
pip install -r requirements.txt
```

Note: you might need to install a ```cupy``` package for your version of CUDA.

## Data preprocessing

Please, refer to NSFF [instructions](https://github.com/zhengqili/Neural-Scene-Flow-Fields#video-preprocessing) on data downloading and preprocessing.

## Training

To start the training a config file is required. We provide sample configs for some of our ablations, they can be found in ```nsff_exp/configs/dl_final```

To train a network with 4-layer 64-wide MLP execute from ```nsff_exp```

```
python run_nerf.py --config configs/dl_final/config_balloon_64_4.txt
```

Decrease ```N_rand``` parameter to regulate a batch size if it can't fit into a GPU.

## Evaluation

To perform evaluation and compute metrics run

```
python evaluation.py --config configs/dl_final/config_balloon_64_4.txt
```

## Novel view rendering

To generate a set of novel views and perform a time-space interpolation run

```
python run_nerf.py --config configs/dl_final/config_balloon_64_4.txt --render_slowmo_bt  --target_idx 10
```

Please, refer to NSFF [readme](https://github.com/zhengqili/Neural-Scene-Flow-Fields#rendering-from-an-example-pretrained-model) for additional inference options 
