# Focus on Primary: Differential Diverse Data Augmentation for Generalization in Visual Reinforcement Learning

**[01/07/2025] Under review as a conference paper at IJCAI 2025**

Official implementations of 
**Focus on Primary: Differential Diverse Data Augmentation for Generalization in Visual Reinforcement Learning** (DDA&D3A)


## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:
```
conda env create -f setup/conda.yaml
conda activate d3a
sh setup/install_envs.sh
```

Benchmarks for generalization in continuous control from pixels are based on [DMControl Generalization Benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) and [RL-Vigen](https://github.com/gemcollector/RL-ViGen).
Note: To run the program you need to add dependency files from [data](https://github.com/nicklashansen/svea-vit/tree/main/src/env/data) to local . /src/env/data

## Training 
The `scripts` directory contains training and evaluation bash scripts for all the included algorithms. Alternatively, you can call the python scripts directly, e.g. for training call
```
python3 src/train.py \
  --algorithm dda \
  --seed 0
```
```
python3 src/train.py \
  --algorithm d3a \
  --seed 0
```
