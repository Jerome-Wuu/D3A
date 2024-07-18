# Focus on Primary: Differential Diverse Data Augmentation for Generalization in Visual Reinforcement Learning

**[07/15/2024] Under review as a conference paper at AAAI 2025**

Official implementations of 
**Focus on Primary: Differential Diverse Data Augmentation for Generalization in Visual Reinforcement Learning** (DDA&D3A)

Benchmark for generalization in continuous control from pixels, based on [DMControl Generalization Benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) and [RL-Vigen](https://github.com/gemcollector/RL-ViGen).

Note: To run the program you need to add dependency files from [data](https://github.com/gemcollector/TLDA/tree/master/src/env/data) to local . /src/env/data
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
