# Generative Augmented Flow Networks

This repository is the implementation of [Generative Augmented Flow Networks](https://openreview.net/forum?id=urF_CBK5XC0) in ICLR 2023 (Spotlight). This codebase is based on the open-source [gflownet](https://github.com/GFNOrg/gflownet) implementation, and please refer to that repo for more documentation.

## Citing

If you used this code in your research or found it helpful, please consider citing our paper:
```
@inproceedings{
	pan2023generative,
	title={Generative Augmented Flow Networks},
	author={Ling Pan and Dinghuai Zhang and Aaron Courville and Longbo Huang and Yoshua Bengio},
	booktitle={International Conference on Learning Representations},
	year={2023},
	url={https://openreview.net/forum?id=urF_CBK5XC0}
}
```

## Requirements

### Grid
- python: 3.6
- torch: 1.3.0
- scipy: 1.5.4
- numpy: 1.19.5
- tdqm

### Molecule discovery
Please check the [gflownet](https://github.com/GFNOrg/gflownet) repo for more details about the environment

## Usage

Please follow the instructions below to replicate the results in the paper. 
- Grid
```
python toy_grid_dag.py --augmented 1 --seed <SEED> --horizon <HORIZON>
```
- Molecule discovery
```
python gflownet.py --w_ri 1
```
