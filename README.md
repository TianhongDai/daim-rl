# Diversity-Augmented Intrinsic Motivation for Deep Reinforcement Learning
Here is the official code for our paper - ["Diversity-Augmented Intrinsic Motivation for Deep Reinforcement Learning"](https://www.sciencedirect.com/science/article/pii/S0925231221015265?via%3Dihub) [Neurocomputing 2021].
## 
![illustrations](figures/illustration.pdf)
## Requirements
- ubuntu-16.04
- cuda-10.0
- pytorch==1.2.0
- gym==0.12.5
- gym[atari]
- mujoco-py==1.50.1.56
- mpi4py
- cloudpickle

## Installation
1. Please install the required packages in the above list.  
2. Install `rl_utils`:
```bash
pip install -e .
```
## Run Experiments
Please enter `rl_algorithms/ppo` folder to run `PPO + DAIM`, and enter `rl_algorithms/a2c` folder to run `A2C + DAIM`.

## BibTex
To cite this code for publications - please use:
```
@article{dai2021diversity,
    title={Diversity-Augmented Intrinsic Motivation for Deep Reinforcement Learning},
    author={Dai, Tianhong and Du, Yali and Fang, Meng and Bharath, Anil Anthony},
    journal={Neurocomputing},
    volume = {468},
    pages = {396-406},
    year={2021},
    publisher={Elsevier}
}
```