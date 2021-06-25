# Diversity-Augmented Intrinsic Motivation for Deep Reinforcement Learning
Here is the official code for our paper - "Diversity-Augmented Intrinsic Motivation for Deep Reinforcement Learning" (Under Review).

## Requirements
- ubuntu-16.04
- cuda-10.0
- pytorch==1.2.0
- gym==0.12.5
- gym[atari]
- mujoco-py==1.50.1.56
- mpi4py
- cloudpickle

## Install 
1. Please install the required packages in the above list.  
2. Install `rl_utils`:
```bash
pip install -e .
```
## Run Experiments
Please enter `rl_algorithms/ppo` folder to run `PPO + DAIM`, and enter `rl_algorithms/a2c` folder to run `A2C + DAIM`.