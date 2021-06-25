# Proximal Policy Optimization (PPO)
Examples to run PPO with DAIM.
## Instructions
1. Train the DAIM agent:
```bash
python -u train.py --env-name='HalfCheetah-v2' --cuda (if cuda is available) --reward-delay-freq=1 --r-ext-coef=1 --log-dir='logs' --seed=123
```