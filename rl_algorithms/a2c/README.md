# Synchronous Advantage Actor-Critic (A2C)
Examples to run A2C with DAIM.
## Instructions
1. Train the DAIM agent:
```bash
python train.py --env-name="BreakoutNoFrameskip-v4" --log-dir="logs" --cuda --r-in-coef=0.01 --lr-decay  --seed=123
```