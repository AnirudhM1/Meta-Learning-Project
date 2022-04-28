# Disentangled Predictive Representation for Meta Reinforcement Learning - Paper Implementation

This is a paper implementation of the above paper done for the Meta Learning course.

This repo contains:
1. A set of python files which trains the agent on the Four Grid domain as specified in the [paper](https://openreview.net/pdf?id=VbLGbcdz16-).
2. An .ipynb file which trains an agent on market RL data.


## Running the python code

**Install the environment**

```bash
pip install -e meta-envs
```

**Install and Enable wandb for logging purposes**
```bash
pip install wandb
wandb login
```

Two methods have been implemented to train the agent:
1. Using standard optimization function as specified in the paper. (train.py)
2. Using a contrastive loss term. (train_contrast.py)

**Standard Optimization function**
```bash
python train.py
```

**Contrastive Learning**
```bash
python train_contrast.py
```
