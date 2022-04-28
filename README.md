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

## Notes and Analysis

1. As seen after the training, the model does not show any increase in the cumulative reward.
2. Upon further probing we notice that the successor features are very similar irrespective of state.
3. This can be attributed to the objective function which only pushes the probabilities of the states and actions on the trajectory but does not make the successors far apart for far apart states.
4. Thus, the model converges to giving very close values of the successors, irrespective of the state giving innaccurate successors.
5. We proposed the use of contrastive learning by giving negative samples and forcing the successors to be far apart for the same.
6. We were unable to verify this approach as despite initial contrastive learning seemed to increase the distance for far apart successors slightly but we did not have enough memory to store more negative samples for the same.


