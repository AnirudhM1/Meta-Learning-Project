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

## Training details

In each epoch, the training is split into 2 parts:

1. First we train the Successor Feature (SF) in an unsupervised manner. We do so by optimizing the objective function given in the paper over a fixed set of walks.
2. We then freeze the training of the SF and train a vector of weights to obtain the Action-Value (Q-value) function for a particular task distribution. This part of the training is done using fitted Q-learning. <br />This updated Q value is then used to generate new walks for the the training of the SF in the next epoch.

## Model details

 To train the SF, we have used 2 differnt models for the Grid World environment and Market RL environment. This is done because the SF is a function of the environment state and action in the case of Grid World. For Market RL, it is a function of only the state. This is because any action taken by the agent cannot influence the next state of the market.
 
 1. **Grid World:**
      The model used is a CNN. It takes in the state as input and outputs the successor Feature representation for each of the 4 possible actions.
      > Since SF is dependant of the action taken, we return 4 differnt vectors representing the SF for each action taken in the particular state.
      <br />
 2. **Market RL**:
       The model used here is an RNN. To ensure that the process is Markov, the state after each action is the concatenation of every state in it's history. Since the shape of the state vector is variable and time dependant, we pass it through LSTM layers and then return the output as a single vector containg the SF for a particular state.
       > Since SF is independent of the action taken in this case, we just return a single vector for each state.


## Notes and Analysis

1. As seen after the training, the model does not show any increase in the cumulative reward.
2. Upon further probing we notice that the successor features are very similar irrespective of state.
3. This can be attributed to the objective function which only pushes the probabilities of the states and actions on the trajectory but does not make the successors far apart for far apart states.
4. Thus, the model converges to giving very close values of the successors, irrespective of the state giving innaccurate successors.
5. We proposed the use of contrastive learning by giving negative samples and forcing the successors to be far apart for the same.
6. We were unable to verify this approach as despite initial contrastive learning seemed to increase the distance for far apart successors slightly but we did not have enough memory to store more negative samples for the same.


