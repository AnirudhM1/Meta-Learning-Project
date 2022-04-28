import meta_envs
from models import Agent, FQN, PSI
from meta_envs.wrappers import OneHotWrapper
import gym
import wandb
import torch





# HYPERPARAMETERS

TOTAL_EPOCHS = 3000
SUCCESSOR_TRAINING_EPOCHS = 100
Q_TRAINING_EPOCHS = 50

EMBEDDING_DIM = 128
HIDDEN_SIZE = [32, 16, 256]
MAX_TRAJECTORY_LENGTH = 100
INITIAL_GATHERING_LENGTH = 300
OPTIMIZATION_FREQUENCY = 1
DISPLAY_RESULT_FREQUENCY = 50
MAX_EPSILON = 1
MIN_EPSILON = 0.01
EPSILON_DECAY_FACTOR = 0.975
FQL_TRAINING_START = 100
GAMMA = 0.8
GRID_SIZE = 13



wandb.init(project="Meta-Learning-Project")

wandb.config = {
  "epochs": TOTAL_EPOCHS,
  "batch_size": EMBEDDING_DIM,
  "model_type": "cnn",
  "gamma": GAMMA,
  "enviroment": "Four-room-domain-v0",
  "grid_size": GRID_SIZE
}



env = gym.make('meta_envs/Four-room-domain-v0', n=GRID_SIZE)
env = OneHotWrapper(env)

input_shape = env.observation_space.shape
input_size = input_shape[0]

current_epsilon = MAX_EPSILON - MIN_EPSILON


psi = PSI([input_size, *HIDDEN_SIZE], EMBEDDING_DIM, env.action_space.n)
fqn = FQN(EMBEDDING_DIM)

wandb.watch(psi)
wandb.watch(fqn)

agent = Agent(env, psi, fqn)




# Initial gathering phase

print('Initial Gathering phase')

for _ in range(INITIAL_GATHERING_LENGTH):
    state = env.reset()
    done = False
    step = 0

    states = []
    actions = []

    while not done and step < MAX_TRAJECTORY_LENGTH:
        step += 1
        action = agent.act(state, epsilon=1)
        next_state, reward, done, _ = env.step(action)
        agent.fqn_replay.push(state, action, reward, next_state, done)
        states.append(state)
        actions.append(action)
        state = next_state
    
    agent.psi_replay.push(states, actions)


print('Completed Gathering of Initial Data')





for epoch in range(TOTAL_EPOCHS):

    # Initial gathering phase

    # print('Initial Gathering phase')

    # for _ in range(INITIAL_GATHERING_LENGTH):
    #     state = env.reset()
    #     done = False
    #     step = 0

    #     states = []
    #     actions = []

    #     while not done and step < MAX_TRAJECTORY_LENGTH:
    #         step += 1
    #         action = agent.act(state, epsilon=current_epsilon + MIN_EPSILON)
    #         next_state, reward, done, _ = env.step(action)
    #         agent.fqn_replay.push(state, action, reward, next_state, done)
    #         states.append(state)
    #         actions.append(action)
    #         state = next_state
        
    #     agent.psi_replay.push(states, actions)


    # print('Completed Gathering of Initial Data')


    # Training psi
    psi_loss = agent.train_psi(SUCCESSOR_TRAINING_EPOCHS, INITIAL_GATHERING_LENGTH)

    print('Completed Successor Training')


    # Training fqn

    agent.reset_fqn(EMBEDDING_DIM)

    cummalative_reward = 0
    cummalative_loss = 0

    for episode in range(Q_TRAINING_EPOCHS):
        state = env.reset()
        done = False
        step = 0

        states = []
        actions = []

        total_reward = 0
        episode_loss = []

        while not done and step < MAX_TRAJECTORY_LENGTH:
            step += 1
            action = agent.act(state, epsilon=current_epsilon + MIN_EPSILON)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.fqn_replay.push(state, action, reward, next_state, done)
            states.append(state)
            actions.append(action)
            state = next_state
            iteration_loss = agent.train_fqn()
            episode_loss.append(iteration_loss.detach())
        

        agent.psi_replay.push(states, actions)
        cummalative_reward += total_reward
        loss = torch.stack(episode_loss).mean().item()
        cummalative_loss += loss

        if episode % DISPLAY_RESULT_FREQUENCY == 0:
            print(f'Episode: {episode+1}    Loss; :{loss}')
            print(f'Epsilon: {current_epsilon + MIN_EPSILON}')
            print()
            print('-----------------------------------------------')
            print()

        current_epsilon *= EPSILON_DECAY_FACTOR

    print('-----------------------------------------------')
    print(f'Epoch: {epoch+1}    Cummalative Reward: {cummalative_reward/Q_TRAINING_EPOCHS}')
    print('-----------------------------------------------')
    print()

    # Logging and saving model
    wandb.log({
        'Cummalative Reward': cummalative_reward/Q_TRAINING_EPOCHS,
        'Cummalative Q-Learning Loss': cummalative_loss/Q_TRAINING_EPOCHS,
        'Cummalative Successor Loss': psi_loss
    })

    torch.save(psi, 'checkpoint/psi.pt')
    torch.save(fqn, 'checkpoint/fqn.pt')