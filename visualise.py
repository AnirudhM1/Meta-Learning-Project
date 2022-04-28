import meta_envs
from models import Agent, FQN, PSI
from meta_envs.wrappers import OneHotWrapper
import gym
import torch
import PIL

TOTAL_EPOCHS = 2000
SUCCESSOR_TRAINING_EPOCHS = 2
Q_TRAINING_EPOCHS = 100

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

env = gym.make('meta_envs/Four-room-domain-v0', n=GRID_SIZE)
env = OneHotWrapper(env)

input_shape = env.observation_space.shape
input_size = input_shape[0]

current_epsilon = MAX_EPSILON - MIN_EPSILON

psi = PSI([input_size, *HIDDEN_SIZE], EMBEDDING_DIM, env.action_space.n)
psi.load_state_dict(torch.load("psi.pt"))
fqn = FQN(EMBEDDING_DIM)

state = env.reset()
img = env.render(mode='rgb_array')
img = PIL.Image.fromarray(img)
img.save("og.png")
agent = Agent(env, psi, fqn)
state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)
log_probs_og = agent.get_successor(state)
file1 = open("log.txt","w")#append mode
file1.write(f"log_probs: \n{log_probs_og}\n")


state,_,_,_ = env.step(1)

img = env.render(mode='rgb_array')
img = PIL.Image.fromarray(img)
img.save("close.png")
state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)
log_probs_close = agent.get_successor(state)
file1.write(f"log_probs: \n{log_probs_close}\n")

state = env.reset()
img = env.render(mode='rgb_array')
img = PIL.Image.fromarray(img)
img.save("far.png")

state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)
log_probs_far = agent.get_successor(state)
file1.write(f"log_probs: \n{log_probs_far}\n")
file1.close()

print(log_probs_og.shape)
close_sim = torch.sum(log_probs_og[0][1] * log_probs_close[0][0])
far_sim = torch.sum(log_probs_og[0][1] * log_probs_far[0][0])

print(close_sim, far_sim)