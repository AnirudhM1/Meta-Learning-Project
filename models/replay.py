from collections import namedtuple, deque
import random
import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FQNReplayBuffer:

    def __init__(self, max_size):
        self.memory = deque([], max_size)
        self.max_size = max_size
        self.device = get_device()

    def push(self, state, action, next_state, reward, done):
        transition = Transition(state, action, next_state, reward, done)
        self.memory.append(transition)
    
    def sample(self, batch_size):
        sample_size = min(batch_size, len(self))
        sample = random.sample(self.memory, sample_size)
        return self.convert_to_batch(sample)

    def sample_to_torch(self, batch_size):
        sample = self.sample(batch_size)

        state = sample.state
        action = sample.action
        next_state = sample.next_state
        reward = sample.reward
        done = sample.done

        state = torch.from_numpy(state).float().to(self.device)
        action = torch.tensor(action, device=self.device).long()
        next_state = torch.from_numpy(next_state).float().to(self.device)
        reward = torch.tensor(reward, device=self.device).float()
        done = torch.tensor(done, device=self.device).float()

        return Transition(state, action, next_state, reward, done)
    
    def __len__(self):
        return len(self.memory)
    
    def convert_to_batch(self, transitions):
        batch = Transition(*zip(*transitions))

        # if isinstance(batch.state, tuple):
            # print(len(batch.state))
        np_state = np.stack(batch.state)
        np_action = np.array(batch.action)
        np_next_state = np.stack(batch.next_state)
        np_reward = np.array(batch.reward)
        np_done = np.array(batch.done)

        np_batch = Transition(np_state, np_action, np_next_state, np_reward, np_done)

        return np_batch
    
    def is_full(self):
        return len(self.memory) == self.max_size
    

Walk = namedtuple('Walk', ('states', 'actions', 'label'))


class PSIReplayBuffer:
    def __init__(self, max_walks):
        self.memory = deque([], max_walks)
        self.max_walks = max_walks
        self.device = get_device()
    
    def push(self, states, actions, label =  1):
        states = np.stack(states)
        actions = np.stack(actions)
        walk = Walk(states, actions,label)
        self.memory.append(walk)
        random.shuffle(self.memory)
    def __len__(self):
        return len(self.memory)
    
    def sample(self):
        sample = random.sample(self.memory, 1)[0]
        return sample
    
    def sample_to_torch(self):
        sample = self.sample()
        states = sample.states
        actions = sample.actions
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.tensor(actions, device=self.device).long()

        walk = Walk(states, actions,sample.label)
        return walk