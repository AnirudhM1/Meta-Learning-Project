import math
import random
import torch
import torch.nn.functional as F
from models.model import FQN

from models.replay import FQNReplayBuffer, PSIReplayBuffer

class Agent:
    def __init__(self, env, psi, fqn, device=None, lr=2e-3, gamma=0.8, batch_size=64, max_capacity=1000, max_walks=300):
        self.env = env
        self.device = device if device is not None else self.get_device()
        self.psi = psi.to(self.device)
        self.fqn = fqn.to(self.device)
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.psi_optimizer = torch.optim.Adam(self.psi.parameters(), lr=self.lr, weight_decay=0.2)
        self.fqn_optimizer = torch.optim.Adam(self.fqn.parameters(), lr=self.lr)

        self.psi_replay = PSIReplayBuffer(max_walks)
        self.fqn_replay = FQNReplayBuffer(max_capacity)

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_successor(self, state):
        # state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        successor_feature = self.psi(state)
        return successor_feature
    
    def get_q_value(self, successor_feature):
        q_values = self.fqn(successor_feature)
        return q_values.squeeze(-1)
    

    def act(self, state, epsilon=0):

        if random.random() < epsilon:
            action = self.env.action_space.sample()
            return action
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device).float()
            successor_feature = self.get_successor(state.unsqueeze(0))
            q_values = self.get_q_value(successor_feature)
            action = q_values.argmax().item()
            return action
    
    def train_psi(self, num_epochs, num_walks):
        loss = []
        for epoch in range(num_epochs):
            objective = 0
            for _ in range(num_walks):
                walk = self.psi_replay.sample_to_torch()
                objective = objective + self.calculate_objective(walk)*walk.label
            
            objective = (objective/num_walks) * -1 # This multiplies by -1 because we want to maximize the objective
            loss.append(objective*-1)

            self.psi_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self.psi_optimizer.step()
            print(f"epoch : {epoch} objective : {objective}")
        loss = torch.stack(loss).mean().item()
        return loss
    
    def train_fqn(self):
        state, action, reward, next_state, done = self.fqn_replay.sample_to_torch(self.batch_size)

        with torch.no_grad():
            successor_feature = self.get_successor(state)
            next_successor_feature = self.get_successor(next_state)

        with torch.no_grad():
            next_q_values = self.get_q_value(next_successor_feature)
        optimal_next_q_value, _ = next_q_values.max(1)
        target_q_values = reward + (self.gamma * optimal_next_q_value * (1 - done))

        input_q_values = self.get_q_value(successor_feature)
        q_values = input_q_values.gather(1, action.unsqueeze(1)).squeeze()

        loss = F.mse_loss(q_values, target_q_values)

        self.fqn_optimizer.zero_grad()
        loss.backward()
        self.fqn_optimizer.step()

        return loss
    
    def calculate_objective(self, walk):
        
        successor = self.psi(walk.states)
        actions = walk.actions
        successor = torch.stack([successor[i,actions[i],:] for i in range(len(actions))])

        discount_matrix = torch.arange(1, successor.shape[0]+1, device=self.device) * math.log(self.gamma)
        discount_matrix = discount_matrix.unsqueeze(0)

        successor = F.normalize(successor,dim=1)
        log_probs = successor@successor.T
        log_probs = torch.nn.LogSigmoid()(log_probs)       


        log_probs = log_probs + discount_matrix
        log_probs = log_probs - discount_matrix.T
        log_probs = torch.triu(log_probs, 1)
        return log_probs.sum()
    

    def reset_fqn(self, embedding_dim):
        self.fqn = FQN(embedding_dim).to(self.device)

