from model import Linear_QNet
from collections import deque
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import random
import os

GAMMA = 0.95
BATCH_SIZE = 64
LEARNING_RATE = 0.00025
MAX_MEMORY = 200_000
MIN_REPLAY_SIZE = 100_000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 50000
TARGET_UPDATE_FREQ = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.online_net = Linear_QNet(5, 2).to(DEVICE)
        self.target_net = Linear_QNet(5, 2).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.memory = deque(maxlen=MAX_MEMORY)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.record = -50
        self.score = 0

    def get_action(self, state, step):
        if step + 1:
            epsilon = np.interp(step - MIN_REPLAY_SIZE, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        else:
            epsilon = 0
        if random.random() <= epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
                q_values = self.online_net(state_t.unsqueeze(0))
                max_q_index = torch.argmax(q_values, dim=1)[0]
                action = max_q_index.detach().item()
        return action

    def train(self, state_old, action, reward, done, state_new, step):
        self.memory.append((state_old, action, reward, done, state_new))
        if len(self.memory) > MIN_REPLAY_SIZE:
            transitions = random.sample(self.memory, BATCH_SIZE)

            state_olds = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rewards = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            state_news = np.asarray([t[4] for t in transitions])

            obses_t = torch.as_tensor(state_olds, dtype=torch.float32, device=DEVICE)
            actions_t = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(-1)
            rews_t = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
            new_obses_t = torch.as_tensor(state_news, dtype=torch.float32, device=DEVICE)

            # Compute Targets
            # double q learning later
            target_q_values = self.target_net(new_obses_t)
            max_target_q_values = target_q_values.max(
                dim=1, keepdim=True)[0]
            targets = rews_t + GAMMA * \
                (1 - dones_t) * max_target_q_values

            # Compute Loss
            q_values = self.online_net(obses_t)
            action_q_values = torch.gather(
                input=q_values, dim=1, index=actions_t)
            loss = nn.functional.smooth_l1_loss(
                action_q_values, targets)

            # Gradient Descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update Target Net
            if max(-1, step - MIN_REPLAY_SIZE) % TARGET_UPDATE_FREQ == 0:
                print(step)
                self.target_net.load_state_dict(self.online_net.state_dict())

            if done:
                if self.score > self.record:
                    self.record = self.score
                    print('Record:', self.record, 'Step:', step)
                    self.save()
                self.score = 0

    def load(self, name):
        self.online_net.load_state_dict(torch.load(f'model/{name}.pth', map_location=DEVICE))

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.online_net.state_dict(), file_name)