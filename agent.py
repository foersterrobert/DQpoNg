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
EPSILON_DECAY = 80000
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
        action = None
        if isinstance(step, int):
            epsilon = np.interp(step, [MIN_REPLAY_SIZE, EPSILON_DECAY + MIN_REPLAY_SIZE], [EPSILON_START, EPSILON_END])
            if random.random() <= epsilon:
                action = random.randint(0, 1)
        if not action:
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
                q_values = self.online_net(state_t.unsqueeze(0))
                max_q_index = torch.argmax(q_values, dim=1)[0]
                action = max_q_index.detach().item()
        return action

    def train(self, experience, step, DDQN):
        self.memory.append(experience)
        if len(self.memory) > MIN_REPLAY_SIZE:
            self.score += experience[2] # reward

            sample_experiences = random.sample(self.memory, BATCH_SIZE)

            state_olds = np.asarray([t[0] for t in sample_experiences])
            actions = np.asarray([t[1] for t in sample_experiences])
            rewards = np.asarray([t[2] for t in sample_experiences])
            dones = np.asarray([t[3] for t in sample_experiences])
            state_news = np.asarray([t[4] for t in sample_experiences])

            state_olds_t = torch.as_tensor(state_olds, dtype=torch.float32, device=DEVICE)
            actions_t = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(-1)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
            state_news_t = torch.as_tensor(state_news, dtype=torch.float32, device=DEVICE)

            # get Outputs
            q_values = self.online_net(state_olds_t)
            q_values_action = q_values.gather(dim=1, index=actions_t)

            # get Targets
            if DDQN:
                q_values_online = self.online_net(state_news_t)
                q_values_online_max = q_values_online.argmax(dim=1, keepdim=True)
                q_values_target = self.target_net(state_news_t)
                q_values_target_selected = q_values_target.gather(dim=1, index=q_values_online_max)
                targets = rewards_t + GAMMA * (1 - dones_t) * q_values_target_selected
            else:
                q_values_target = self.target_net(state_news_t)
                q_values_target_max = q_values_target.max(dim=1, keepdim=True)[0]
                targets = rewards_t + GAMMA * (1 - dones_t) * q_values_target_max

            # Compute Loss
            loss = nn.functional.smooth_l1_loss(q_values_action, targets)

            # Gradient Descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update Target Net
            if max(-1, step - MIN_REPLAY_SIZE) % TARGET_UPDATE_FREQ == 0:
                print(step)
                self.target_net.load_state_dict(self.online_net.state_dict())

            # check for record and save model if done
            if experience[3]: # done
                if self.score > self.record:
                    self.record = self.score
                    print('Record:', self.record, 'Step:', step)
                    self.save()
                self.score = 0

    def load(self, name):
        # checkpoint = torch.load(f'model/{name}.pth', map_location=DEVICE)
        # self.online_net.load_state_dict(checkpoint['online_net'])
        # self.target_net.load_state_dict(checkpoint['target_net'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.online_net.load_state_dict(torch.load(f'model/{name}.pth', map_location=DEVICE))

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        # save_state = {
        #     'online_net': self.online_net.state_dict(),
        #     'target_net': self.target_net.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        # }
        # torch.save(save_state, file_name)
        torch.save(self.online_net.state_dict(), file_name)