import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(5, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = [
            round(game.player1.y / (480 - game.player1.height), 2),
            round(game.player2.y / (480 - game.player2.height), 2),
            round(game.ball.y / 480, 2),
            round(game.ball.x / 640, 2),
            round(game.ball.angle / 255, 2),
        ]

        return np.array(state, dtype=np.float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0,0]
        if random.randint(0, 200) < 60 - self.n_games:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move