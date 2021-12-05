from game import Game
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Linear_QNet
import torch
import random
from collections import deque
import itertools

GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
MAX_MEMORY = 200_000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 10000

def main(play, train):
    game = Game(play, train)
    online_net = Linear_QNet(5, 2)
    run = True
    if train:
        target_net = Linear_QNet(5, 2)
        target_net.load_state_dict(online_net.state_dict())
        memory = deque(maxlen=MAX_MEMORY)
        optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
        record = 0
        state_old = game.getState()
        for step in itertools.count(): 
            if not run:
                break
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and play:
                    if event.key == pygame.K_UP:
                        game.player2.mode = -1
                    
                    if event.key == pygame.K_DOWN:
                        game.player2.mode = 1

                if event.type == pygame.QUIT:
                    run = False
                    
            if not play:
                if game.ball.y < game.player2.y + game.player2.height/2:
                    game.player2.mode = -1

                elif game.ball.y > game.player2.y - game.player2.height/2:
                    game.player2.mode = 1

            # Load with random actions at first
            if step < MIN_REPLAY_SIZE:
                action = random.randint(0, 1)
                if action == 0:
                    game.player1.mode = -1
                else:
                    game.player1.mode = 1
                reward, done, score = game.run()
                state_new = game.getState()
                memory.append((state_old, action, reward, done, state_new))
                state_old = state_new

            else:
                epsilon = np.interp(step-MIN_REPLAY_SIZE, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
                if random.random() <= epsilon:
                    action = random.randint(0, 1)
                else:
                    action = online_net.act(state_old)
                if action == 0:
                    game.player1.mode = -1
                else:
                    game.player1.mode = 1

                reward, done, score = game.run()
                state_new = game.getState()
                memory.append((state_old, action, reward, done, state_new))
                state_old = state_new

                transitions = random.sample(memory, BATCH_SIZE)

                obses = np.asarray([t[0] for t in transitions])
                actions = np.asarray([t[1] for t in transitions])
                rews = np.asarray([t[2] for t in transitions])
                dones = np.asarray([t[3] for t in transitions])
                new_obses = np.asarray([t[4] for t in transitions])

                obses_t = torch.as_tensor(obses, dtype=torch.float32)
                actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
                rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
                dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
                new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

                # Compute Targets
                # targets = r + gamma * target q vals * (1 - dones)
                target_q_values = target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

                # Compute Loss
                q_values = online_net(obses_t)
                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

                loss = nn.functional.smooth_l1_loss(action_q_values, targets)

                # Gradient Descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update Target Net
                if step-MIN_REPLAY_SIZE % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(online_net.state_dict())

                if done:
                    if score > record:
                        record = score
                        print('Record:', record)
                        online_net.save()
    else:
        online_net.load_state_dict(torch.load('model/model.pth'))
        while run:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and play:
                    if event.key == pygame.K_UP:
                        game.player2.mode = -1
                    
                    if event.key == pygame.K_DOWN:
                        game.player2.mode = 1

                if event.type == pygame.QUIT:
                    run = False
                    
            if not play:
                if game.ball.y < game.player2.y + game.player2.height/2:
                    game.player2.mode = -1

                elif game.ball.y > game.player2.y - game.player2.height/2:
                    game.player2.mode = 1

            state = game.getState()
            action = online_net.act(state)

            if action == 0:
                game.player1.mode = -1

            else:
                game.player1.mode = 1

            reward, done, score = game.run()
            
    pygame.quit()

if __name__ == '__main__':
    main(False, True)