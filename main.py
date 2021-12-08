from game import Game
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Linear_QNet
import random
from collections import deque
import itertools
import argparse

N_FRAMES = 2
GAMMA = 0.95
BATCH_SIZE = 64
LEARNING_RATE = 0.00025
MAX_MEMORY = 200_000
MIN_REPLAY_SIZE = 100_000 * N_FRAMES
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 50000 * N_FRAMES
TARGET_UPDATE_FREQ = 10000 * N_FRAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def human_action(game):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                game.player2.mode = -1
            if event.key == pygame.K_DOWN:
                game.player2.mode = 1
        if event.type == pygame.QUIT:
            return False
    return True

def bot_action(game):
        if game.ball.y < game.player2.y + game.player2.height/2:
            game.player2.mode = -1
        elif game.ball.y > game.player2.y - game.player2.height/2:
            game.player2.mode = 1

def main(args):
    game = Game(args['see'], args['train'])
    online_net = Linear_QNet(5, 2).to(DEVICE)
    if args['load']:
        online_net.load_state_dict(torch.load('model/model1.pth', map_location=DEVICE))
    run = True
    if args['train']:
        target_net = Linear_QNet(5, 2).to(DEVICE)
        target_net.load_state_dict(online_net.state_dict())
        memory = deque(maxlen=MAX_MEMORY)
        optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
        record = -50
        score = 0

        state_old = game.getState(False)
        for step in itertools.count():
            if not run:
                break
            if args['human']:
                run = human_action(game)
            elif args['bot']:
                bot_action(game)
            
            if step % N_FRAMES == 0: 
                if args['selfPlay']:
                    swapped_state = game.getState(True)
                    action = online_net.act(swapped_state)
                    if action == 0:
                            game.player2.mode = -1
                    else:
                        game.player2.mode = 1
                if step < MIN_REPLAY_SIZE: # Load with random actions at first
                    action = random.randint(0, 1)
                    if action == 0:
                        game.player1.mode = -1
                    else:
                        game.player1.mode = 1
                    reward, done = game.run()
                    state_new = game.getState(False)
                    memory.append((state_old, action, reward, done, state_new))
                    state_old = state_new
                else:
                    epsilon = np.interp(
                        step-MIN_REPLAY_SIZE, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
                    if random.random() <= epsilon:
                        action = random.randint(0, 1)
                    else:
                        action = online_net.act(state_old)
                    if action == 0:
                        game.player1.mode = -1
                    else:
                        game.player1.mode = 1

                    reward, done = game.run()
                    score += reward
                    state_new = game.getState(False)
                    memory.append((state_old, action, reward, done, state_new))
                    state_old = state_new

                    transitions = random.sample(memory, BATCH_SIZE)

                    obses = np.asarray([t[0] for t in transitions])
                    actions = np.asarray([t[1] for t in transitions])
                    rews = np.asarray([t[2] for t in transitions])
                    dones = np.asarray([t[3] for t in transitions])
                    new_obses = np.asarray([t[4] for t in transitions])

                    obses_t = torch.as_tensor(obses, dtype=torch.float32, device=DEVICE)
                    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(-1)
                    rews_t = torch.as_tensor(rews, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
                    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
                    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=DEVICE)

                    # Compute Targets
                    # targets = r + gamma * target q vals * (1 - dones)
                    target_q_values = target_net(new_obses_t)
                    max_target_q_values = target_q_values.max(
                        dim=1, keepdim=True)[0]
                    targets = rews_t + GAMMA * \
                        (1 - dones_t) * max_target_q_values

                    # Compute Loss
                    q_values = online_net(obses_t)
                    action_q_values = torch.gather(
                        input=q_values, dim=1, index=actions_t)
                    loss = nn.functional.smooth_l1_loss(
                        action_q_values, targets)

                    # Gradient Descent
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update Target Net
                    if max(1, step-MIN_REPLAY_SIZE) % TARGET_UPDATE_FREQ == 0:
                        print(step, epsilon)
                        target_net.load_state_dict(online_net.state_dict())

                    if done:
                        if score > record:
                            record = score
                            print('Record:', record, 'Step:', step)
                            online_net.save()
                        score = 0
    
            else:
                reward, done = game.run()
    else:
        for step in itertools.count():
            if not run:
                break

            if args['human']:
                run = human_action(game)
            elif args['bot']:
                bot_action(game)

            if step % N_FRAMES == 0:
                if args['selfPlay']:
                    swapped_state = game.getState(True)
                    action = online_net.act(swapped_state)
                    if action == 0:
                            game.player2.mode = -1
                    else:
                        game.player2.mode = 1
                state = game.getState(False)
                action = online_net.act(state)
                if action == 0:
                    game.player1.mode = -1
                else:
                    game.player1.mode = 1
                reward, done = game.run()

    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--see', default='y', help='see the game')
    parser.add_argument('--human', default='y', help='play the game')
    parser.add_argument('--bot', default='n', help='train the model')
    parser.add_argument('--selfPlay', default='n', help='train the model')
    parser.add_argument('--load', default='y', help='train the model')
    parser.add_argument('--train', default='n', help='train the model')
    args = parser.parse_args()
    argsDict = vars(args)
    for key, value in argsDict.items():
        if value == 'y':
            argsDict[key] = True
        elif value == 'n':
            argsDict[key] = False
    main(argsDict)
