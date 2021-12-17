from game import Game
import pygame
from agent import Agent
from hand import Hand
import cv2
import itertools
import argparse

N_FRAMES = 2
N_FRAMES_HAND = 4

def check_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True

def bot_action(game, player):
    if player == 1:
        if game.ball.y < game.player1.y + game.player1.height/2:
            game.player1.mode = -1
        elif game.ball.y > game.player1.y - game.player1.height/2:
            game.player1.mode = 1
    else:
        if game.ball.y < game.player2.y + game.player2.height/2:
            game.player2.mode = -1
        elif game.ball.y > game.player2.y - game.player2.height/2:
            game.player2.mode = 1

def main(args):
    game = Game(args['see'], args['train'])
    player1 = Agent()
    # player2 = Agent()
    if args['load']:
        player1.load('player1')
        # player2.load('player2')
    if args['human']:
        hand = Hand()
        cap = cv2.VideoCapture(0)
    run = True
    for frame in itertools.count():
        if not run:
            break
        if args['see']:
            run = check_events()
        if args['bot']:
            bot_action(game, 2)
        if args['human'] and frame % N_FRAMES_HAND == 0:
            success, image = cap.read()
            run, action = hand.get_action(success, image)
            game.player2.mode = action
        if frame % N_FRAMES == 0:
            state_old = game.getState()
            action1 = player1.get_action(state_old, frame/N_FRAMES if args['train'] else 'testing')
            if action1 == 0:
                game.player1.mode = -1
            else:
                game.player1.mode = 1
            # action2 = player2.get_action(state_old, frame/N_FRAMES if args['train'] else False)
            # if action2 == 0:
            #     game.player2.mode = -1
            # else:
            #     game.player2.mode = 1
            reward, done = game.run()
            if args['train']:
                state_new = game.getState()
                player1.train((state_old, action1, reward[0], done[0], state_new), frame/N_FRAMES, True)
                # player2.train((state_old, action2, reward[1], done[1], state_new), frame/N_FRAMES, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--see', default='y', help='see the game')
    parser.add_argument('--human', default='y', help='play the game')
    parser.add_argument('--bot', default='n', help='train the model')
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
