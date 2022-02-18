from game import Game
import pygame
from agent import Agent
from hand import Hand
import cv2
import itertools
import argparse
from threading import Thread
import sys

N_FRAMES = 4

def check_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True

def bot_action(game):
    if game.ball.y < game.player2.y + game.player2.height/2:
        game.player2.mode = -1
    elif game.ball.y > game.player2.y - game.player2.height/2:
        game.player2.mode = 1

def human_action(cap, hand, game):
    while True:
        success, image = cap.read()
        run, action = hand.get_action(success, image)
        game.player2.mode = action

def main(args):
    game = Game(args['see'], args['train'])
    agent = Agent('Pong.pth')
    if args['load']:
        agent.load()
    if args['human']:
        hand = Hand()
        cap = cv2.VideoCapture(0)
        thread = Thread(target=human_action, args=(cap, hand, game))
        thread.daemon = True
        thread.start()
    run = True
    state_old = game.getState()
    for frame in itertools.count():
        if not run:
            break
        if args['see']:
            run = check_events()
        if args['bot']:
            bot_action(game)
        if frame % N_FRAMES == 0:
            action1 = agent.get_action(state_old, frame if args['train'] else 'testing')
            game.player1.mode = action1 * 2 - 1 # 0, 1 -> -1, 1
        reward, done = game.run()
        state_new = game.getState()
        if args['train']:
            agent.train((state_old, action1, reward, done, state_new), frame, True)
        state_old = state_new

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
    sys.exit()
