from game import Game
import pygame
from agent import Agent
import itertools
import argparse

N_FRAMES = 2

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
    player1 = Agent()
    if args['load']:
        player1.load('model')
    run = True
    for frame in itertools.count():
        if not run:
            break
        if args['bot']:
            bot_action(game)
        elif args['human']:
            run = human_action(game)
        if frame % N_FRAMES == 0:
            state_old = game.getState()
            action = player1.get_action(state_old, frame/N_FRAMES if args['train'] else -1)
            if action == 0:
                game.player1.mode = -1
            else:
                game.player1.mode = 1
            reward, done = game.run()
            if args['train']:
                state_new = game.getState()
                player1.train(state_old, action, reward, done, state_new, frame/N_FRAMES)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--see', default='y', help='see the game')
    parser.add_argument('--human', default='n', help='play the game')
    parser.add_argument('--bot', default='y', help='train the model')
    parser.add_argument('--selfPlay', default='n', help='train the model')
    parser.add_argument('--load', default='n', help='train the model')
    parser.add_argument('--train', default='y', help='train the model')
    args = parser.parse_args()
    argsDict = vars(args)
    for key, value in argsDict.items():
        if value == 'y':
            argsDict[key] = True
        elif value == 'n':
            argsDict[key] = False
    main(argsDict)
