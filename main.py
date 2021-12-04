from game import Game
from agent import Agent
import pygame
import numpy as np

def main(play, train):
    game = Game()
    agent = Agent()
    record = 0

    run = True
    while run:
        # player2
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

        # player1
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)

        if np.array_equal(action, [1, 0]):
            game.player1.mode = -1

        elif np.array_equal(action, [0, 1]):
            game.player1.mode = 1

        reward, done, score = game.run(play)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)
        if done:
            agent.train_long_memory()
            if score > record:
                print("New record: ", score)
                record = score
                agent.model.save()

    pygame.quit()

if __name__ == '__main__':
    main(False, True)