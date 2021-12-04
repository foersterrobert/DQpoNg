from game import Game
import pygame

def main(play):
    game = Game()
    while True:
        if play:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        game.player2.mode = -1
                    
                    if event.key == pygame.K_DOWN:
                        game.player2.mode = 1

        else:
            if game.ball.y > game.player2.y:
                game.player2.mode = -1

            elif game.ball.y < game.player2.y:
                game.player2.mode = 1
                
        game.run(play)

if __name__ == '__main__':
    main(True)