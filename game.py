import pygame

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((640, 480))
        self.clock = pygame.time.Clock()
        self.player1 = Paddle(self.screen, 10)
        self.player2 = Paddle(self.screen, 600)
        self.ball = Ball(self.screen)

    def update(self):
        self.player1.update()
        self.player2.update()
        self.ball.update()

    def render(self):
        self.screen.fill((0, 0, 0))
        self.player1.render()
        self.player2.render()
        self.ball.render()
        pygame.display.flip()

    def run(self, play):
        self.update()
        self.render()
        if play:
            self.clock.tick(60)


class Ball:
    def __init__(self, screen):
        self.screen = screen
        self.x = 0
        self.y = 0
        self.x_speed = 0
        self.y_speed = 0

    def update(self):
        self.x += self.x_speed
        self.y += self.y_speed

    def render(self):
        pygame.draw.circle(self.screen, (255, 255, 255), (self.x, self.y), 10)

class Paddle:
    def __init__(self, screen, x):
        self.screen = screen
        self.y = 0
        self.x = x
        self.speed = 5
        self.score = 0
        self.mode = 0

    def update(self):
        self.y += self.mode * self.speed

    def render(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (self.x, self.y, 10, 100))