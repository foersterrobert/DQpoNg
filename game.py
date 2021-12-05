import pygame
import random
import numpy as np
from math import cos, sin, radians

class Game:
    def __init__(self, play, train):
        self.play = play
        self.train = train
        pygame.init()
        if self.play:
            pygame.display.set_caption('Pong')
            pygame.display.set_icon(pygame.image.load('assets/icon.png'))
            self.myFont = pygame.font.SysFont('arial', 30)
            self.screen = pygame.display.set_mode((640, 480))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        self.player1 = Paddle(self.screen, 5)
        self.player2 = Paddle(self.screen, 625)
        self.ball = Ball(self.screen)

    def update(self):
        self.player1.update()
        self.player2.update()
        reward, done, score = self.ball.update(self.player1, self.player2)
        return reward, done, score

    def render(self):
        self.screen.fill((0, 0, 0))
        self.player1.render()
        self.player2.render()
        self.ball.render()
        textSurface = self.myFont.render(f'AI {self.player1.score}:{self.player2.score} YOU', False, (255, 255, 255))
        self.screen.blit(textSurface, (250, 20))
        pygame.display.flip()

    def getState(self):
        state = [
            round(self.player1.y / (480 - self.player1.height), 2),
            round(self.player2.y / (480 - self.player2.height), 2),
            round(self.ball.y / 480, 2),
            round(self.ball.x / 640, 2),
            round(self.ball.angle / 255, 2),
        ]

        return np.array(state, dtype=np.float)

    def run(self):
        reward, done, score = self.update()
        if self.play:
            self.render()
            self.clock.tick(60)
        return reward, done, score


class Ball:
    def __init__(self, screen):
        self.screen = screen
        self.x = 320
        self.y = 240
        self.angle = random.randint(-45, 45) + 180 * random.randint(0, 1)
        self.speed = 8
        self.radius = 6
        self.scoreDiff = 0

    def update(self, player1, player2):
        reward = 0
        done = False
        
        # Check if ball hits the top or bottom
        if self.y + self.radius > 480 or self.y - self.radius < 0:
            self.angle = -self.angle

        # left collide
        if self.x - self.radius >= player1.x and self.x - self.radius <= player1.x + player1.width:
            if self.y - player1.y >= -self.radius:
                if self.y - player1.y <= 1/8 * (player1.height + self.radius):
                    self.angle = -45

                elif self.y - player1.y <= 2/8 * (player1.height + self.radius):
                    self.angle = -30

                elif self.y - player1.y <= 3/8 * (player1.height + self.radius):
                    self.angle = -15

                elif self.y - player1.y <= 4/8 * (player1.height + self.radius):
                    self.angle = -10

                elif self.y - player1.y <= 5/8 * (player1.height + self.radius):
                    self.angle = 10

                elif self.y - player1.y <= 6/8 * (player1.height + self.radius):
                    self.angle = 15

                elif self.y - player1.y <= 7/8 * (player1.height + self.radius):
                    self.angle = 30

                elif self.y - player1.y <= 8/8 * (player1.height + self.radius):
                    self.angle = 45
                reward = 0.1

        # right collide
        elif self.x + self.radius >= player2.x and self.x + self.radius <= player2.x + player2.width:
            if self.y - player2.y >= -self.radius:
                if self.y - player2.y <= 1/8 * (player2.height + self.radius):
                    self.angle = -135

                elif self.y - player2.y <= 2/8 * (player2.height + self.radius):
                    self.angle = -150

                elif self.y - player2.y <= 3/8 * (player2.height + self.radius):
                    self.angle = -165

                elif self.y - player2.y <= 4/8 * (player2.height + self.radius):
                    self.angle = 170

                elif self.y - player2.y <= 5/8 * (player2.height + self.radius):
                    self.angle = 190

                elif self.y - player2.y <= 6/8 * (player2.height + self.radius):
                    self.angle = 165

                elif self.y - player2.y <= 7/8 * (player2.height + self.radius):
                    self.angle = 150

                elif self.y - player2.y <= 8/8 * (player2.height + self.radius):
                    self.angle = 135

        self.y += self.speed*sin(radians(self.angle))
        self.x += self.speed*cos(radians(self.angle))

        # Check if the Ball went right
        if self.x - self.radius >= 670:
            player1.score += 1
            reward = 1
            self.scoreDiff += 1
            self.x = player2.x - player2.width * 2 - self.radius
            self.y = 240
            self.angle = 180

        temp = self.scoreDiff
        
        # Check if the Ball went left
        if self.x + self.radius <= -30:
            player2.score += 1
            reward = -1
            if player2.score % 5 == 0:
                done = True
                self.scoreDiff = 0
            self.x = player1.x + player1.width * 2 + self.radius
            self.y = 240
            self.angle = 0

        return reward, done, temp

    def render(self):
        pygame.draw.circle(self.screen, (255, 255, 255), (self.x, self.y), self.radius)


class Paddle:
    def __init__(self, screen, x):
        self.screen = screen
        self.x = x
        self.speed = 4
        self.width = 10
        self.height = 80
        self.y = 240 - self.height / 2
        self.score = 0
        self.mode = 0

    def update(self):
        self.y += self.mode * self.speed
        self.y = max(0, min(self.y, 480 - self.height))

    def render(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (self.x, self.y, self.width, self.height))