import pygame
import random
import numpy as np
from math import cos, sin, radians

class Game:
    def __init__(self, play, train):
        self.play = play
        self.train = train
        if self.play:
            pygame.init()
            pygame.display.set_caption('Pong')
            pygame.display.set_icon(pygame.image.load('assets/icon.png'))
            self.myFont = pygame.font.SysFont('arial', 30)
            self.screen = pygame.display.set_mode((640, 480))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        self.player1 = Paddle(self.screen, 5, [-45, -30, -15, -10, 10, 15, 30, 45])
        self.player2 = Paddle(self.screen, 625, [225, 210, 195, 190, 170, 165, 150, 135]) # 135 -> 225 | 180 + (180 - x)
        self.ball = Ball(self.screen)

    def update(self):
        self.player1.update()
        self.player2.update()
        reward, done = self.ball.update(self.player1, self.player2, self.train)
        return reward, done

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
        reward, done = self.update()
        if self.play:
            self.render()
            self.clock.tick(60)
        return reward, done

class Ball:
    def __init__(self, screen):
        self.screen = screen
        self.frame = 0
        self.x = 320
        self.y = 240
        self.angle = random.choice([-45, -30, -15, -10, 10, 15, 30, 45]) + 180 * random.randint(0, 1)
        self.speed = 8
        self.radius = 6

    def update(self, player1, player2, train):
        reward = 0
        done = False
        if train:
            self.frame += 1
        
        # Check if ball hits the top or bottom
        if self.y + self.radius > 480 or self.y - self.radius < 0:
            if self.angle <= 45:
                self.angle = -self.angle
            else:
                self.angle = 360 - self.angle

        # left collide
        if self.x - self.radius >= player1.x and self.x - self.radius <= player1.x + player1.width:
            if self.y - player1.y >= -self.radius:
                for i in range(len(player1.angles)):
                    if self.y - player1.y <= (i+1)/len(player1.angles) * (player1.height + self.radius):
                        self.angle = player1.angles[i]
                        break
                reward = 2

        # right collide
        elif self.x + self.radius >= player2.x and self.x + self.radius <= player2.x + player2.width:
            if self.y - player2.y >= -self.radius:
                for i in range(len(player2.angles)):
                    if self.y - player2.y <= (i+1)/len(player2.angles) * (player2.height + self.radius):
                        self.angle = player2.angles[i]
                        break

        self.y += self.speed*sin(radians(self.angle))
        self.x += self.speed*cos(radians(self.angle))

        # Check if the Ball went right
        if self.x - self.radius >= 670:
            player1.score += 1
            reward = 10
            self.x = player2.x - player2.width * 2 - self.radius
            self.y = 240
            self.angle = random.choice(player2.angles[2:-2])
            self.frame = 0
        
        # Check if the Ball went left
        if self.x + self.radius <= -30 or self.frame > 1000:
            player2.score += 1
            reward = -10
            if player2.score % 5 == 0:
                done = True
            self.x = player1.x + player1.width * 2 + self.radius
            self.y = 240
            self.angle = random.choice(player1.angles[2:-2])
            self.frame = 0

        return reward, done

    def render(self):
        pygame.draw.circle(self.screen, (255, 255, 255), (self.x, self.y), self.radius)

class Paddle:
    def __init__(self, screen, x, angles):
        self.angles = angles
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