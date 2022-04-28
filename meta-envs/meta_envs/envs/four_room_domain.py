import gym
from gym import error, spaces
import numpy as np
import random
from matplotlib import pyplot as plt

import pygame
from torch import rand

class World:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=np.int16)
        self.set_grid()
        n_goal = random.randint(1, 5)
        n_penalty = random.randint(1, 2)
        self.set_agent_pos()
        self.set_reward_pos(n_goal=n_goal, n_penalty=n_penalty)
        # self.penalty = (-1, -1)
    
    def set_grid(self):
        for i in range(self.n):
            self.grid[i, 0] = 1
            self.grid[i, self.n-1] = 1
            self.grid[0, i] = 1
            self.grid[self.n-1, i] = 1
        
        for i in range(self.n):
            self.grid[i, self.n // 2] = 1
        
        for i in range(self.n // 2):
            self.grid[self.n // 2, i] = 1
        for i in range(self.n // 2):
            self.grid[(self.n // 2)+1, self.n-i-1] = 1

        # Gates
        self.grid[self.n // 4, self.n // 2] = 0
        self.grid[self.n - self.n // 4, self.n // 2] = 0
        self.grid[self.n // 2, self.n // 4] = 0
        self.grid[(self.n //2)+1, self.n // 2 +  self.n // 4] = 0

        
    
    def set_agent_pos(self):
        while True:
            x = random.randint(0, self.n-1)
            y = random.randint(0, self.n-1)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 2
                self.agent_pos = (x, y)
                break

    def set_reward_pos(self, n_goal=1, n_penalty=1):

        n_g = n_goal
        n_p = n_penalty
        self.n_g = n_g

        while n_p > 0:
            x = random.randint(0, self.n-1)
            y = random.randint(0, self.n-1)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 4
                n_p -= 1
        
        while n_g > 0:
            x = random.randint(0, self.n-1)
            y = random.randint(0, self.n-1)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 3
                n_g -= 1
        





class Grid(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, n: int):
        super().__init__()

        self.n = n

        # 0=empty    1=wall    2=agent    3=goal    4=penalty
        self.observation_space = spaces.Box(low=0, high=4, shape=(n, n), dtype=np.int16)
        # 0=left   1=right    2=up    3=down
        self.action_space = spaces.Discrete(4)
        self.world = World(n)
        self.window = None
        self.clock = None
        self.window_size = 512
        self.penalty = (-1, -1)

    def step(self, action):
        x, y = self.world.agent_pos

        if action == 0:
            x_dash = x
            y_dash = y - 1

        elif action == 1:
            x_dash = x
            y_dash = y + 1
        
        elif action == 2:
            x_dash = x - 1
            y_dash = y
        
        elif action == 3:
            x_dash = x + 1
            y_dash = y
        
        else:
            raise error.InvalidAction('Invalid action')
        
        if not self.is_valid_action(x_dash, y_dash):
            return self.get_obs(), -1, True, {}
        
        reward = self.get_reward(x_dash, y_dash)
        if self.world.grid[x_dash, y_dash] == 4:
            self.penalty = (x_dash, y_dash)
            
        self.world.grid[x, y] = 0
        self.world.grid[x_dash, y_dash] = 2
        self.world.agent_pos = (x_dash, y_dash)
        if self.penalty[0] == x and self.penalty[1] == y:
            self.world.grid[x, y] = 4
            self.penalty = (-1, -1)
        done = False
        if reward == 100:
            self.world.n_g -= 1
        
        if self.world.n_g == 0:
            done = True
        # if reward == -10:
            # self.world.set_reward_pos(0, 1)
        return self.get_obs(), reward, done, {}


    def reset(self):
        self.world = World(self.n)
        return self.get_obs()


    def is_valid_action(self, x, y):
        if x == 0 or x == self.world.n-1 or y == 0 or y == self.world.n:
            return False
        if self.world.grid[x, y] == 1:
            return False
        return True
    
    def get_reward(self, x, y):
        if self.world.grid[x, y] == 3:
            return 100
        elif self.world.grid[x, y] == 4:
            return -10
        else:
            return 0

    def get_obs(self):
        return self.world.grid

    def render(self, mode='human'):
        
        # mapping = [(255, 255, 255), (0, 0, 0), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        # img_array = self.convert_to_rgb_array(self.world.grid, mapping)

        # if(mode == 'rgb_array'):
        #     return img_array


        # plt.figure()
        # plt.imshow(img_array)
        # plt.show()
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.n
        )  # The size of a single grid square in pixels

        self.draw_grid(canvas, pix_square_size)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    
    def convert_to_rgb_array(self, grid, mapping):
        img_array = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                img_array[i, j] = mapping[grid[i, j]]
        return img_array
    

    def draw_grid(self, canvas, pix_square_size):
        mapping = [(255, 255, 255), (0, 0, 0), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        
        for i in range(self.n):
            for j in range(self.n):
                map_idx = self.world.grid[i, j]
                pygame.draw.rect(
                    canvas,
                    mapping[map_idx],
                    pygame.Rect(
                        pix_square_size * np.array([i, j]),
                        (pix_square_size, pix_square_size),
                    ),
                )
        
        for x in range(self.n + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
    

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

                        