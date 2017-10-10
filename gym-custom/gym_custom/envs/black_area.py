import gym
from gym import utils, spaces
from six import StringIO, b
from gym.utils import seeding
import sys
import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = ["WWWWWWWW",
	"WWSWWBWW",
	"WWBBBBBW",
	"BBBBWBWW",
	"WBBBWBBW",
	"WWBBBBWW",
	"WWWBBBBB",
	"WWWWWWBW"]

class BlackAreaEnvV1(gym.Env):

    """
     nA: number of action
     nS: number of state
     sp: start position
     P: list of which next start is led to from state and action
    """


    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.reward = 0
        self.map = MAPS[:]
        self.nrow, self.ncol = nrow, ncol = 8, 8
        self.nA = nA = 4
        self.nS = nS = nrow * ncol

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    r,c = inc(row, col, a)
                    nextS = to_s(r,c)
                    self.P[s][a] = nextS, self.map[r][c]
        self.sp = 10
        self._seed()
        self._reset()
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.map = MAPS[:]
        self.sp = 10
        self.reward = 0
        def to_s(row, col):
            return row*self.ncol + col

        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,self.nrow-1)
            elif a==2: # right
                col = min(col+1,self.ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    r,c = inc(row, col, a)
                    nextS = to_s(r,c)
                    self.P[s][a] = nextS, self.map[r][c]
        self.lastaction=None
        return self.sp

    def _step(self, action):
        
        row, col = self.sp // self.ncol, self.sp % self.ncol
        self.map[row] = list(self.map[row])
        self.map[row][col] = 'W'
        self.map[row] = "".join(self.map[row])

        def to_s(row, col):
            return row*self.ncol + col

        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,self.nrow-1)
            elif a==2: # right
                col = min(col+1,self.ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    r,c = inc(row, col, a)
                    nextS = to_s(r,c)
                    self.P[s][a] = nextS, self.map[r][c]
        self.sp, land = self.P[self.sp][action]
        done = False
        if land == 'W':
            done = True
        else:
            self.reward += 1
        self.lastaction = action
        return ((self.sp, land), self.reward, done, "")

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        row, col = self.sp // self.ncol, self.sp % self.ncol

        outfile.write("\n".join(''.join(line) for line in self.map)+"\n")

        if mode != 'human':
            return outfile


