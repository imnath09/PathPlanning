

import sys
sys.path.append('..')

from MAPs.Map import *

import gym
from gym.envs.classic_control import rendering

class GymMaze(gym.Env, MapBase):
    def __init__(self, map = map1):
        gym.Env.__init__(self)
        MapBase.__init__(self, map = map)
        self.path_polygon = []# 成功路径

        self.viewer = rendering.Viewer(self.height * PIXEL, self.width * PIXEL)
        # 画线
        '''
        for i in range(self.width + 1):
            line = rendering.Line((0, PIXEL * i), (PIXEL * self.height, PIXEL * i))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)
        for i in range(self.height + 1):
            line = rendering.Line((PIXEL * i, 0), (PIXEL * i, PIXEL * self.width))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)
        '''
        # 只画边线
        '''line = rendering.Line((1, 1), (PIXEL * self.height - 1, 1))
        line.set_color(0, 0, 0)
        self.viewer.add_geom(line)
        line = rendering.Line((1, PIXEL * self.width - 1), (PIXEL * self.height - 1, PIXEL * self.width - 1))
        line.set_color(0, 0, 0)
        self.viewer.add_geom(line)
        line = rendering.Line((1.1, 1), (1.1, PIXEL * self.width - 1))
        line.set_color(0, 0, 0)
        self.viewer.add_geom(line)
        line = rendering.Line((PIXEL * self.height - 1, 1), (PIXEL * self.height - 1, PIXEL * self.width - 1))
        line.set_color(0, 0, 0)
        self.viewer.add_geom(line)'''
        # 画障碍
        for o in self.obstacles:
            pl = self.draw_shape(o, radius = 0.5)
        # 起点
        start = self.draw_shape(self.start, color = (0, 0, 1), shape = 'circle')
        # 终点
        dest = self.draw_shape(self.destination, color = (1, 0, 0))
        # 智能体
        self.agent_trans = rendering.Transform()
        p = (self.walker + (0.5, 0.5)) * PIXEL
        self.agent_trans.set_translation(p[0], p[1])
        self.agent = self.draw_shape(
            self.walker,
            trans = self.agent_trans,
            color = (1, 1, 0),
            shape = 'circle',
            radius = 0.3)

        self.viewer.render()

    def draw_walkers(self, walkers):
        # 移动点
        aaa=[np.array([0.5,0.2,0.1]),
             np.array([0.9,0.7,0.3]),
             np.array([0.2,0.6,0.9]),
             np.array([0.1,0.9,0.1]),
             np.array([0.1,0.9,0.1]),
             np.array([0.1,0.9,0.1]),
             np.array([0.1,0.9,0.1]),
             ]
        i = 0
        self.colors = {}
        self.sources = {}
        #ss = self.srcs+[self.finalsource] if self.finalsource is not None else self.srcs
        for src in walkers:
            # 随机颜色
            color = aaa[i]
            i += 1
            #color = np.array([np.random.random(), np.random.random(), np.random.random()])
            # 起始点
            t = self.draw_shape(src.cur, color = color, radius = 0.5) # shape = 'circle', 
            self.colors[src] = color
            # 反差颜色
            #opscolor = (1, 1, 1) - color
            # 移动点
            tr = rendering.Transform()
            p = (src.cur + (0.5, 0.5)) * PIXEL
            tr.set_translation(p[0], p[1])
            '''agt = self.draw_shape(
                src.cur, trans = tr, color = opscolor, 
                shape = 'circle', radius = 0.4)'''
            self.sources[src] = tr

    def reset(self):
        pos = MapBase.reset(self)
        p = (pos + (0.5, 0.5)) * PIXEL
        self.agent_trans.set_translation(p[0], p[1])
        self.viewer.render()
        return pos

    def step(self, action : int):
        pos, reward, done, info = MapBase.step(self, action)
        p = (pos + (0.5, 0.5)) * PIXEL
        self.agent_trans.set_translation(p[0], p[1])
        return pos, reward, done, info

    def render(self):
        #time.sleep(1)
        if self.new_sln:
            self.draw_path()
        self.viewer.render()

##################################################

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def draw_path(self):
        #if ((self.path_polygon.__len__() > self.maze.s_path.__len__()) or
        #    (self.path_polygon.__len__() == 0 and self.maze.s_path.__len__() > 0)):
        for pl in self.path_polygon:
            self.viewer.geoms.remove(pl)
        self.path_polygon.clear()
        for p in self.cur_path:
            pl = self.draw_shape(p, radius = 0.2, color = (0, 0, 0.5), shape = 'circle')
            self.path_polygon.append(pl)

    def draw_shape(self, pos, trans = None, color = (0, 0, 0), shape = 'polygon', radius = 0.4):
        if shape == 'polygon':
            pl = rendering.make_polygon([
                (radius * PIXEL, radius * PIXEL), (radius * PIXEL, -radius * PIXEL),
                (-radius * PIXEL, -radius * PIXEL), (-radius * PIXEL, radius * PIXEL)],
                filled = True)
        else:
            pl = rendering.make_circle(radius * PIXEL, 20, filled = True)
        pl.set_color(color[0], color[1], color[2])
        if trans is None:
            trans = rendering.Transform(translation = tuple((pos + (0.5, 0.5)) * PIXEL))
        pl.add_attr(trans)
        self.viewer.add_geom(pl)
        return pl

if __name__ == '__main__':
    env = GymMaze()
    action = 0
    while True:
        di = input('actions:')
        if di == 'w':
            action = 0
        elif di == 's':
            action = 1
        elif di == 'a':
            action = 2
        elif di == 'd':
            action = 3

        env.step(action)
        env.render()
