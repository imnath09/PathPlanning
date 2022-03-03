
import time
from Maze.unrenderedmaze import *
import gym

from gym.envs.classic_control import rendering


class GymMaze(gym.Env, UnrenderedMaze):
    def __init__(self):
        gym.Env.__init__(self)
        UnrenderedMaze.__init__(self)
        self.path_polygon = []

        self.viewer = rendering.Viewer(MAZE_WIDTH * PIXEL, MAZE_HEIGHT * PIXEL)
        # 画线
        for i in range(MAZE_HEIGHT + 1):
            line = rendering.Line((0, PIXEL * i), (PIXEL * MAZE_WIDTH, PIXEL * i))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)
        for i in range(MAZE_WIDTH + 1):
            line = rendering.Line((PIXEL * i, 0), (PIXEL * i, PIXEL * MAZE_HEIGHT))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)
        # 画障碍
        for o in self.obstacles:
            pl = self.draw_shape(o)
            self.viewer.add_geom(pl)
        # 起点
        start = self.draw_shape(START, color = (0, 0, 1), shape = 'circle')
        self.viewer.add_geom(start)
        # 终点
        dest = self.draw_shape(DESTINATION, color = (1, 0, 0))
        self.viewer.add_geom(dest)
        # 智能体
        self.agent_trans = rendering.Transform()
        self.agent = self.draw_shape(
            self.walker,
            self.agent_trans,
            (1, 1, 0),
            'circle',
            0.3)
        self.viewer.add_geom(self.agent)

        self.viewer.render()

    def reset(self):
        pos = UnrenderedMaze.reset(self)
        p = (pos + (0.5, 0.5)) * PIXEL
        self.agent_trans.set_translation(p[0], p[1])
        self.viewer.render()
        return pos

    def step(self, action : int):
        pos, reward, done, info = UnrenderedMaze.step(self, action)
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
            self.viewer.add_geom(pl)
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
        return pl

