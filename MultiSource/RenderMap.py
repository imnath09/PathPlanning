
from MultiMapBase import *
from MultipleReversal import *

import gym
from gym.envs.classic_control import rendering

PIXEL = 30

class RenderMap(gym.Env, MultipleReversal):
    def __init__(self):
        gym.Env.__init__(self)
        super().__init__()
        # super(子类，self).__init__(参数1，参数2，....) 或者 super().__init__(参数1，参数2，....)
        # 父类.__init__(self,参数1，参数2，...)

        self.viewer = rendering.Viewer(self.height * PIXEL, self.width * PIXEL)

        # 画线
        for i in range(self.width + 1):
            line = rendering.Line((0, PIXEL * i), (PIXEL * self.height, PIXEL * i))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)
        for i in range(self.height + 1):
            line = rendering.Line((PIXEL * i, 0), (PIXEL * i, PIXEL * self.width))
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

        # 画障碍
        for o in self.obstacles:
            pl = self.draw_shape(o)
            self.viewer.add_geom(pl)

        # 移动点
        self.colors = {}
        self.sources = {}
        for src in self.srcs:
            # 起始点
            t = self.draw_shape(src.cur, color = (0, 0, 1), shape = 'circle', radius = 0.3)
            self.viewer.add_geom(t)
            # 随机颜色
            color = np.array([np.random.random(), np.random.random(), np.random.random()])
            self.colors[src] = color
            # 反差颜色
            opscolor = (1, 1, 1) - color
            # 移动点
            tr = rendering.Transform()
            p = (src.cur + (0.5, 0.5)) * PIXEL
            tr.set_translation(p[0], p[1])
            agt = self.draw_shape(
                src.cur, trans = tr, color = opscolor, 
                shape = 'circle', radius = 0.4)
            self.viewer.add_geom(agt)
            self.sources[src] = tr

        self.viewer.render()

    def add_block(self, src, pos):
        color = self.colors[src]
        p = self.draw_shape(pos, color = color, radius = 0.2)
        self.viewer.add_geom(p)
        self.viewer.render()
        #time.sleep(0.1)

    def move_block(self, src, pos):
        trans = self.sources[src]
        newpos = (pos + (0.5, 0.5)) * PIXEL
        trans.set_translation(newpos[0], newpos[1])
        self.viewer.render()
        #time.sleep(0.1)

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

if __name__ == '__main__':
    rm = RenderMap()
    rm.explore()
    input('press any key to exit')
