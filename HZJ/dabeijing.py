import numpy as np  # To deal with data in form of matrices
import tkinter as tk  # To build GUI
import time  # Time is needed to slow down the agent and to see how he runs
from PIL import Image, ImageTk  # For adding images into the canvas widget
import math

# Setting the sizes for the environment
pixels = 20   # pixels
env_height = 25  # grid height
env_width = 17  # grid width

# Global variable for dictionary with coordinates for the final route
a = {}


# Creating class for the environment
class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right',]#'up-left','up-right','down-left','down-right']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('DQN')
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))
        self.build_environment()
        self.height = env_height
        self.width = env_width

        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}

        # Key for the dictionaries
        self.i = 0
        self.g = 0
        self.distance = 0
        # self.middle = 0
        self.middle = float('inf')
        self.dist = 0
        self.l = 0
        self.a = 0
        # self.a0 = 0
        self.a0 = float('inf')


        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for longest found route
        self.longest = 0
        self.s = 0
        self.ss = 0
        self.lo = 0
        self.ll = 0

        # Showing the steps for the shortest route
        self.shortest = 0

    # Function to build the environment
    def build_environment(self):
        self.canvas_widget = tk.Canvas(self,  bg='white',
                                       height=env_height * pixels,
                                       width=env_width * pixels)

        # # Uploading an image for background
        # img_background = Image.open("images/bg.png")
        # self.background = ImageTk.PhotoImage(img_background)
        # # Creating background on the widget
        # self.bg = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.background)

        # Creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='white')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='white')

        # Creating objects of  Obstacles
        # An array to help with building rectangles
        self.o = np.array([pixels / 2, pixels / 2])

        obstacle1_center = self.o + np.array([pixels * 1, 0])
        self.obstacle1 = self.canvas_widget.create_rectangle(
            obstacle1_center[0] - 10, obstacle1_center[1] - 10,  # Top left corner
            obstacle1_center[0] + 10, obstacle1_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle1 = [self.canvas_widget.coords(self.obstacle1)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle1)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle1)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle1)[3] - 3]

        obstacle2_center = self.o + np.array([pixels * 2, 0])
        self.obstacle2 = self.canvas_widget.create_rectangle(
            obstacle2_center[0] - 10, obstacle2_center[1] - 10,  # Top left corner
            obstacle2_center[0] + 10, obstacle2_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle2 = [self.canvas_widget.coords(self.obstacle2)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle2)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle2)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle2)[3] - 3]

        obstacle3_center = self.o + np.array([pixels * 3, 0])
        self.obstacle3 = self.canvas_widget.create_rectangle(
            obstacle3_center[0] - 10, obstacle3_center[1] - 10,  # Top left corner
            obstacle3_center[0] + 10, obstacle3_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle3 = [self.canvas_widget.coords(self.obstacle3)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle3)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle3)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle3)[3] - 3]

        obstacle4_center = self.o + np.array([pixels * 4, 0])
        self.obstacle4 = self.canvas_widget.create_rectangle(
            obstacle4_center[0] - 10, obstacle4_center[1] - 10,  # Top left corner
            obstacle4_center[0] + 10, obstacle4_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle4 = [self.canvas_widget.coords(self.obstacle4)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle4)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle4)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle4)[3] - 3]

        obstacle5_center = self.o + np.array([pixels * 5, 0])
        self.obstacle5 = self.canvas_widget.create_rectangle(
            obstacle5_center[0] - 10, obstacle5_center[1] - 10,  # Top left corner
            obstacle5_center[0] + 10, obstacle5_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle5 = [self.canvas_widget.coords(self.obstacle5)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle5)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle5)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle5)[3] - 3]

        obstacle6_center = self.o + np.array([pixels * 6, 0])
        self.obstacle6 = self.canvas_widget.create_rectangle(
            obstacle6_center[0] - 10, obstacle6_center[1] - 10,  # Top left corner
            obstacle6_center[0] + 10, obstacle6_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle6 = [self.canvas_widget.coords(self.obstacle6)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle6)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle6)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle6)[3] - 3]

        obstacle7_center = self.o + np.array([pixels * 7, 0])
        self.obstacle7 = self.canvas_widget.create_rectangle(
            obstacle7_center[0] - 10, obstacle7_center[1] - 10,  # Top left corner
            obstacle7_center[0] + 10, obstacle7_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle7 = [self.canvas_widget.coords(self.obstacle7)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle7)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle7)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle7)[3] - 3]

        obstacle8_center = self.o + np.array([pixels * 8, 0])
        self.obstacle8 = self.canvas_widget.create_rectangle(
            obstacle8_center[0] - 10, obstacle8_center[1] - 10,  # Top left corner
            obstacle8_center[0] + 10, obstacle8_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle8 = [self.canvas_widget.coords(self.obstacle8)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle8)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle8)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle8)[3] - 3]

        obstacle9_center = self.o + np.array([pixels * 9, 0])
        self.obstacle9 = self.canvas_widget.create_rectangle(
            obstacle9_center[0] - 10, obstacle9_center[1] - 10,  # Top left corner
            obstacle9_center[0] + 10, obstacle9_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle9 = [self.canvas_widget.coords(self.obstacle9)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle9)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle9)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle9)[3] - 3]

        obstacle10_center = self.o + np.array([pixels * 10, 0])
        self.obstacle10 = self.canvas_widget.create_rectangle(
            obstacle10_center[0] - 10, obstacle10_center[1] - 10,  # Top left corner
            obstacle10_center[0] + 10, obstacle10_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle10 = [self.canvas_widget.coords(self.obstacle10)[0] + 3,
                                 self.canvas_widget.coords(self.obstacle10)[1] + 3,
                                 self.canvas_widget.coords(self.obstacle10)[2] - 3,
                                 self.canvas_widget.coords(self.obstacle10)[3] - 3]

        obstacle11_center = self.o + np.array([pixels * 11, 0])
        self.obstacle11 = self.canvas_widget.create_rectangle(
            obstacle11_center[0] - 10, obstacle11_center[1] - 10,  # Top left corner
            obstacle11_center[0] + 10, obstacle11_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle11 = [self.canvas_widget.coords(self.obstacle11)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle11)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle11)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle11)[3] - 3]

        obstacle12_center = self.o + np.array([pixels * 12, 0])
        self.obstacle12 = self.canvas_widget.create_rectangle(
            obstacle12_center[0] - 10, obstacle12_center[1] - 10,  # Top left corner
            obstacle12_center[0] + 10, obstacle12_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle12 = [self.canvas_widget.coords(self.obstacle12)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle12)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle12)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle12)[3] - 3]

        obstacle13_center = self.o + np.array([pixels * 13, 0])
        self.obstacle13 = self.canvas_widget.create_rectangle(
            obstacle13_center[0] - 10, obstacle13_center[1] - 10,  # Top left corner
            obstacle13_center[0] + 10, obstacle13_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle13 = [self.canvas_widget.coords(self.obstacle13)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle13)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle13)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle13)[3] - 3]

        obstacle14_center = self.o + np.array([pixels * 14, 0])
        self.obstacle14 = self.canvas_widget.create_rectangle(
            obstacle14_center[0] - 10, obstacle14_center[1] - 10,  # Top left corner
            obstacle14_center[0] + 10, obstacle14_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle14 = [self.canvas_widget.coords(self.obstacle14)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle14)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle14)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle14)[3] - 3]

        obstacle15_center = self.o + np.array([pixels * 15, 0])
        self.obstacle15 = self.canvas_widget.create_rectangle(
            obstacle15_center[0] - 10, obstacle15_center[1] - 10,  # Top left corner
            obstacle15_center[0] + 10, obstacle15_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle15 = [self.canvas_widget.coords(self.obstacle15)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle15)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle15)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle15)[3] - 3]

        obstacle16_center = self.o + np.array([pixels * 16, 0])
        self.obstacle16 = self.canvas_widget.create_rectangle(
            obstacle16_center[0] - 10, obstacle16_center[1] - 10,  # Top left corner
            obstacle16_center[0] + 10, obstacle16_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle16 = [self.canvas_widget.coords(self.obstacle16)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle16)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle16)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle16)[3] - 3]

        # obstacle17_center = self.o + np.array([pixels * 17, 0])
        # self.obstacle17 = self.canvas_widget.create_rectangle(
        #     obstacle17_center[0] - 10, obstacle17_center[1] - 10,  # Top left corner
        #     obstacle17_center[0] + 10, obstacle17_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle17 = [self.canvas_widget.coords(self.obstacle17)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle17)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle17)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle17)[3] - 3]
        #
        # obstacle18_center = self.o + np.array([pixels * 18, 0])
        # self.obstacle18 = self.canvas_widget.create_rectangle(
        #     obstacle18_center[0] - 10, obstacle18_center[1] - 10,  # Top left corner
        #     obstacle18_center[0] + 10, obstacle18_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle18 = [self.canvas_widget.coords(self.obstacle18)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle18)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle18)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle18)[3] - 3]
        #
        # obstacle19_center = self.o + np.array([pixels * 19, 0])
        # self.obstacle19 = self.canvas_widget.create_rectangle(
        #     obstacle19_center[0] - 10, obstacle19_center[1] - 10,  # Top left corner
        #     obstacle19_center[0] + 10, obstacle19_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle19 = [self.canvas_widget.coords(self.obstacle19)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle19)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle19)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle19)[3] - 3]
        #
        # obstacle20_center = self.o + np.array([pixels * 20, 0])
        # self.obstacle20 = self.canvas_widget.create_rectangle(
        #     obstacle20_center[0] - 10, obstacle20_center[1] - 10,  # Top left corner
        #     obstacle20_center[0] + 10, obstacle20_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle20 = [self.canvas_widget.coords(self.obstacle20)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle20)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle20)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle20)[3] - 3]
        #
        # obstacle21_center = self.o + np.array([pixels * 21, 0])
        # self.obstacle21 = self.canvas_widget.create_rectangle(
        #     obstacle21_center[0] - 10, obstacle21_center[1] - 10,  # Top left corner
        #     obstacle21_center[0] + 10, obstacle21_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle21 = [self.canvas_widget.coords(self.obstacle21)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle21)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle21)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle21)[3] - 3]
        #
        # obstacle22_center = self.o + np.array([pixels * 22, 0])
        # self.obstacle22 = self.canvas_widget.create_rectangle(
        #     obstacle22_center[0] - 10, obstacle22_center[1] - 10,  # Top left corner
        #     obstacle22_center[0] + 10, obstacle22_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle22 = [self.canvas_widget.coords(self.obstacle22)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle22)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle22)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle22)[3] - 3]
        #
        # obstacle23_center = self.o + np.array([pixels * 22, pixels * 7])
        # self.obstacle23 = self.canvas_widget.create_rectangle(
        #     obstacle23_center[0] - 10, obstacle23_center[1] - 10,  # Top left corner
        #     obstacle23_center[0] + 10, obstacle23_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle23 = [self.canvas_widget.coords(self.obstacle23)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle23)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle23)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle23)[3] - 3]
        #
        # obstacle24_center = self.o + np.array([pixels * 22, pixels * 8])
        # self.obstacle24 = self.canvas_widget.create_rectangle(
        #     obstacle24_center[0] - 10, obstacle24_center[1] - 10,  # Top left corner
        #     obstacle24_center[0] + 10, obstacle24_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle24 = [self.canvas_widget.coords(self.obstacle24)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle24)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle24)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle24)[3] - 3]
        #
        # obstacle25_center = self.o + np.array([pixels * 22, pixels * 9])
        # self.obstacle25 = self.canvas_widget.create_rectangle(
        #     obstacle25_center[0] - 10, obstacle25_center[1] - 10,  # Top left corner
        #     obstacle25_center[0] + 10, obstacle25_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle25 = [self.canvas_widget.coords(self.obstacle25)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle25)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle25)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle25)[3] - 3]
        #
        # obstacle26_center = self.o + np.array([pixels * 22, pixels * 10])
        # self.obstacle26 = self.canvas_widget.create_rectangle(
        #     obstacle26_center[0] - 10, obstacle26_center[1] - 10,  # Top left corner
        #     obstacle26_center[0] + 10, obstacle26_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle26 = [self.canvas_widget.coords(self.obstacle26)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle26)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle26)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle26)[3] - 3]
        #
        # obstacle27_center = self.o + np.array([pixels * 22, pixels * 11])
        # self.obstacle27 = self.canvas_widget.create_rectangle(
        #     obstacle27_center[0] - 10, obstacle27_center[1] - 10,  # Top left corner
        #     obstacle27_center[0] + 10, obstacle27_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle27 = [self.canvas_widget.coords(self.obstacle27)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle27)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle27)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle27)[3] - 3]
        #
        # obstacle28_center = self.o + np.array([pixels * 22, pixels * 12])
        # self.obstacle28 = self.canvas_widget.create_rectangle(
        #     obstacle28_center[0] - 10, obstacle28_center[1] - 10,  # Top left corner
        #     obstacle28_center[0] + 10, obstacle28_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle28 = [self.canvas_widget.coords(self.obstacle28)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle28)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle28)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle28)[3] - 3]
        #
        # obstacle29_center = self.o + np.array([pixels * 22, pixels * 13])
        # self.obstacle29 = self.canvas_widget.create_rectangle(
        #     obstacle29_center[0] - 10, obstacle29_center[1] - 10,  # Top left corner
        #     obstacle29_center[0] + 10, obstacle29_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle29 = [self.canvas_widget.coords(self.obstacle29)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle29)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle29)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle29)[3] - 3]
        #
        # obstacle30_center = self.o + np.array([pixels * 22, pixels * 14])
        # self.obstacle30 = self.canvas_widget.create_rectangle(
        #     obstacle30_center[0] - 10, obstacle30_center[1] - 10,  # Top left corner
        #     obstacle30_center[0] + 10, obstacle30_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle30 = [self.canvas_widget.coords(self.obstacle30)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle30)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle30)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle30)[3] - 3]
        #
        # obstacle31_center = self.o + np.array([pixels * 22, pixels * 15])
        # self.obstacle31 = self.canvas_widget.create_rectangle(
        #     obstacle31_center[0] - 10, obstacle31_center[1] - 10,  # Top left corner
        #     obstacle31_center[0] + 10, obstacle31_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle31 = [self.canvas_widget.coords(self.obstacle31)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle31)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle31)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle31)[3] - 3]
        #
        # obstacle32_center = self.o + np.array([pixels * 22, pixels * 16])
        # self.obstacle32 = self.canvas_widget.create_rectangle(
        #     obstacle32_center[0] - 10, obstacle32_center[1] - 10,  # Top left corner
        #     obstacle32_center[0] + 10, obstacle32_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle32 = [self.canvas_widget.coords(self.obstacle32)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle32)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle32)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle32)[3] - 3]
        #
        # obstacle33_center = self.o + np.array([pixels * 22, pixels * 17])
        # self.obstacle33 = self.canvas_widget.create_rectangle(
        #     obstacle33_center[0] - 10, obstacle33_center[1] - 10,  # Top left corner
        #     obstacle33_center[0] + 10, obstacle33_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle33 = [self.canvas_widget.coords(self.obstacle33)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle33)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle33)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle33)[3] - 3]
        #
        # obstacle34_center = self.o + np.array([pixels * 22, pixels * 18])
        # self.obstacle34 = self.canvas_widget.create_rectangle(
        #     obstacle34_center[0] - 10, obstacle34_center[1] - 10,  # Top left corner
        #     obstacle34_center[0] + 10, obstacle34_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle34 = [self.canvas_widget.coords(self.obstacle34)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle34)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle34)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle34)[3] - 3]
        #
        # obstacle35_center = self.o + np.array([pixels * 21, pixels * 18])
        # self.obstacle35 = self.canvas_widget.create_rectangle(
        #     obstacle35_center[0] - 10, obstacle35_center[1] - 10,  # Top left corner
        #     obstacle35_center[0] + 10, obstacle35_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle35 = [self.canvas_widget.coords(self.obstacle35)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle35)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle35)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle35)[3] - 3]
        #
        # obstacle36_center = self.o + np.array([pixels * 20, pixels * 18])
        # self.obstacle36 = self.canvas_widget.create_rectangle(
        #     obstacle36_center[0] - 10, obstacle36_center[1] - 10,  # Top left corner
        #     obstacle36_center[0] + 10, obstacle36_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle36 = [self.canvas_widget.coords(self.obstacle36)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle36)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle36)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle36)[3] - 3]
        #
        # obstacle37_center = self.o + np.array([pixels * 19, pixels * 18])
        # self.obstacle37 = self.canvas_widget.create_rectangle(
        #     obstacle37_center[0] - 10, obstacle37_center[1] - 10,  # Top left corner
        #     obstacle37_center[0] + 10, obstacle37_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle37 = [self.canvas_widget.coords(self.obstacle37)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle37)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle37)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle37)[3] - 3]
        #
        # obstacle38_center = self.o + np.array([pixels * 19, pixels * 17])
        # self.obstacle38 = self.canvas_widget.create_rectangle(
        #     obstacle38_center[0] - 10, obstacle38_center[1] - 10,  # Top left corner
        #     obstacle38_center[0] + 10, obstacle38_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle38 = [self.canvas_widget.coords(self.obstacle38)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle38)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle38)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle38)[3] - 3]
        #
        # obstacle39_center = self.o + np.array([pixels * 19, pixels * 16])
        # self.obstacle39 = self.canvas_widget.create_rectangle(
        #     obstacle39_center[0] - 10, obstacle39_center[1] - 10,  # Top left corner
        #     obstacle39_center[0] + 10, obstacle39_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle39 = [self.canvas_widget.coords(self.obstacle39)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle39)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle39)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle39)[3] - 3]
        #
        # obstacle40_center = self.o + np.array([pixels * 19, pixels * 15])
        # self.obstacle40 = self.canvas_widget.create_rectangle(
        #     obstacle40_center[0] - 10, obstacle40_center[1] - 10,  # Top left corner
        #     obstacle40_center[0] + 10, obstacle40_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle40 = [self.canvas_widget.coords(self.obstacle40)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle40)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle40)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle40)[3] - 3]
        #
        # obstacle41_center = self.o + np.array([pixels * 19, pixels * 14])
        # self.obstacle41 = self.canvas_widget.create_rectangle(
        #     obstacle41_center[0] - 10, obstacle41_center[1] - 10,  # Top left corner
        #     obstacle41_center[0] + 10, obstacle41_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle41 = [self.canvas_widget.coords(self.obstacle41)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle41)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle41)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle41)[3] - 3]
        #
        # obstacle42_center = self.o + np.array([pixels * 19, pixels * 13])
        # self.obstacle42 = self.canvas_widget.create_rectangle(
        #     obstacle42_center[0] - 10, obstacle42_center[1] - 10,  # Top left corner
        #     obstacle42_center[0] + 10, obstacle42_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle42 = [self.canvas_widget.coords(self.obstacle42)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle42)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle42)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle42)[3] - 3]
        #
        # obstacle43_center = self.o + np.array([pixels * 19, pixels * 12])
        # self.obstacle43 = self.canvas_widget.create_rectangle(
        #     obstacle43_center[0] - 10, obstacle43_center[1] - 10,  # Top left corner
        #     obstacle43_center[0] + 10, obstacle43_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle43 = [self.canvas_widget.coords(self.obstacle43)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle43)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle43)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle43)[3] - 3]
        #
        # obstacle44_center = self.o + np.array([pixels * 19, pixels * 11])
        # self.obstacle44 = self.canvas_widget.create_rectangle(
        #     obstacle44_center[0] - 10, obstacle44_center[1] - 10,  # Top left corner
        #     obstacle44_center[0] + 10, obstacle44_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle44 = [self.canvas_widget.coords(self.obstacle44)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle44)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle44)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle44)[3] - 3]
        #
        # obstacle45_center = self.o + np.array([pixels * 19, pixels * 10])
        # self.obstacle45 = self.canvas_widget.create_rectangle(
        #     obstacle45_center[0] - 10, obstacle45_center[1] - 10,  # Top left corner
        #     obstacle45_center[0] + 10, obstacle45_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle45 = [self.canvas_widget.coords(self.obstacle45)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle45)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle45)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle45)[3] - 3]
        #
        # obstacle46_center = self.o + np.array([pixels * 19, pixels * 9])
        # self.obstacle46 = self.canvas_widget.create_rectangle(
        #     obstacle46_center[0] - 10, obstacle46_center[1] - 10,  # Top left corner
        #     obstacle46_center[0] + 10, obstacle46_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle46 = [self.canvas_widget.coords(self.obstacle46)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle46)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle46)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle46)[3] - 3]
        #
        # obstacle47_center = self.o + np.array([pixels * 19, pixels * 8])
        # self.obstacle47 = self.canvas_widget.create_rectangle(
        #     obstacle47_center[0] - 10, obstacle47_center[1] - 10,  # Top left corner
        #     obstacle47_center[0] + 10, obstacle47_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle47 = [self.canvas_widget.coords(self.obstacle47)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle47)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle47)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle47)[3] - 3]
        #
        # obstacle48_center = self.o + np.array([pixels * 19, pixels * 7])
        # self.obstacle48 = self.canvas_widget.create_rectangle(
        #     obstacle48_center[0] - 10, obstacle48_center[1] - 10,  # Top left corner
        #     obstacle48_center[0] + 10, obstacle48_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle48 = [self.canvas_widget.coords(self.obstacle48)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle48)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle48)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle48)[3] - 3]
        #
        # obstacle49_center = self.o + np.array([pixels * 20, pixels * 7])
        # self.obstacle49 = self.canvas_widget.create_rectangle(
        #     obstacle49_center[0] - 10, obstacle49_center[1] - 10,  # Top left corner
        #     obstacle49_center[0] + 10, obstacle49_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle49 = [self.canvas_widget.coords(self.obstacle49)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle49)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle49)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle49)[3] - 3]
        #
        # obstacle50_center = self.o + np.array([pixels * 21, pixels * 7])
        # self.obstacle50 = self.canvas_widget.create_rectangle(
        #     obstacle50_center[0] - 10, obstacle50_center[1] - 10,  # Top left corner
        #     obstacle50_center[0] + 10, obstacle50_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_obstacle50 = [self.canvas_widget.coords(self.obstacle50)[0] + 3,
        #                           self.canvas_widget.coords(self.obstacle50)[1] + 3,
        #                           self.canvas_widget.coords(self.obstacle50)[2] - 3,
        #                           self.canvas_widget.coords(self.obstacle50)[3] - 3]

        obstacle51_center = self.o + np.array([pixels * 15, pixels * 7])
        self.obstacle51 = self.canvas_widget.create_rectangle(
            obstacle51_center[0] - 10, obstacle51_center[1] - 10,  # Top left corner
            obstacle51_center[0] + 10, obstacle51_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle51 = [self.canvas_widget.coords(self.obstacle51)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle51)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle51)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle51)[3] - 3]

        obstacle52_center = self.o + np.array([pixels * 14, pixels * 7])
        self.obstacle52 = self.canvas_widget.create_rectangle(
            obstacle52_center[0] - 10, obstacle52_center[1] - 10,  # Top left corner
            obstacle52_center[0] + 10, obstacle52_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle52 = [self.canvas_widget.coords(self.obstacle52)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle52)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle52)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle52)[3] - 3]

        obstacle53_center = self.o + np.array([pixels * 13, pixels * 7])
        self.obstacle53 = self.canvas_widget.create_rectangle(
            obstacle53_center[0] - 10, obstacle53_center[1] - 10,  # Top left corner
            obstacle53_center[0] + 10, obstacle53_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle53 = [self.canvas_widget.coords(self.obstacle53)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle53)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle53)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle53)[3] - 3]

        obstacle54_center = self.o + np.array([pixels * 12, pixels * 7])
        self.obstacle54 = self.canvas_widget.create_rectangle(
            obstacle54_center[0] - 10, obstacle54_center[1] - 10,  # Top left corner
            obstacle54_center[0] + 10, obstacle54_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle54 = [self.canvas_widget.coords(self.obstacle54)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle54)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle54)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle54)[3] - 3]

        obstacle55_center = self.o + np.array([pixels * 12, pixels * 8])
        self.obstacle55 = self.canvas_widget.create_rectangle(
            obstacle55_center[0] - 10, obstacle55_center[1] - 10,  # Top left corner
            obstacle55_center[0] + 10, obstacle55_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle55 = [self.canvas_widget.coords(self.obstacle55)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle55)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle55)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle55)[3] - 3]

        obstacle56_center = self.o + np.array([pixels * 12, pixels * 9])
        self.obstacle56 = self.canvas_widget.create_rectangle(
            obstacle56_center[0] - 10, obstacle56_center[1] - 10,  # Top left corner
            obstacle56_center[0] + 10, obstacle56_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle56 = [self.canvas_widget.coords(self.obstacle56)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle56)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle56)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle56)[3] - 3]

        obstacle57_center = self.o + np.array([pixels * 12, pixels * 10])
        self.obstacle57 = self.canvas_widget.create_rectangle(
            obstacle57_center[0] - 10, obstacle57_center[1] - 10,  # Top left corner
            obstacle57_center[0] + 10, obstacle57_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle57 = [self.canvas_widget.coords(self.obstacle57)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle57)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle57)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle57)[3] - 3]

        obstacle58_center = self.o + np.array([pixels * 12, pixels * 11])
        self.obstacle58 = self.canvas_widget.create_rectangle(
            obstacle58_center[0] - 10, obstacle58_center[1] - 10,  # Top left corner
            obstacle58_center[0] + 10, obstacle58_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle58 = [self.canvas_widget.coords(self.obstacle58)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle58)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle58)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle58)[3] - 3]

        obstacle59_center = self.o + np.array([pixels * 12, pixels * 12])
        self.obstacle59 = self.canvas_widget.create_rectangle(
            obstacle59_center[0] - 10, obstacle59_center[1] - 10,  # Top left corner
            obstacle59_center[0] + 10, obstacle59_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle59 = [self.canvas_widget.coords(self.obstacle59)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle59)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle59)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle59)[3] - 3]

        obstacle60_center = self.o + np.array([pixels * 12, pixels * 13])
        self.obstacle60 = self.canvas_widget.create_rectangle(
            obstacle60_center[0] - 10, obstacle60_center[1] - 10,  # Top left corner
            obstacle60_center[0] + 10, obstacle60_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle60 = [self.canvas_widget.coords(self.obstacle60)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle60)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle60)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle60)[3] - 3]

        obstacle61_center = self.o + np.array([pixels * 12, pixels * 14])
        self.obstacle61 = self.canvas_widget.create_rectangle(
            obstacle61_center[0] - 10, obstacle61_center[1] - 10,  # Top left corner
            obstacle61_center[0] + 10, obstacle61_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle61 = [self.canvas_widget.coords(self.obstacle61)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle61)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle61)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle61)[3] - 3]

        obstacle62_center = self.o + np.array([pixels * 12, pixels * 15])
        self.obstacle62 = self.canvas_widget.create_rectangle(
            obstacle62_center[0] - 10, obstacle62_center[1] - 10,  # Top left corner
            obstacle62_center[0] + 10, obstacle62_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle62 = [self.canvas_widget.coords(self.obstacle62)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle62)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle62)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle62)[3] - 3]

        obstacle63_center = self.o + np.array([pixels * 12, pixels * 16])
        self.obstacle63 = self.canvas_widget.create_rectangle(
            obstacle63_center[0] - 10, obstacle63_center[1] - 10,  # Top left corner
            obstacle63_center[0] + 10, obstacle63_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle63 = [self.canvas_widget.coords(self.obstacle63)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle63)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle63)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle63)[3] - 3]

        obstacle64_center = self.o + np.array([pixels * 12, pixels * 17])
        self.obstacle64 = self.canvas_widget.create_rectangle(
            obstacle64_center[0] - 10, obstacle64_center[1] - 10,  # Top left corner
            obstacle64_center[0] + 10, obstacle64_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle64 = [self.canvas_widget.coords(self.obstacle64)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle64)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle64)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle64)[3] - 3]

        obstacle65_center = self.o + np.array([pixels * 12, pixels * 18])
        self.obstacle65 = self.canvas_widget.create_rectangle(
            obstacle65_center[0] - 10, obstacle65_center[1] - 10,  # Top left corner
            obstacle65_center[0] + 10, obstacle65_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle65 = [self.canvas_widget.coords(self.obstacle65)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle65)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle65)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle65)[3] - 3]

        obstacle66_center = self.o + np.array([pixels * 13, pixels * 18])
        self.obstacle66 = self.canvas_widget.create_rectangle(
            obstacle66_center[0] - 10, obstacle66_center[1] - 10,  # Top left corner
            obstacle66_center[0] + 10, obstacle66_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle66 = [self.canvas_widget.coords(self.obstacle66)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle66)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle66)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle66)[3] - 3]

        obstacle67_center = self.o + np.array([pixels * 14, pixels * 18])
        self.obstacle67 = self.canvas_widget.create_rectangle(
            obstacle67_center[0] - 10, obstacle67_center[1] - 10,  # Top left corner
            obstacle67_center[0] + 10, obstacle67_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle67 = [self.canvas_widget.coords(self.obstacle67)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle67)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle67)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle67)[3] - 3]

        obstacle68_center = self.o + np.array([pixels * 15, pixels * 18])
        self.obstacle68 = self.canvas_widget.create_rectangle(
            obstacle68_center[0] - 10, obstacle68_center[1] - 10,  # Top left corner
            obstacle68_center[0] + 10, obstacle68_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle68 = [self.canvas_widget.coords(self.obstacle68)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle68)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle68)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle68)[3] - 3]

        obstacle69_center = self.o + np.array([pixels * 15, pixels * 17])
        self.obstacle69 = self.canvas_widget.create_rectangle(
            obstacle69_center[0] - 10, obstacle69_center[1] - 10,  # Top left corner
            obstacle69_center[0] + 10, obstacle69_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle69 = [self.canvas_widget.coords(self.obstacle69)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle69)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle69)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle69)[3] - 3]

        obstacle70_center = self.o + np.array([pixels * 15, pixels * 16])
        self.obstacle70 = self.canvas_widget.create_rectangle(
            obstacle70_center[0] - 10, obstacle70_center[1] - 10,  # Top left corner
            obstacle70_center[0] + 10, obstacle70_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle70 = [self.canvas_widget.coords(self.obstacle70)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle70)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle70)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle70)[3] - 3]

        obstacle71_center = self.o + np.array([pixels * 15, pixels * 15])
        self.obstacle71 = self.canvas_widget.create_rectangle(
            obstacle71_center[0] - 10, obstacle71_center[1] - 10,  # Top left corner
            obstacle71_center[0] + 10, obstacle71_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle71 = [self.canvas_widget.coords(self.obstacle71)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle71)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle71)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle71)[3] - 3]

        obstacle72_center = self.o + np.array([pixels * 15, pixels * 14])
        self.obstacle72 = self.canvas_widget.create_rectangle(
            obstacle72_center[0] - 10, obstacle72_center[1] - 10,  # Top left corner
            obstacle72_center[0] + 10, obstacle72_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle72 = [self.canvas_widget.coords(self.obstacle72)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle72)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle72)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle72)[3] - 3]

        obstacle73_center = self.o + np.array([pixels * 15, pixels * 13])
        self.obstacle73 = self.canvas_widget.create_rectangle(
            obstacle73_center[0] - 10, obstacle73_center[1] - 10,  # Top left corner
            obstacle73_center[0] + 10, obstacle73_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle73 = [self.canvas_widget.coords(self.obstacle73)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle73)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle73)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle73)[3] - 3]

        obstacle74_center = self.o + np.array([pixels * 15, pixels * 12])
        self.obstacle74 = self.canvas_widget.create_rectangle(
            obstacle74_center[0] - 10, obstacle74_center[1] - 10,  # Top left corner
            obstacle74_center[0] + 10, obstacle74_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle74 = [self.canvas_widget.coords(self.obstacle74)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle74)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle74)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle74)[3] - 3]

        obstacle75_center = self.o + np.array([pixels * 15, pixels * 11])
        self.obstacle75 = self.canvas_widget.create_rectangle(
            obstacle75_center[0] - 10, obstacle75_center[1] - 10,  # Top left corner
            obstacle75_center[0] + 10, obstacle75_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle75 = [self.canvas_widget.coords(self.obstacle75)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle75)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle75)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle75)[3] - 3]

        obstacle76_center = self.o + np.array([pixels * 15, pixels * 10])
        self.obstacle76 = self.canvas_widget.create_rectangle(
            obstacle76_center[0] - 10, obstacle76_center[1] - 10,  # Top left corner
            obstacle76_center[0] + 10, obstacle76_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle76 = [self.canvas_widget.coords(self.obstacle76)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle76)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle76)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle76)[3] - 3]

        obstacle77_center = self.o + np.array([pixels * 15, pixels * 9])
        self.obstacle77 = self.canvas_widget.create_rectangle(
            obstacle77_center[0] - 10, obstacle77_center[1] - 10,  # Top left corner
            obstacle77_center[0] + 10, obstacle77_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle77 = [self.canvas_widget.coords(self.obstacle77)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle77)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle77)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle77)[3] - 3]

        obstacle78_center = self.o + np.array([pixels * 15, pixels * 8])
        self.obstacle78 = self.canvas_widget.create_rectangle(
            obstacle78_center[0] - 10, obstacle78_center[1] - 10,  # Top left corner
            obstacle78_center[0] + 10, obstacle78_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle78 = [self.canvas_widget.coords(self.obstacle78)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle78)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle78)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle78)[3] - 3]

        obstacle79_center = self.o + np.array([pixels * 1, pixels * 7])
        self.obstacle79 = self.canvas_widget.create_rectangle(
            obstacle79_center[0] - 10, obstacle79_center[1] - 10,  # Top left corner
            obstacle79_center[0] + 10, obstacle79_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle79 = [self.canvas_widget.coords(self.obstacle79)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle79)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle79)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle79)[3] - 3]

        obstacle80_center = self.o + np.array([pixels * 2, pixels * 7])
        self.obstacle80 = self.canvas_widget.create_rectangle(
            obstacle80_center[0] - 10, obstacle80_center[1] - 10,  # Top left corner
            obstacle80_center[0] + 10, obstacle80_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle80 = [self.canvas_widget.coords(self.obstacle80)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle80)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle80)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle80)[3] - 3]

        obstacle81_center = self.o + np.array([pixels * 3, pixels * 7])
        self.obstacle81 = self.canvas_widget.create_rectangle(
            obstacle81_center[0] - 10, obstacle81_center[1] - 10,  # Top left corner
            obstacle81_center[0] + 10, obstacle81_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle81 = [self.canvas_widget.coords(self.obstacle81)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle81)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle81)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle81)[3] - 3]

        obstacle82_center = self.o + np.array([pixels * 4, pixels * 7])
        self.obstacle82 = self.canvas_widget.create_rectangle(
            obstacle82_center[0] - 10, obstacle82_center[1] - 10,  # Top left corner
            obstacle82_center[0] + 10, obstacle82_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle82 = [self.canvas_widget.coords(self.obstacle82)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle82)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle82)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle82)[3] - 3]

        obstacle83_center = self.o + np.array([pixels * 4, pixels * 8])
        self.obstacle83 = self.canvas_widget.create_rectangle(
            obstacle83_center[0] - 10, obstacle83_center[1] - 10,  # Top left corner
            obstacle83_center[0] + 10, obstacle83_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle83 = [self.canvas_widget.coords(self.obstacle83)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle83)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle83)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle83)[3] - 3]

        obstacle84_center = self.o + np.array([pixels * 4, pixels * 9])
        self.obstacle84 = self.canvas_widget.create_rectangle(
            obstacle84_center[0] - 10, obstacle84_center[1] - 10,  # Top left corner
            obstacle84_center[0] + 10, obstacle84_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle84 = [self.canvas_widget.coords(self.obstacle84)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle84)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle84)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle84)[3] - 3]

        obstacle85_center = self.o + np.array([pixels * 4, pixels * 10])
        self.obstacle85 = self.canvas_widget.create_rectangle(
            obstacle85_center[0] - 10, obstacle85_center[1] - 10,  # Top left corner
            obstacle85_center[0] + 10, obstacle85_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle85 = [self.canvas_widget.coords(self.obstacle85)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle85)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle85)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle85)[3] - 3]

        obstacle86_center = self.o + np.array([pixels * 4, pixels * 11])
        self.obstacle86 = self.canvas_widget.create_rectangle(
            obstacle86_center[0] - 10, obstacle86_center[1] - 10,  # Top left corner
            obstacle86_center[0] + 10, obstacle86_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle86 = [self.canvas_widget.coords(self.obstacle86)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle86)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle86)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle86)[3] - 3]

        obstacle87_center = self.o + np.array([pixels * 4, pixels * 12])
        self.obstacle87 = self.canvas_widget.create_rectangle(
            obstacle87_center[0] - 10, obstacle87_center[1] - 10,  # Top left corner
            obstacle87_center[0] + 10, obstacle87_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle87 = [self.canvas_widget.coords(self.obstacle87)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle87)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle87)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle87)[3] - 3]

        obstacle88_center = self.o + np.array([pixels * 4, pixels * 13])
        self.obstacle88 = self.canvas_widget.create_rectangle(
            obstacle88_center[0] - 10, obstacle88_center[1] - 10,  # Top left corner
            obstacle88_center[0] + 10, obstacle88_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle88 = [self.canvas_widget.coords(self.obstacle88)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle88)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle88)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle88)[3] - 3]

        obstacle89_center = self.o + np.array([pixels * 4, pixels * 14])
        self.obstacle89 = self.canvas_widget.create_rectangle(
            obstacle89_center[0] - 10, obstacle89_center[1] - 10,  # Top left corner
            obstacle89_center[0] + 10, obstacle89_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle89 = [self.canvas_widget.coords(self.obstacle89)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle89)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle89)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle89)[3] - 3]

        obstacle90_center = self.o + np.array([pixels * 4, pixels * 15])
        self.obstacle90 = self.canvas_widget.create_rectangle(
            obstacle90_center[0] - 10, obstacle90_center[1] - 10,  # Top left corner
            obstacle90_center[0] + 10, obstacle90_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle90 = [self.canvas_widget.coords(self.obstacle90)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle90)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle90)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle90)[3] - 3]

        obstacle91_center = self.o + np.array([pixels * 4, pixels * 16])
        self.obstacle91 = self.canvas_widget.create_rectangle(
            obstacle91_center[0] - 10, obstacle91_center[1] - 10,  # Top left corner
            obstacle91_center[0] + 10, obstacle91_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle91 = [self.canvas_widget.coords(self.obstacle91)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle91)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle91)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle91)[3] - 3]

        obstacle92_center = self.o + np.array([pixels * 4, pixels * 17])
        self.obstacle92 = self.canvas_widget.create_rectangle(
            obstacle92_center[0] - 10, obstacle92_center[1] - 10,  # Top left corner
            obstacle92_center[0] + 10, obstacle92_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle92 = [self.canvas_widget.coords(self.obstacle92)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle92)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle92)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle92)[3] - 3]

        obstacle93_center = self.o + np.array([pixels * 4, pixels * 18])
        self.obstacle93 = self.canvas_widget.create_rectangle(
            obstacle93_center[0] - 10, obstacle93_center[1] - 10,  # Top left corner
            obstacle93_center[0] + 10, obstacle93_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle93 = [self.canvas_widget.coords(self.obstacle93)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle93)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle93)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle93)[3] - 3]

        obstacle94_center = self.o + np.array([pixels * 3, pixels * 18])
        self.obstacle94 = self.canvas_widget.create_rectangle(
            obstacle94_center[0] - 10, obstacle94_center[1] - 10,  # Top left corner
            obstacle94_center[0] + 10, obstacle94_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle94 = [self.canvas_widget.coords(self.obstacle94)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle94)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle94)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle94)[3] - 3]

        obstacle95_center = self.o + np.array([pixels * 2, pixels * 18])
        self.obstacle95 = self.canvas_widget.create_rectangle(
            obstacle95_center[0] - 10, obstacle95_center[1] - 10,  # Top left corner
            obstacle95_center[0] + 10, obstacle95_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle95 = [self.canvas_widget.coords(self.obstacle95)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle95)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle95)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle95)[3] - 3]

        obstacle96_center = self.o + np.array([pixels * 1, pixels * 18])
        self.obstacle96 = self.canvas_widget.create_rectangle(
            obstacle96_center[0] - 10, obstacle96_center[1] - 10,  # Top left corner
            obstacle96_center[0] + 10, obstacle96_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle96 = [self.canvas_widget.coords(self.obstacle96)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle96)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle96)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle96)[3] - 3]

        obstacle97_center = self.o + np.array([pixels * 1, pixels * 17])
        self.obstacle97 = self.canvas_widget.create_rectangle(
            obstacle97_center[0] - 10, obstacle97_center[1] - 10,  # Top left corner
            obstacle97_center[0] + 10, obstacle97_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle97 = [self.canvas_widget.coords(self.obstacle97)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle97)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle97)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle97)[3] - 3]

        obstacle98_center = self.o + np.array([pixels * 1, pixels * 16])
        self.obstacle98 = self.canvas_widget.create_rectangle(
            obstacle98_center[0] - 10, obstacle98_center[1] - 10,  # Top left corner
            obstacle98_center[0] + 10, obstacle98_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle98 = [self.canvas_widget.coords(self.obstacle98)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle98)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle98)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle98)[3] - 3]

        obstacle99_center = self.o + np.array([pixels * 1, pixels * 15])
        self.obstacle99 = self.canvas_widget.create_rectangle(
            obstacle99_center[0] - 10, obstacle99_center[1] - 10,  # Top left corner
            obstacle99_center[0] + 10, obstacle99_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle99 = [self.canvas_widget.coords(self.obstacle99)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle99)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle99)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle99)[3] - 3]

        obstacle100_center = self.o + np.array([pixels * 1, pixels * 14])
        self.obstacle100 = self.canvas_widget.create_rectangle(
            obstacle100_center[0] - 10, obstacle100_center[1] - 10,  # Top left corner
            obstacle100_center[0] + 10, obstacle100_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle100 = [self.canvas_widget.coords(self.obstacle100)[0] + 3,
                                  self.canvas_widget.coords(self.obstacle100)[1] + 3,
                                  self.canvas_widget.coords(self.obstacle100)[2] - 3,
                                  self.canvas_widget.coords(self.obstacle100)[3] - 3]

        obstacle101_center = self.o + np.array([pixels * 1, pixels * 13])
        self.obstacle101 = self.canvas_widget.create_rectangle(
            obstacle101_center[0] - 10, obstacle101_center[1] - 10,  # Top left corner
            obstacle101_center[0] + 10, obstacle101_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle101 = [self.canvas_widget.coords(self.obstacle101)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle101)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle101)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle101)[3] - 3]

        obstacle102_center = self.o + np.array([pixels * 1, pixels * 12])
        self.obstacle102 = self.canvas_widget.create_rectangle(
            obstacle102_center[0] - 10, obstacle102_center[1] - 10,  # Top left corner
            obstacle102_center[0] + 10, obstacle102_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle102 = [self.canvas_widget.coords(self.obstacle102)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle102)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle102)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle102)[3] - 3]

        obstacle103_center = self.o + np.array([pixels * 1, pixels * 11])
        self.obstacle103 = self.canvas_widget.create_rectangle(
            obstacle103_center[0] - 10, obstacle103_center[1] - 10,  # Top left corner
            obstacle103_center[0] + 10, obstacle103_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle103 = [self.canvas_widget.coords(self.obstacle103)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle103)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle103)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle103)[3] - 3]

        obstacle104_center = self.o + np.array([pixels * 1, pixels * 10])
        self.obstacle104 = self.canvas_widget.create_rectangle(
            obstacle104_center[0] - 10, obstacle104_center[1] - 10,  # Top left corner
            obstacle104_center[0] + 10, obstacle104_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle104 = [self.canvas_widget.coords(self.obstacle104)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle104)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle104)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle104)[3] - 3]

        obstacle105_center = self.o + np.array([pixels * 1, pixels * 9])
        self.obstacle105 = self.canvas_widget.create_rectangle(
            obstacle105_center[0] - 10, obstacle105_center[1] - 10,  # Top left corner
            obstacle105_center[0] + 10, obstacle105_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle105 = [self.canvas_widget.coords(self.obstacle105)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle105)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle105)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle105)[3] - 3]

        obstacle106_center = self.o + np.array([pixels * 1, pixels * 8])
        self.obstacle106 = self.canvas_widget.create_rectangle(
            obstacle106_center[0] - 10, obstacle106_center[1] - 10,  # Top left corner
            obstacle106_center[0] + 10, obstacle106_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle106 = [self.canvas_widget.coords(self.obstacle106)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle106)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle106)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle106)[3] - 3]

        obstacle107_center = self.o + np.array([pixels * 8, pixels * 7])
        self.obstacle107 = self.canvas_widget.create_rectangle(
            obstacle107_center[0] - 10, obstacle107_center[1] - 10,  # Top left corner
            obstacle107_center[0] + 10, obstacle107_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle107 = [self.canvas_widget.coords(self.obstacle107)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle107)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle107)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle107)[3] - 3]

        obstacle108_center = self.o + np.array([pixels * 8, pixels * 8])
        self.obstacle108 = self.canvas_widget.create_rectangle(
            obstacle108_center[0] - 10, obstacle108_center[1] - 10,  # Top left corner
            obstacle108_center[0] + 10, obstacle108_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle108 = [self.canvas_widget.coords(self.obstacle108)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle108)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle108)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle108)[3] - 3]

        obstacle109_center = self.o + np.array([pixels * 8, pixels * 9])
        self.obstacle109 = self.canvas_widget.create_rectangle(
            obstacle109_center[0] - 10, obstacle109_center[1] - 10,  # Top left corner
            obstacle109_center[0] + 10, obstacle109_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle109 = [self.canvas_widget.coords(self.obstacle109)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle109)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle109)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle109)[3] - 3]

        obstacle110_center = self.o + np.array([pixels * 8, pixels * 10])
        self.obstacle110 = self.canvas_widget.create_rectangle(
            obstacle110_center[0] - 10, obstacle110_center[1] - 10,  # Top left corner
            obstacle110_center[0] + 10, obstacle110_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle110 = [self.canvas_widget.coords(self.obstacle110)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle110)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle110)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle110)[3] - 3]

        obstacle111_center = self.o + np.array([pixels * 8, pixels * 11])
        self.obstacle111 = self.canvas_widget.create_rectangle(
            obstacle111_center[0] - 10, obstacle111_center[1] - 10,  # Top left corner
            obstacle111_center[0] + 10, obstacle111_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle111 = [self.canvas_widget.coords(self.obstacle111)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle111)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle111)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle111)[3] - 3]

        obstacle112_center = self.o + np.array([pixels * 8, pixels * 12])
        self.obstacle112 = self.canvas_widget.create_rectangle(
            obstacle112_center[0] - 10, obstacle112_center[1] - 10,  # Top left corner
            obstacle112_center[0] + 10, obstacle112_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle112 = [self.canvas_widget.coords(self.obstacle112)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle112)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle112)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle112)[3] - 3]

        obstacle113_center = self.o + np.array([pixels * 8, pixels * 13])
        self.obstacle113 = self.canvas_widget.create_rectangle(
            obstacle113_center[0] - 10, obstacle113_center[1] - 10,  # Top left corner
            obstacle113_center[0] + 10, obstacle113_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle113 = [self.canvas_widget.coords(self.obstacle113)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle113)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle113)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle113)[3] - 3]

        obstacle114_center = self.o + np.array([pixels * 8, pixels * 14])
        self.obstacle114 = self.canvas_widget.create_rectangle(
            obstacle114_center[0] - 10, obstacle114_center[1] - 10,  # Top left corner
            obstacle114_center[0] + 10, obstacle114_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle114 = [self.canvas_widget.coords(self.obstacle114)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle114)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle114)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle114)[3] - 3]

        obstacle115_center = self.o + np.array([pixels * 8, pixels * 15])
        self.obstacle115 = self.canvas_widget.create_rectangle(
            obstacle115_center[0] - 10, obstacle115_center[1] - 10,  # Top left corner
            obstacle115_center[0] + 10, obstacle115_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle115 = [self.canvas_widget.coords(self.obstacle115)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle115)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle115)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle115)[3] - 3]

        obstacle116_center = self.o + np.array([pixels * 8, pixels * 16])
        self.obstacle116 = self.canvas_widget.create_rectangle(
            obstacle116_center[0] - 10, obstacle116_center[1] - 10,  # Top left corner
            obstacle116_center[0] + 10, obstacle116_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle116 = [self.canvas_widget.coords(self.obstacle116)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle116)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle116)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle116)[3] - 3]

        obstacle117_center = self.o + np.array([pixels * 8, pixels * 17])
        self.obstacle117 = self.canvas_widget.create_rectangle(
            obstacle117_center[0] - 10, obstacle117_center[1] - 10,  # Top left corner
            obstacle117_center[0] + 10, obstacle117_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle117 = [self.canvas_widget.coords(self.obstacle117)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle117)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle117)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle117)[3] - 3]

        obstacle118_center = self.o + np.array([pixels * 8, pixels * 18])
        self.obstacle118 = self.canvas_widget.create_rectangle(
            obstacle118_center[0] - 10, obstacle118_center[1] - 10,  # Top left corner
            obstacle118_center[0] + 10, obstacle118_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle118 = [self.canvas_widget.coords(self.obstacle118)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle118)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle118)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle118)[3] - 3]

        obstacle119_center = self.o + np.array([pixels * 7, pixels * 10])
        self.obstacle119 = self.canvas_widget.create_rectangle(
            obstacle119_center[0] - 10, obstacle119_center[1] - 10,  # Top left corner
            obstacle119_center[0] + 10, obstacle119_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle119 = [self.canvas_widget.coords(self.obstacle119)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle119)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle119)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle119)[3] - 3]

        obstacle120_center = self.o + np.array([pixels * 9, pixels * 12])
        self.obstacle120 = self.canvas_widget.create_rectangle(
            obstacle120_center[0] - 10, obstacle120_center[1] - 10,  # Top left corner
            obstacle120_center[0] + 10, obstacle120_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle120 = [self.canvas_widget.coords(self.obstacle120)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle120)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle120)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle120)[3] - 3]

        obstacle121_center = self.o + np.array([pixels * 7, pixels * 14])
        self.obstacle121 = self.canvas_widget.create_rectangle(
            obstacle121_center[0] - 10, obstacle121_center[1] - 10,  # Top left corner
            obstacle121_center[0] + 10, obstacle121_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle121 = [self.canvas_widget.coords(self.obstacle121)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle121)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle121)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle121)[3] - 3]

        obstacle122_center = self.o + np.array([pixels * 9, pixels * 16])
        self.obstacle122 = self.canvas_widget.create_rectangle(
            obstacle122_center[0] - 10, obstacle122_center[1] - 10,  # Top left corner
            obstacle122_center[0] + 10, obstacle122_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle122 = [self.canvas_widget.coords(self.obstacle122)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle122)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle122)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle122)[3] - 3]

        obstacle123_center = self.o + np.array([pixels * 7, pixels * 18])
        self.obstacle123 = self.canvas_widget.create_rectangle(
            obstacle123_center[0] - 10, obstacle123_center[1] - 10,  # Top left corner
            obstacle123_center[0] + 10, obstacle123_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_obstacle123 = [self.canvas_widget.coords(self.obstacle123)[0] + 3,
                                   self.canvas_widget.coords(self.obstacle123)[1] + 3,
                                   self.canvas_widget.coords(self.obstacle123)[2] - 3,
                                   self.canvas_widget.coords(self.obstacle123)[3] - 3]

        # belt1_center = self.o + np.array([pixels * 4, pixels * 5])
        # self.belt1 = self.canvas_widget.create_rectangle(
        #     belt1_center[0] - 10, belt1_center[1] - 10,  # Top left corner
        #     belt1_center[0] + 10, belt1_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt1 = [self.canvas_widget.coords(self.belt1)[0] + 3,
        #                      self.canvas_widget.coords(self.belt1)[1] + 3,
        #                      self.canvas_widget.coords(self.belt1)[2] - 3,
        #                      self.canvas_widget.coords(self.belt1)[3] - 3]
        #
        # belt2_center = self.o + np.array([pixels * 4, pixels * 4])
        # self.belt2 = self.canvas_widget.create_rectangle(
        #     belt2_center[0] - 10, belt2_center[1] - 10,  # Top left corner
        #     belt2_center[0] + 10, belt2_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt2 = [self.canvas_widget.coords(self.belt2)[0] + 3,
        #                      self.canvas_widget.coords(self.belt2)[1] + 3,
        #                      self.canvas_widget.coords(self.belt2)[2] - 3,
        #                      self.canvas_widget.coords(self.belt2)[3] - 3]
        #
        # belt3_center = self.o + np.array([pixels * 5, pixels * 4])
        # self.belt3 = self.canvas_widget.create_rectangle(
        #     belt3_center[0] - 10, belt3_center[1] - 10,  # Top left corner
        #     belt3_center[0] + 10, belt3_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt3 = [self.canvas_widget.coords(self.belt3)[0] + 3,
        #                      self.canvas_widget.coords(self.belt3)[1] + 3,
        #                      self.canvas_widget.coords(self.belt3)[2] - 3,
        #                      self.canvas_widget.coords(self.belt3)[3] - 3]

        belt4_center = self.o + np.array([pixels * 6, pixels * 4])
        self.belt4 = self.canvas_widget.create_rectangle(
            belt4_center[0] - 10, belt4_center[1] - 10,  # Top left corner
            belt4_center[0] + 10, belt4_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt4 = [self.canvas_widget.coords(self.belt4)[0] + 3,
                             self.canvas_widget.coords(self.belt4)[1] + 3,
                             self.canvas_widget.coords(self.belt4)[2] - 3,
                             self.canvas_widget.coords(self.belt4)[3] - 3]

        belt5_center = self.o + np.array([pixels * 7, pixels * 3])
        self.belt5 = self.canvas_widget.create_rectangle(
            belt5_center[0] - 10, belt5_center[1] - 10,  # Top left corner
            belt5_center[0] + 10, belt5_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt5 = [self.canvas_widget.coords(self.belt5)[0] + 3,
                             self.canvas_widget.coords(self.belt5)[1] + 3,
                             self.canvas_widget.coords(self.belt5)[2] - 3,
                             self.canvas_widget.coords(self.belt5)[3] - 3]

        belt6_center = self.o + np.array([pixels * 8, pixels * 3])
        self.belt6 = self.canvas_widget.create_rectangle(
            belt6_center[0] - 10, belt6_center[1] - 10,  # Top left corner
            belt6_center[0] + 10, belt6_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt6 = [self.canvas_widget.coords(self.belt6)[0] + 3,
                             self.canvas_widget.coords(self.belt6)[1] + 3,
                             self.canvas_widget.coords(self.belt6)[2] - 3,
                             self.canvas_widget.coords(self.belt6)[3] - 3]

        belt7_center = self.o + np.array([pixels * 9, pixels * 3])
        self.belt7 = self.canvas_widget.create_rectangle(
            belt7_center[0] - 10, belt7_center[1] - 10,  # Top left corner
            belt7_center[0] + 10, belt7_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt7 = [self.canvas_widget.coords(self.belt7)[0] + 3,
                             self.canvas_widget.coords(self.belt7)[1] + 3,
                             self.canvas_widget.coords(self.belt7)[2] - 3,
                             self.canvas_widget.coords(self.belt7)[3] - 3]

        belt8_center = self.o + np.array([pixels * 10, pixels * 4])
        self.belt8 = self.canvas_widget.create_rectangle(
            belt8_center[0] - 10, belt8_center[1] - 10,  # Top left corner
            belt8_center[0] + 10, belt8_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt8 = [self.canvas_widget.coords(self.belt8)[0] + 3,
                             self.canvas_widget.coords(self.belt8)[1] + 3,
                             self.canvas_widget.coords(self.belt8)[2] - 3,
                             self.canvas_widget.coords(self.belt8)[3] - 3]

        # belt9_center = self.o + np.array([pixels * 11, pixels * 4])
        # self.belt9 = self.canvas_widget.create_rectangle(
        #     belt9_center[0] - 10, belt9_center[1] - 10,  # Top left corner
        #     belt9_center[0] + 10, belt9_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt9 = [self.canvas_widget.coords(self.belt9)[0] + 3,
        #                      self.canvas_widget.coords(self.belt9)[1] + 3,
        #                      self.canvas_widget.coords(self.belt9)[2] - 3,
        #                      self.canvas_widget.coords(self.belt9)[3] - 3]
        #
        # belt10_center = self.o + np.array([pixels * 12, pixels * 4])
        # self.belt10 = self.canvas_widget.create_rectangle(
        #     belt10_center[0] - 10, belt10_center[1] - 10,  # Top left corner
        #     belt10_center[0] + 10, belt10_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt10 = [self.canvas_widget.coords(self.belt10)[0] + 3,
        #                       self.canvas_widget.coords(self.belt10)[1] + 3,
        #                       self.canvas_widget.coords(self.belt10)[2] - 3,
        #                       self.canvas_widget.coords(self.belt10)[3] - 3]
        #
        # belt11_center = self.o + np.array([pixels * 12, pixels * 5])
        # self.belt11 = self.canvas_widget.create_rectangle(
        #     belt11_center[0] - 10, belt11_center[1] - 10,  # Top left corner
        #     belt11_center[0] + 10, belt11_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt11 = [self.canvas_widget.coords(self.belt11)[0] + 3,
        #                       self.canvas_widget.coords(self.belt11)[1] + 3,
        #                       self.canvas_widget.coords(self.belt11)[2] - 3,
        #                       self.canvas_widget.coords(self.belt11)[3] - 3]
        #
        # belt12_center = self.o + np.array([pixels * 11, pixels * 5])
        # self.belt12 = self.canvas_widget.create_rectangle(
        #     belt12_center[0] - 10, belt12_center[1] - 10,  # Top left corner
        #     belt12_center[0] + 10, belt12_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt12 = [self.canvas_widget.coords(self.belt12)[0] + 3,
        #                       self.canvas_widget.coords(self.belt12)[1] + 3,
        #                       self.canvas_widget.coords(self.belt12)[2] - 3,
        #                       self.canvas_widget.coords(self.belt12)[3] - 3]

        belt13_center = self.o + np.array([pixels * 10, pixels * 5])
        self.belt13 = self.canvas_widget.create_rectangle(
            belt13_center[0] - 10, belt13_center[1] - 10,  # Top left corner
            belt13_center[0] + 10, belt13_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt13 = [self.canvas_widget.coords(self.belt13)[0] + 3,
                              self.canvas_widget.coords(self.belt13)[1] + 3,
                              self.canvas_widget.coords(self.belt13)[2] - 3,
                              self.canvas_widget.coords(self.belt13)[3] - 3]

        belt14_center = self.o + np.array([pixels * 9, pixels * 5])
        self.belt14 = self.canvas_widget.create_rectangle(
            belt14_center[0] - 10, belt14_center[1] - 10,  # Top left corner
            belt14_center[0] + 10, belt14_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt14 = [self.canvas_widget.coords(self.belt14)[0] + 3,
                              self.canvas_widget.coords(self.belt14)[1] + 3,
                              self.canvas_widget.coords(self.belt14)[2] - 3,
                              self.canvas_widget.coords(self.belt14)[3] - 3]

        belt15_center = self.o + np.array([pixels * 8, pixels * 5])
        self.belt15 = self.canvas_widget.create_rectangle(
            belt15_center[0] - 10, belt15_center[1] - 10,  # Top left corner
            belt15_center[0] + 10, belt15_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt15 = [self.canvas_widget.coords(self.belt15)[0] + 3,
                              self.canvas_widget.coords(self.belt15)[1] + 3,
                              self.canvas_widget.coords(self.belt15)[2] - 3,
                              self.canvas_widget.coords(self.belt15)[3] - 3]

        belt16_center = self.o + np.array([pixels * 7, pixels * 5])
        self.belt16 = self.canvas_widget.create_rectangle(
            belt16_center[0] - 10, belt16_center[1] - 10,  # Top left corner
            belt16_center[0] + 10, belt16_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt16 = [self.canvas_widget.coords(self.belt16)[0] + 3,
                              self.canvas_widget.coords(self.belt16)[1] + 3,
                              self.canvas_widget.coords(self.belt16)[2] - 3,
                              self.canvas_widget.coords(self.belt16)[3] - 3]

        belt17_center = self.o + np.array([pixels * 6, pixels * 5])
        self.belt17 = self.canvas_widget.create_rectangle(
            belt17_center[0] - 10, belt17_center[1] - 10,  # Top left corner
            belt17_center[0] + 10, belt17_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt17 = [self.canvas_widget.coords(self.belt17)[0] + 3,
                              self.canvas_widget.coords(self.belt17)[1] + 3,
                              self.canvas_widget.coords(self.belt17)[2] - 3,
                              self.canvas_widget.coords(self.belt17)[3] - 3]

        # belt18_center = self.o + np.array([pixels * 5, pixels * 5])
        # self.belt18 = self.canvas_widget.create_rectangle(
        #     belt18_center[0] - 10, belt18_center[1] - 10,  # Top left corner
        #     belt18_center[0] + 10, belt18_center[1] + 10,  # Bottom right corner
        #     outline='white', fill='#000000')
        # self.coords_belt18 = [self.canvas_widget.coords(self.belt18)[0] + 3,
        #                       self.canvas_widget.coords(self.belt18)[1] + 3,
        #                       self.canvas_widget.coords(self.belt18)[2] - 3,
        #                       self.canvas_widget.coords(self.belt18)[3] - 3]

        belt19_center = self.o + np.array([pixels * 6, pixels * 3])
        self.belt19 = self.canvas_widget.create_rectangle(
            belt19_center[0] - 10, belt19_center[1] - 10,  # Top left corner
            belt19_center[0] + 10, belt19_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt19 = [self.canvas_widget.coords(self.belt19)[0] + 3,
                              self.canvas_widget.coords(self.belt19)[1] + 3,
                              self.canvas_widget.coords(self.belt19)[2] - 3,
                              self.canvas_widget.coords(self.belt19)[3] - 3]

        belt20_center = self.o + np.array([pixels * 10, pixels * 3])
        self.belt20 = self.canvas_widget.create_rectangle(
            belt20_center[0] - 10, belt20_center[1] - 10,  # Top left corner
            belt20_center[0] + 10, belt20_center[1] + 10,  # Bottom right corner
            outline='white', fill='#000000')
        self.coords_belt20 = [self.canvas_widget.coords(self.belt20)[0] + 3,
                              self.canvas_widget.coords(self.belt20)[1] + 3,
                              self.canvas_widget.coords(self.belt20)[2] - 3,
                              self.canvas_widget.coords(self.belt20)[3] - 3]

        # Creating an agent of Mobile Robot - red point
        self.flag_center4 = self.o + np.array([pixels * 5, pixels * 24])
        self.agent = self.canvas_widget.create_oval(
            self.flag_center4[0] - 10, self.flag_center4[1] - 10,
            self.flag_center4[0] + 10, self.flag_center4[1] + 10,
            outline='grey', fill='yellow')

        self.agent = self.canvas_widget.create_oval(
            self.o[0] - 7, self.o[1] - 7,
            self.o[0] + 7, self.o[1] + 7,
            outline='#FF1493', fill='#FF1493')
        self.agent1 = self.canvas_widget.create_oval(
            self.o[0] - 10, self.o[1] - 10,
            self.o[0] + 10, self.o[1] + 10,
            outline='white', fill='#000000')
        self.coords_agent1 = [self.canvas_widget.coords(self.agent1)[0] + 3,
                              self.canvas_widget.coords(self.agent1)[1] + 3,
                              self.canvas_widget.coords(self.agent1)[2] - 3,
                              self.canvas_widget.coords(self.agent1)[3] - 3]

        # # Middle Point - red point
        # flag_center01 = self.o + np.array([pixels * 1, pixels])
        # # Building the flag
        # self.flag01 = self.canvas_widget.create_rectangle(
        #     flag_center01[0] - 10, flag_center01[1] - 10,  # Top left corner
        #     flag_center01[0] + 10, flag_center01[1] + 10,  # Bottom right corner
        #     outline='white', fill='red')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag01 = [self.canvas_widget.coords(self.flag01)[0] + 3,
        #                      self.canvas_widget.coords(self.flag01)[1] + 3,
        #                      self.canvas_widget.coords(self.flag01)[2] - 3,
        #                      self.canvas_widget.coords(self.flag01)[3] - 3]
        #
        # flag_center02 = self.o + np.array([pixels * 5, pixels])
        # # Building the flag
        # self.flag02 = self.canvas_widget.create_rectangle(
        #     flag_center02[0] - 10, flag_center02[1] - 10,  # Top left corner
        #     flag_center02[0] + 10, flag_center02[1] + 10,  # Bottom right corner
        #     outline='white', fill='red')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag02 = [self.canvas_widget.coords(self.flag02)[0] + 3,
        #                       self.canvas_widget.coords(self.flag02)[1] + 3,
        #                       self.canvas_widget.coords(self.flag02)[2] - 3,
        #                       self.canvas_widget.coords(self.flag02)[3] - 3]
        #
        # flag_center03 = self.o + np.array([pixels * 9, pixels])
        # # Building the flag
        # self.flag03 = self.canvas_widget.create_rectangle(
        #     flag_center03[0] - 10, flag_center03[1] - 10,  # Top left corner
        #     flag_center03[0] + 10, flag_center03[1] + 10,  # Bottom right corner
        #     outline='white', fill='red')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag03 = [self.canvas_widget.coords(self.flag03)[0] + 3,
        #                       self.canvas_widget.coords(self.flag03)[1] + 3,
        #                       self.canvas_widget.coords(self.flag03)[2] - 3,
        #                       self.canvas_widget.coords(self.flag03)[3] - 3]

        flag_center04 = self.o + np.array([pixels * 13, pixels])
        # Building the flag
        self.flag04 = self.canvas_widget.create_oval(
            flag_center04[0] - 10, flag_center04[1] - 10,  # Top left corner
            flag_center04[0] + 10, flag_center04[1] + 10,  # Bottom right corner
            outline='white', fill='#696969')
        # Saving the coordinates of the final point according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_flag04 = [self.canvas_widget.coords(self.flag04)[0] + 3,
                              self.canvas_widget.coords(self.flag04)[1] + 3,
                              self.canvas_widget.coords(self.flag04)[2] - 3,
                              self.canvas_widget.coords(self.flag04)[3] - 3]

        # flag_center05 = self.o + np.array([pixels * 17, pixels])
        # # Building the flag
        # self.flag05 = self.canvas_widget.create_rectangle(
        #     flag_center05[0] - 10, flag_center05[1] - 10,  # Top left corner
        #     flag_center05[0] + 10, flag_center05[1] + 10,  # Bottom right corner
        #     outline='white', fill='red')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag05 = [self.canvas_widget.coords(self.flag05)[0] + 3,
        #                       self.canvas_widget.coords(self.flag05)[1] + 3,
        #                       self.canvas_widget.coords(self.flag05)[2] - 3,
        #                       self.canvas_widget.coords(self.flag05)[3] - 3]
        #
        # flag_center06 = self.o + np.array([pixels * 21, pixels])
        # # Building the flag
        # self.flag06 = self.canvas_widget.create_rectangle(
        #     flag_center06[0] - 10, flag_center06[1] - 10,  # Top left corner
        #     flag_center06[0] + 10, flag_center06[1] + 10,  # Bottom right corner
        #     outline='white', fill='red')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag06 = [self.canvas_widget.coords(self.flag06)[0] + 3,
        #                       self.canvas_widget.coords(self.flag06)[1] + 3,
        #                       self.canvas_widget.coords(self.flag06)[2] - 3,
        #                       self.canvas_widget.coords(self.flag06)[3] - 3]

        # # Final Point - yellow point
        # flag_center1 = self.o + np.array([pixels * 20, pixels * 24])
        # # Building the flag
        # self.flag1 = self.canvas_widget.create_rectangle(
        #     flag_center1[0] - 10, flag_center1[1] - 10,  # Top left corner
        #     flag_center1[0] + 10, flag_center1[1] + 10,  # Bottom right corner
        #     outline='grey', fill='yellow')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag1 = [self.canvas_widget.coords(self.flag1)[0] + 3,
        #                     self.canvas_widget.coords(self.flag1)[1] + 3,
        #                     self.canvas_widget.coords(self.flag1)[2] - 3,
        #                     self.canvas_widget.coords(self.flag1)[3] - 3]
        #
        # flag_center2 = self.o + np.array([pixels * 15, pixels * 24])
        # # Building the flag
        # self.flag2 = self.canvas_widget.create_rectangle(
        #     flag_center2[0] - 10, flag_center2[1] - 10,  # Top left corner
        #     flag_center2[0] + 10, flag_center2[1] + 10,  # Bottom right corner
        #     outline='grey', fill='yellow')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag2 = [self.canvas_widget.coords(self.flag2)[0] + 3,
        #                      self.canvas_widget.coords(self.flag2)[1] + 3,
        #                      self.canvas_widget.coords(self.flag2)[2] - 3,
        #                      self.canvas_widget.coords(self.flag2)[3] - 3]
        #
        # flag_center3 = self.o + np.array([pixels * 10, pixels * 24])
        # # Building the flag
        # self.flag3 = self.canvas_widget.create_rectangle(
        #     flag_center3[0] - 10, flag_center3[1] - 10,  # Top left corner
        #     flag_center3[0] + 10, flag_center3[1] + 10,  # Bottom right corner
        #     outline='grey', fill='yellow')
        # # Saving the coordinates of the final point according to the size of agent
        # # In order to fit the coordinates of the agent
        # self.coords_flag3 = [self.canvas_widget.coords(self.flag3)[0] + 3,
        #                      self.canvas_widget.coords(self.flag3)[1] + 3,
        #                      self.canvas_widget.coords(self.flag3)[2] - 3,
        #                      self.canvas_widget.coords(self.flag3)[3] - 3]

        flag_center4 = self.o + np.array([pixels * 5, pixels * 24])
        # Building the flag
        self.flag4 = self.canvas_widget.create_oval(
            flag_center4[0] - 10, flag_center4[1] - 10,  # Top left corner
            flag_center4[0] + 10, flag_center4[1] + 10,  # Bottom right corner
            outline='grey', fill='#D3D3D3')
        # Saving the coordinates of the final point according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_flag4 = [self.canvas_widget.coords(self.flag4)[0] + 3,
                             self.canvas_widget.coords(self.flag4)[1] + 3,
                             self.canvas_widget.coords(self.flag4)[2] - 3,
                             self.canvas_widget.coords(self.flag4)[3] - 3]

        # Packing everything
        self.canvas_widget.pack()

    # Function to reset the environment and start new Episode
    def reset(self):
        #self.update()
        #time.sleep(0.5)

        # Updating agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_oval(
            self.flag_center4[0] - 7, self.flag_center4[1] - 7,
            self.flag_center4[0] + 7, self.flag_center4[1] + 7,
            outline='red', fill='red')

        # Clearing the dictionary and the i
        self.d = {}
        self.i = 0

        # Return observation
        #return self.canvas_widget.coords(self.agent)
        return (np.array(self.canvas_widget.coords(self.agent)[:2]) - np.array(self.canvas_widget.coords(self.flag04)[:2])) / (env_height * pixels)

    # Function to get the next observation and reward by doing next step
    def step(self, action):
        # Current state of the agent
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0, 0])

        # Updating next state according to the action
        # Action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels
        # Action 'down'
        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
        # Action right
        elif action == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels
        # Action left
        elif action == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels
        elif action == 4:
            if state[1] >= pixels and state[0] < (env_width - 1) * pixels:
                base_action[1] -= pixels
                base_action[0] += pixels
        elif action == 5:
            if state[1] >= pixels and state[0] >= pixels:
                base_action[1] -= pixels
                base_action[0] -= pixels
        elif action == 6:
            if state[1] < (env_height - 1) * pixels and state[0] >= pixels:
                base_action[1] += pixels
                base_action[0] -= pixels
        elif action == 7:
            if state[1] < (env_height - 1) * pixels and state[0] < (env_width - 1) * pixels:
                base_action[1] += pixels
                base_action[0] += pixels

        # Moving the agent according to the action
        self.canvas_widget.move(self.agent, base_action[0], base_action[1])

        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.canvas_widget.coords(self.agent)

        # Updating next state
        next_coords = self.d[self.i]

        # Updating key for the dictionary
        self.i += 1

        # Calculating the reward for the agent
        if next_coords == self.coords_flag04:
            self.a = 0
            for k in range(1, len(self.d) - 1):
                if not (((self.d[k - 1][0] + 20 == self.d[k][0] == self.d[k + 1][0] - 20) and (
                        self.d[k - 1][1] + 20 == self.d[k][1] == self.d[k + 1][1] - 20)) or (
                                (self.d[k - 1][0] + 20 == self.d[k][0] == self.d[k + 1][0] - 20) and (
                                self.d[k - 1][1] - 20 == self.d[k][1] == self.d[k + 1][1] + 20))
                        or ((self.d[k - 1][0] - 20 == self.d[k][0] == self.d[k + 1][0] + 20) and (
                                self.d[k - 1][1] + 20 == self.d[k][1] == self.d[k + 1][1] - 20)) or (
                                self.d[k - 1][0] - 20 == self.d[k][0] == self.d[k + 1][0] + 20) and (
                                self.d[k - 1][1] - 20 == self.d[k][1] == self.d[k + 1][1] + 20)
                        or ((self.d[k - 1][0] == self.d[k][0] == self.d[k + 1][0]) and (
                                self.d[k - 1][1] + 20 == self.d[k][1] == self.d[k + 1][1] - 20)) or (
                                (self.d[k - 1][0] == self.d[k][0] == self.d[k + 1][0]) and (
                                self.d[k - 1][1] - 20 == self.d[k][1] == self.d[k + 1][1] + 20)) or (
                                (self.d[k - 1][1] == self.d[k][1] == self.d[k + 1][1]) and (
                                self.d[k - 1][0] - 20 == self.d[k][0] == self.d[k + 1][0] + 20)) or (
                                (self.d[k - 1][1] == self.d[k][1] == self.d[k + 1][1]) and (
                                self.d[k - 1][0] + 20 == self.d[k][0] == self.d[k + 1][0] - 20))):
                    self.a = self.a + 1
            if not (((self.coords_flag4[0] + 20 == self.d[0][0] == self.d[1][0] - 20) and (
                        self.coords_flag4[1] + 20 == self.d[0][1] == self.d[1][1] - 20)) or (
                                (self.coords_flag4[0] + 20 == self.d[0][0] == self.d[1][0] - 20) and (
                                self.coords_flag4[1] - 20 == self.d[0][1] == self.d[1][1] + 20))
                        or ((self.coords_flag4[0] - 20 == self.d[0][0] == self.d[1][0] + 20) and (
                                self.coords_flag4[1] + 20 == self.d[0][1] == self.d[1][1] - 20)) or (
                                self.coords_flag4[0] - 20 == self.d[0][0] == self.d[1][0] + 20) and (
                                self.coords_flag4[1] - 20 == self.d[0][1] == self.d[1][1] + 20)
                        or ((self.coords_flag4[0] == self.d[0][0] == self.d[1][0]) and (
                                self.coords_flag4[1] + 20 == self.d[0][1] == self.d[1][1] - 20)) or (
                                (self.coords_flag4[0] == self.d[0][0] == self.d[1][0]) and (
                                self.coords_flag4[1] - 20 == self.d[0][1] == self.d[1][1] + 20)) or (
                                (self.coords_flag4[1] == self.d[0][1] == self.d[1][1]) and (
                                self.coords_flag4[0] - 20 == self.d[0][0] == self.d[1][0] + 20)) or (
                                (self.coords_flag4[1] == self.d[0][1] == self.d[1][1]) and (
                                self.coords_flag4[0] + 20 == self.d[0][0] == self.d[1][0] - 20))):
                self.a = self.a + 1

            self.distance = 0
            for x in range(1, len(self.d)):
                self.distance = self.distance + (
                        (self.d[x][0] - self.d[x - 1][0]) ** 2 + (self.d[x][1] - self.d[x - 1][1]) ** 2) ** (
                                        1 / 2)
            self.distance = self.distance + (
                    (self.d[0][0] - self.coords_flag4[0]) ** 2 + (self.d[0][1] - self.coords_flag4[1]) ** 2) ** (
                                    1 / 2)
            #print('self.distance',self.distance)
            #print('self.s ',self.s)

            # Filling the dictionary first time
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                    # for y in range(1, len(self.d)):
                    #     self.middle = self.middle + (
                    #             (self.d[y][0] - self.d[y - 1][0]) ** 2 + (self.d[y][1] - self.d[y - 1][1]) ** 2) ** (
                    #                           1 / 2)
                    # self.middle = self.middle + (
                    #         (self.d[0][0] - self.coords_flag4[0]) ** 2 + (
                    #         self.d[0][1] - self.coords_flag4[1]) ** 2) ** (
                    #                       1 / 2)
                # for k in range(1, len(self.d) - 1):
                #     if not (((self.d[k - 1][0] + 20 == self.d[k][0] == self.d[k + 1][0] - 20) and (
                #             self.d[k - 1][1] + 20 == self.d[k][1] == self.d[k + 1][1] - 20)) or (
                #                     (self.d[k - 1][0] + 20 == self.d[k][0] == self.d[k + 1][0] - 20) and (
                #                     self.d[k - 1][1] - 20 == self.d[k][1] == self.d[k + 1][1] + 20))
                #             or ((self.d[k - 1][0] - 20 == self.d[k][0] == self.d[k + 1][0] + 20) and (
                #                     self.d[k - 1][1] + 20 == self.d[k][1] == self.d[k + 1][1] - 20)) or (
                #                     self.d[k - 1][0] - 20 == self.d[k][0] == self.d[k + 1][0] + 20) and (
                #                     self.d[k - 1][1] - 20 == self.d[k][1] == self.d[k + 1][1] + 20)
                #             or ((self.d[k - 1][0] == self.d[k][0] == self.d[k + 1][0]) and (
                #                     self.d[k - 1][1] + 20 == self.d[k][1] == self.d[k + 1][1] - 20)) or (
                #                     (self.d[k - 1][0] == self.d[k][0] == self.d[k + 1][0]) and (
                #                     self.d[k - 1][1] - 20 == self.d[k][1] == self.d[k + 1][1] + 20)) or (
                #                     (self.d[k - 1][1] == self.d[k][1] == self.d[k + 1][1]) and (
                #                     self.d[k - 1][0] - 20 == self.d[k][0] == self.d[k + 1][0] + 20)) or (
                #                     (self.d[k - 1][1] == self.d[k][1] == self.d[k + 1][1]) and (
                #                     self.d[k - 1][0] + 20 == self.d[k][0] == self.d[k + 1][0] - 20))):
                #         self.a0 = self.a0 + 1
                # if not (((self.coords_flag4[0] + 20 == self.d[0][0] == self.d[1][0] - 20) and (
                #         self.coords_flag4[1] + 20 == self.d[0][1] == self.d[1][1] - 20)) or (
                #                 (self.coords_flag4[0] + 20 == self.d[0][0] == self.d[1][0] - 20) and (
                #                 self.coords_flag4[1] - 20 == self.d[0][1] == self.d[1][1] + 20))
                #         or ((self.coords_flag4[0] - 20 == self.d[0][0] == self.d[1][0] + 20) and (
                #                 self.coords_flag4[1] + 20 == self.d[0][1] == self.d[1][1] - 20)) or (
                #                 self.coords_flag4[0] - 20 == self.d[0][0] == self.d[1][0] + 20) and (
                #                 self.coords_flag4[1] - 20 == self.d[0][1] == self.d[1][1] + 20)
                #         or ((self.coords_flag4[0] == self.d[0][0] == self.d[1][0]) and (
                #                 self.coords_flag4[1] + 20 == self.d[0][1] == self.d[1][1] - 20)) or (
                #                 (self.coords_flag4[0] == self.d[0][0] == self.d[1][0]) and (
                #                 self.coords_flag4[1] - 20 == self.d[0][1] == self.d[1][1] + 20)) or (
                #                 (self.coords_flag4[1] == self.d[0][1] == self.d[1][1]) and (
                #                 self.coords_flag4[0] - 20 == self.d[0][0] == self.d[1][0] + 20)) or (
                #                 (self.coords_flag4[1] == self.d[0][1] == self.d[1][1]) and (
                #                 self.coords_flag4[0] + 20 == self.d[0][0] == self.d[1][0] - 20))):
                #     self.a0 = self.a0 + 1
                #     for x in range(1, len(self.d)):
                #         self.distance = self.distance + (
                #                 (self.d[x][0] - self.d[x - 1][0]) ** 2 + (self.d[x][1] - self.d[x - 1][1]) ** 2) ** (
                #                                 1 / 2)
                # self.s = self.distance
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)
                self.s = self.middle
                self.lo = 0
                self.ss = self.a0
                self.ll = 0

            # Checking if the currently found route is shorter
            # if (len(self.d) < len(self.f)) or (self.distance < self.s):
            # if (self.distance <= self.s) and (self.a <= self.ss):
            # if (self.distance < self.s) or (self.a < self.ss):
            #     # Saving the number of steps for the shortest route
            #     self.shortest = len(self.d)
            #     self.s = self.distance
            #     self.ss = self.a
            #
            #     # Clearing the dictionary for the final route
            #     self.f = {}
            #     # Reassigning the dictionary
            #     for j in range(len(self.d)):
            #         self.f[j] = self.d[j]
            if (self.distance < self.s):
                self.shortest = len(self.d)
                self.s = self.distance
                self.ss = self.a
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
            if (self.distance == self.s):
                if (self.a < self.ss):
                    # Saving the number of steps for the shortest route
                    self.shortest = len(self.d)
                    self.s = self.distance
                    self.ss = self.a
                    # Clearing the dictionary for the final route
                    self.f = {}
                    # Reassigning the dictionary
                    for j in range(len(self.d)):
                        self.f[j] = self.d[j]
                # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)
            if self.distance > self.lo:
                self.lo = self.distance
            if (self.ll < self.a):
                self.ll = self.a


            # # Filling the dictionary first time
            # if self.c == True:
            #     for j in range(len(self.d)):
            #         self.f[j] = self.d[j]
            #     self.c = False
            #     self.longest = len(self.d)
            #     self.shortest = len(self.d)
            #
            # # Checking if the currently found route is shorter
            # if len(self.d) < len(self.f):
            #     # Saving the number of steps for the shortest route
            #     self.shortest = len(self.d)
            #     # Clearing the dictionary for the final route
            #     self.f = {}
            #     # Reassigning the dictionary
            #     for j in range(len(self.d)):
            #         self.f[j] = self.d[j]
            # # Saving the number of steps for the longest route
            # if len(self.d) > self.longest:
            #     self.longest = len(self.d)


            # time.sleep(0.1)
            # reward = 1 - 0.9 * (len(self.d) / self.longest)
            # reward = 1 - 0.9 * (self.distance / self.lo) + 1 - 0.2 * self.a
            reward = 1 - 0.9 * (self.distance / self.lo) + 1 - 0.001 * self.a
            done = True
            next_state = 'goal'

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)
            #print('self.d', self.d)
            #print('self.f', self.f)

        elif next_coords in [self.coords_obstacle1,
                                 self.coords_obstacle2,
                                 self.coords_obstacle3,
                                 self.coords_obstacle4,
                                 self.coords_obstacle5,
                                 self.coords_obstacle6,
                                 self.coords_obstacle7,
                                 self.coords_obstacle8,
                                 self.coords_obstacle9,
                                 self.coords_obstacle10,
                                 self.coords_obstacle11,
                                 self.coords_obstacle12,
                                 self.coords_obstacle13,
                                 self.coords_obstacle14,
                                 self.coords_obstacle15,
                                 self.coords_obstacle16,
                                 # self.coords_obstacle17,
                                 # self.coords_obstacle18,
                                 # self.coords_obstacle19,
                                 # self.coords_obstacle20,
                                 # self.coords_obstacle21,
                                 # self.coords_obstacle22,
                                 # self.coords_obstacle23,
                                 # self.coords_obstacle24,
                                 # self.coords_obstacle25,
                                 # self.coords_obstacle26,
                                 # self.coords_obstacle27,
                                 # self.coords_obstacle28,
                                 # self.coords_obstacle29,
                                 # self.coords_obstacle30,
                                 # self.coords_obstacle31,
                                 # self.coords_obstacle32,
                                 # self.coords_obstacle33,
                                 # self.coords_obstacle34,
                                 # self.coords_obstacle35,
                                 # self.coords_obstacle36,
                                 # self.coords_obstacle37,
                                 # self.coords_obstacle38,
                                 # self.coords_obstacle39,
                                 # self.coords_obstacle40,
                                 # self.coords_obstacle41,
                                 # self.coords_obstacle42,
                                 # self.coords_obstacle43,
                                 # self.coords_obstacle44,
                                 # self.coords_obstacle45,
                                 # self.coords_obstacle46,
                                 # self.coords_obstacle47,
                                 # self.coords_obstacle48,
                                 # self.coords_obstacle49,
                                 # self.coords_obstacle50,
                                 self.coords_obstacle51,
                                 self.coords_obstacle52,
                                 self.coords_obstacle53,
                                 self.coords_obstacle54,
                                 self.coords_obstacle55,
                                 self.coords_obstacle56,
                                 self.coords_obstacle57,
                                 self.coords_obstacle58,
                                 self.coords_obstacle59,
                                 self.coords_obstacle60,
                                 self.coords_obstacle61,
                                 self.coords_obstacle62,
                                 self.coords_obstacle63,
                                 self.coords_obstacle64,
                                 self.coords_obstacle65,
                                 self.coords_obstacle66,
                                 self.coords_obstacle67,
                                 self.coords_obstacle68,
                                 self.coords_obstacle69,
                                 self.coords_obstacle70,
                                 self.coords_obstacle71,
                                 self.coords_obstacle72,
                                 self.coords_obstacle73,
                                 self.coords_obstacle74,
                                 self.coords_obstacle75,
                                 self.coords_obstacle76,
                                 self.coords_obstacle77,
                                 self.coords_obstacle78,
                                 self.coords_obstacle79,
                                 self.coords_obstacle80,
                                 self.coords_obstacle81,
                                 self.coords_obstacle82,
                                 self.coords_obstacle83,
                                 self.coords_obstacle84,
                                 self.coords_obstacle85,
                                 self.coords_obstacle86,
                                 self.coords_obstacle87,
                                 self.coords_obstacle88,
                                 self.coords_obstacle89,
                                 self.coords_obstacle90,
                                 self.coords_obstacle91,
                                 self.coords_obstacle92,
                                 self.coords_obstacle93,
                                 self.coords_obstacle94,
                                 self.coords_obstacle95,
                                 self.coords_obstacle96,
                                 self.coords_obstacle97,
                                 self.coords_obstacle98,
                                 self.coords_obstacle99,
                                 self.coords_obstacle100,
                                 self.coords_obstacle101,
                                 self.coords_obstacle102,
                                 self.coords_obstacle103,
                                 self.coords_obstacle104,
                                 self.coords_obstacle105,
                                 self.coords_obstacle106,
                                 self.coords_obstacle107,
                                 self.coords_obstacle108,
                                 self.coords_obstacle109,
                                 self.coords_obstacle110,
                                 self.coords_obstacle111,
                                 self.coords_obstacle112,
                                 self.coords_obstacle113,
                                 self.coords_obstacle114,
                                 self.coords_obstacle115,
                                 self.coords_obstacle116,
                                 self.coords_obstacle117,
                                 self.coords_obstacle118,
                                 self.coords_obstacle119,
                                 self.coords_obstacle120,
                                 self.coords_obstacle121,
                                 self.coords_obstacle122,
                                 self.coords_obstacle123,
                                 # self.coords_belt1,
                                 # self.coords_belt2,
                                 # self.coords_belt3,
                                 self.coords_belt4,
                                 self.coords_belt5,
                                 self.coords_belt6,
                                 self.coords_belt7,
                                 self.coords_belt8,
                                 # self.coords_belt9,
                                 # self.coords_belt10,
                                 # self.coords_belt11,
                                 # self.coords_belt12,
                                 self.coords_belt13,
                                 self.coords_belt14,
                                 self.coords_belt15,
                                 self.coords_belt16,
                                 self.coords_belt17,
                                 # self.coords_belt18
                                 self.coords_belt19,
                                 self.coords_belt20,
                                 ]:

            reward = -1
            done = True
            next_state = 'obstacle'

        # elif ((np.array(next_coords[:2])[0] - (state[:2])[0]) < 0) or ((np.array(next_coords[:2])[1] - (state[:2])[1]) < 0) :
        #     reward = -0.006
        #     done = False
            # # Clearing the dictionary and the i
            # self.d = {}
            # self.i = 0


        else:
            reward = 1 - ((((state[:2] - np.array(self.canvas_widget.coords(self.flag04)[:2]))[0])**2 + ((state[:2] - np.array(self.canvas_widget.coords(self.flag04)[:2]))[1])**2)**(1/2)) / ((((np.array(self.canvas_widget.coords(self.flag4)[:2]) - np.array(self.canvas_widget.coords(self.flag04)[:2]))[0])**2 + ((np.array(self.canvas_widget.coords(self.flag4)[:2]) - np.array(self.canvas_widget.coords(self.flag04)[:2]))[1])**2)**(1/2))
            done = False
            next_state = 'walk'
            # reward = 0
            # done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas_widget.coords(self.flag04)[:2])) / (env_height * pixels)
        # print('reward', reward)
        # print((np.array(self.canvas_widget.coords(self.agent)[:2])-np.array(self.canvas_widget.coords(self.flag4)[:2])))
        # print(((np.array(next_coords[:2]) - np.array(self.canvas_widget.coords(self.flag4)[:2]))[1]-(np.array(self.canvas_widget.coords(self.agent)[:2])-np.array(self.canvas_widget.coords(self.flag4)[:2]))[1]))
        # print(np.array(next_coords[:2])-np.array(self.canvas_widget.coords(self.agent)[:2]))
        return s_, reward, done, next_state

    # Function to refresh the environment
    def render(self):
        #time.sleep(0.03)
        self.update()
        #pass

    # Function to show the found route
    def final(self):
        # Deleting the agent at the end
        self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)
        print('The shortest distance:', self.s)
        print('The longest distance:', self.lo)
        print('The most turns:', self.ll)
        print('The least turn(s):', self.ss)

        # Creating initial point
        self.initial_point = self.canvas_widget.create_oval(
            self.flag_center4[0] - 4, self.flag_center4[1] - 4,
            self.flag_center4[0] + 4, self.flag_center4[1] + 4,
            fill='white', outline='black')

        # Filling the route
        for j in range(len(self.f)):
            # Showing the coordinates of the final route
            print(self.f[j])
            self.track = self.canvas_widget.create_oval(
                self.f[j][0] - 3 + self.o[0] - 4, self.f[j][1] - 3 + self.o[1] - 4,
                self.f[j][0] - 3 + self.o[0] + 4, self.f[j][1] - 3 + self.o[1] + 4,
                fill='white', outline='black')
            # Writing the final route in the global variable a
            a[j] = self.f[j]
        for k in range(1, len(self.f) - 1):
            if not (((self.f[k - 1][0] + 20 == self.f[k][0] == self.f[k + 1][0] - 20) and (
                    self.f[k - 1][1] + 20 == self.f[k][1] == self.f[k + 1][1] - 20)) or (
                            (self.f[k - 1][0] + 20 == self.f[k][0] == self.f[k + 1][0] - 20) and (
                            self.f[k - 1][1] - 20 == self.f[k][1] == self.f[k + 1][1] + 20))
                    or ((self.f[k - 1][0] - 20 == self.f[k][0] == self.f[k + 1][0] + 20) and (
                            self.f[k - 1][1] + 20 == self.f[k][1] == self.f[k + 1][1] - 20)) or (
                            self.f[k - 1][0] - 20 == self.f[k][0] == self.f[k + 1][0] + 20) and (
                            self.f[k - 1][1] - 20 == self.f[k][1] == self.f[k + 1][1] + 20)
                    or ((self.f[k - 1][0] == self.f[k][0] == self.f[k + 1][0]) and (
                            self.f[k - 1][1] + 20 == self.f[k][1] == self.f[k + 1][1] - 20)) or (
                            (self.f[k - 1][0] == self.f[k][0] == self.f[k + 1][0]) and (
                            self.f[k - 1][1] - 20 == self.f[k][1] == self.f[k + 1][1] + 20)) or (
                            (self.f[k - 1][1] == self.f[k][1] == self.f[k + 1][1]) and (
                            self.f[k - 1][0] - 20 == self.f[k][0] == self.f[k + 1][0] + 20)) or (
                            (self.f[k - 1][1] == self.f[k][1] == self.f[k + 1][1]) and (
                            self.f[k - 1][0] + 20 == self.f[k][0] == self.f[k + 1][0] - 20))):
                self.g = self.g + 1
        if not (((self.coords_flag4[0] + 20 == self.f[0][0] == self.f[1][0] - 20) and (
                self.coords_flag4[1] + 20 == self.f[0][1] == self.f[1][1] - 20)) or (
                        (self.coords_flag4[0] + 20 == self.f[0][0] == self.f[1][0] - 20) and (
                        self.coords_flag4[1] - 20 == self.f[0][1] == self.f[1][1] + 20))
                or ((self.coords_flag4[0] - 20 == self.f[0][0] == self.f[1][0] + 20) and (
                        self.coords_flag4[1] + 20 == self.f[0][1] == self.f[1][1] - 20)) or (
                        self.coords_flag4[0] - 20 == self.f[0][0] == self.f[1][0] + 20) and (
                        self.coords_flag4[1] - 20 == self.f[0][1] == self.f[1][1] + 20)
                or ((self.coords_flag4[0] == self.f[0][0] == self.f[1][0]) and (
                        self.coords_flag4[1] + 20 == self.f[0][1] == self.f[1][1] - 20)) or (
                        (self.coords_flag4[0] == self.f[0][0] == self.f[1][0]) and (
                        self.coords_flag4[1] - 20 == self.f[0][1] == self.f[1][1] + 20)) or (
                        (self.coords_flag4[1] == self.f[0][1] == self.f[1][1]) and (
                        self.coords_flag4[0] - 20 == self.f[0][0] == self.f[1][0] + 20)) or (
                        (self.coords_flag4[1] == self.f[0][1] == self.f[1][1]) and (
                        self.coords_flag4[0] + 20 == self.f[0][0] == self.f[1][0] - 20))):
            self.g = self.g + 1
        # if not((self.flag_center4[0] - 7 == self.f[0][0] == self.f[1][0]) or (self.flag_center4[1] - 7 == self.f[0][1] == self.f[1][1]) or ((self.flag_center4[0] - 7+20 == self.f[0][0] == self.f[1][0]-20) and (self.flag_center4[1] - 7+20 == self.f[0][1] == self.f[1][1]-20)) or ((self.flag_center4[0] - 7+20 == self.f[0][0] == self.f[1][0]-20) and (self.flag_center4[1] - 7-20 == self.f[0][1] == self.f[1][1]+20))
        #     or ((self.flag_center4[0] - 7-20 == self.f[0][0] == self.f[1][0]+20) and (self.flag_center4[1] - 7+20 == self.f[0][1] == self.f[1][1]-20)) or ((self.flag_center4[0] - 7-20 == self.f[0][0] == self.f[1][0]+20) and (self.flag_center4[1] - 7-20 == self.f[1][1] == self.f[1][1]+20))):
        #     self.g = self.g + 1
        print('turn', self.g)
        for x in range(1, len(self.f)):
            self.dist = self.dist + (
                    (self.f[x][0] - self.f[x - 1][0]) ** 2 + (self.f[x][1] - self.f[x - 1][1]) ** 2) ** (
                                    1 / 2)
        self.dist = self.dist + (
                (self.f[0][0] - self.coords_flag4[0]) ** 2 + (self.f[0][1] - self.coords_flag4[1]) ** 2) ** (
                            1 / 2)
        self.l = self.dist
        print('distance', self.l)


# Returning the final dictionary with route coordinates
# Then it will be used in agent_brain.py
def final_states():
    return a


# This we need to debug the environment
# If we want to run and see the environment without running full algorithm
if __name__ == '__main__':
    env = Environment()
    env.mainloop()
