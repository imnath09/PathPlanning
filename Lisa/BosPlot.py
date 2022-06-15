import matplotlib.pyplot as plt

############### 服务器数据 s对应stepreward=-0.001 l对应-0.01
s4_82_107_1514=[2.99, 1.9, 1.61, 1.9, 2.59, 1.7, 1.74, 1.24, 3.28, 1.83, 1.26, 3.65, 2.6, 2.22, 1.51, 1.34, 1.6, 1.6, 1.74, 1.94, 3.24, 1.93, 2.51, 1.16, 2.93, 1.5, 1.04, 3.02, 2.65, 1.15, 2.0, 3.83, 1.19, 2.63, 1.34, 2.05, 1.74, 2.83, 1.95, 1.05, 1.46, 1.21, 1.3, 1.41, 1.3, 1.93, 3.67, 1.95, 3.99, 1.5, 2.21, 7.76, 2.7, 1.84, 4.35, 2.34, 2.56, 2.44, 1.0, 1.79, 1.97, 1.2, 1.43, 0.95, 4.19, 1.74, 1.61, 2.05, 2.34, 2.24, 2.38, 2.88, 1.31, 1.34, 1.11, 1.74, 0.91, 2.35, 2.71, 1.59, 4.94, 2.13, 3.13, 1.05, 2.54, 1.0, 3.32, 3.55, 2.08, 1.93, 1.14, 2.61, 1.97, 2.52, 3.19, 5.4, 2.18, 1.94, 2.88, 1.59]
l4_82_107_1514=[1.42, 1.26, 1.18, 1.83, 2.14, 1.66, 1.55, 2.14, 2.26, 2.1, 1.0, 1.82, 2.22, 2.67, 1.39, 1.83, 1.56, 1.24, 3.13, 1.58, 1.8, 0.84, 1.38, 2.14, 1.1, 2.42, 4.62, 2.47, 2.02, 1.65, 0.93, 1.43, 1.16, 2.16, 1.28, 1.71, 2.89, 3.94, 1.17, 2.47, 1.75, 1.43, 1.2, 1.51, 2.12, 1.38, 1.79, 1.7, 2.74, 1.28, 1.77, 2.75, 1.41, 3.07, 3.66, 1.45, 1.24, 4.19, 4.9, 4.89, 1.77, 1.59, 2.05, 2.4, 2.77, 2.55, 2.12, 2.37, 3.23, 1.35, 1.73, 2.17, 1.22, 1.96, 3.57, 2.01, 1.76, 0.84, 1.34, 2.81, 2.01, 1.48, 1.2, 3.75, 2.28, 0.89, 2.93, 0.75, 1.16, 2.33, 1.33, 2.84, 0.77, 2.26, 2.1, 1.53, 1.93, 1.8, 3.64, 2.13]

s3_82_1514=[1.32, 5.36, 2.62, 2.59, 0.78, 1.53, 1.39, 4.78, 1.25, 1.6, 2.93, 2.06, 2.12, 2.26, 2.65, 1.71, 1.89, 3.33, 4.18, 4.73, 1.64, 2.5, 2.49, 3.57, 2.87, 2.18, 2.48, 2.53, 2.63, 2.13, 3.45, 1.88, 2.85, 3.43, 1.65, 2.62, 2.03, 7.05, 1.85, 2.43, 2.65, 3.52, 2.66, 2.71, 2.87, 3.8, 2.41, 2.71, 3.18, 5.66, 4.02, 2.11, 2.21, 5.72, 2.19, 3.22, 2.15, 1.62, 1.47, 2.3, 1.81, 1.6, 6.19, 1.32, 5.13, 3.02, 2.88, 2.78, 2.04, 5.69, 1.87, 2.05, 3.2, 2.75, 3.63, 3.4, 3.35, 2.54, 2.71, 3.06, 3.28, 2.44, 4.03, 5.37, 7.12, 2.86, 2.51, 2.2, 3.9, 3.28, 2.92, 2.44, 2.68, 3.52, 2.42, 1.97, 2.73, 3.62, 1.71, 3.47]
l3_82_1514=[1.45, 1.69, 1.34, 2.52, 1.99, 2.15, 2.55, 4.41, 4.4, 3.58, 3.85, 0.86, 3.58, 3.07, 5.72, 4.26, 3.29, 4.57, 2.82, 3.03, 2.12, 2.56, 1.78, 1.97, 2.94, 2.82, 4.21, 7.19, 1.5, 1.23, 1.91, 1.15, 3.03, 6.19, 5.16, 1.69, 1.28, 2.29, 5.73, 5.14, 3.32, 1.52, 2.42, 2.17, 4.85, 5.54, 3.36, 4.63, 1.93, 3.26, 1.79, 2.41, 3.87, 3.75, 4.33, 3.25, 0.81, 2.43, 1.82, 7.14, 3.13, 1.86, 3.39, 2.9, 2.66, 2.11, 2.53, 4.08, 4.27, 2.76, 2.5, 4.01, 4.19, 1.61, 4.16, 6.7, 2.62, 3.99, 2.63, 2.17, 1.83, 3.78, 3.92, 3.2, 2.96, 2.4, 3.53, 2.83, 1.24, 3.73, 2.91, 2.8, 3.92, 1.98, 10.15, 1.82, 2.45, 2.9, 4.01, 2.8]

s2_82=[18.51, 21.5, 19.9, 14.54, 17.07, 9.76, 12.85, 17.91, 18.64, 12.54, 17.39, 16.42, 15.06, 11.66, 16.03, 18.46, 18.07, 18.18, 16.14, 25.96, 16.81, 25.99, 22.06, 19.16, 18.9, 20.73, 15.25, 15.16, 12.6, 12.99, 11.07, 15.23, 18.79, 19.46, 20.36, 8.9, 16.34, 19.3, 9.39, 24.57, 19.17, 17.31, 12.72, 16.09, 19.37, 15.72, 18.4, 20.75, 16.97, 13.17, 20.92, 17.52, 20.81, 19.96, 14.9, 15.99, 18.57, 23.26, 15.88, 15.04, 20.03, 23.97, 14.84, 14.32, 14.79, 18.27, 24.62, 23.35, 21.27, 14.56, 19.6, 21.29, 16.76, 15.0, 28.07, 23.66, 12.44, 22.51, 22.63, 24.81, 27.86, 25.23, 22.96, 19.33, 21.14, 24.68, 14.96, 26.13, 14.61, 19.87, 20.72, 15.57, 19.43, 17.96, 13.48, 29.76, 12.99, 18.08, 18.43, 17.99]
l2_82=[15.51, 22.5, 13.51, 16.04, 15.36, 21.0, 21.17, 16.96, 19.09, 25.89, 15.48, 16.6, 17.18, 18.76, 18.62, 18.99, 18.18, 27.57, 20.66, 25.52, 15.07, 23.16, 17.84, 22.98, 20.11, 15.63, 21.79, 16.86, 13.81, 30.73, 14.34, 16.47, 13.52, 25.57, 14.48, 15.34, 20.14, 23.26, 19.5, 14.3, 18.76, 20.68, 19.0, 19.06, 11.55, 24.22, 14.1, 22.8, 21.52, 19.49, 22.61, 15.87, 15.0, 19.03, 16.55, 12.83, 23.27, 20.75, 22.65, 20.69, 20.81, 24.26, 23.15, 23.09, 17.49, 21.37, 18.63, 18.81, 21.18, 20.3, 21.27, 19.78, 18.26, 17.41, 15.87, 21.76, 25.91, 11.52, 17.79, 16.78, 19.79, 18.94, 25.29, 22.67, 19.51, 23.62, 20.07, 21.37, 15.74, 20.28, 20.23, 20.4, 12.38, 19.87, 16.9, 20.12, 17.85, 17.33, 11.66, 27.39]

s2_107=[8.62, 15.05, 11.57, 12.31, 24.05, 6.34, 17.47, 12.95, 10.97, 15.86, 20.62, 16.63, 21.0, 15.03, 17.57, 15.72, 12.26, 13.56, 16.95, 11.18, 13.1, 14.69, 14.9, 18.13, 7.77, 16.6, 17.24, 14.19, 17.8, 19.5, 16.39, 11.3, 20.75, 16.54, 14.91, 13.24, 19.35, 9.45, 11.27, 20.03, 7.47, 22.23, 13.46, 18.32, 19.18, 19.62, 18.95, 19.82, 11.1, 19.6, 17.27, 8.09, 22.22, 18.17, 10.74, 17.24, 17.58, 15.98, 5.81, 21.16, 8.19, 11.21, 22.51, 17.4, 16.21, 19.68, 10.1, 15.61, 11.2, 15.89, 19.85, 11.91, 12.04, 9.47, 10.89, 19.4, 17.06, 14.04, 14.48, 14.18, 20.35, 24.11, 14.8, 13.28, 9.08, 17.43, 20.81, 20.16, 17.4, 19.95, 24.25, 21.13, 14.02, 25.71, 3.8, 16.83, 22.42, 23.02, 20.73, 19.83]
l2_107=[27.63, 18.93, 21.57, 15.24, 18.6, 23.56, 21.22, 19.62, 12.92, 20.03, 15.68, 13.23, 17.55, 19.54, 18.65, 9.09, 15.29, 13.32, 24.27, 20.33, 22.0, 22.12, 21.37, 17.92, 18.44, 13.21, 6.56, 10.36, 29.01, 18.79, 21.9, 20.1, 15.5, 17.7, 18.54, 18.53, 24.54, 19.77, 19.2, 21.49, 20.55, 15.66, 13.2, 16.64, 19.04, 18.01, 13.07, 30.3, 17.96, 21.03, 22.32, 11.65, 14.79, 18.96, 19.51, 19.63, 16.94, 13.81, 15.49, 18.95, 15.58, 15.67, 17.24, 23.2, 25.16, 15.91, 13.08, 15.59, 12.74, 17.53, 9.0, 15.27, 19.29, 21.35, 19.29, 21.3, 16.49, 22.2, 17.57, 16.2, 17.22, 21.18, 15.61, 21.79, 16.87, 18.03, 20.16, 15.28, 9.16, 19.4, 18.32, 17.63, 19.85, 19.7, 14.77, 13.9, 17.59, 15.58, 13.86, 17.89]

s2_1310=[3.2, 3.98, 2.68, 2.45, 2.27, 6.12, 3.01, 3.06, 3.72, 2.95, 2.62, 5.16, 3.64, 5.93, 6.37, 9.48, 2.46, 2.47, 8.26, 2.78, 1.87, 3.35, 1.92, 4.23, 8.33, 2.39, 14.01, 4.64, 3.44, 2.4, 8.62, 2.38, 3.03, 3.64, 7.23, 3.77, 2.48, 5.32, 4.83, 2.02, 6.24, 2.66, 3.28, 3.76, 2.23, 2.54, 9.48, 4.91, 3.54, 4.72, 4.47, 3.79, 7.8, 2.98, 11.11, 3.4, 8.54, 11.74, 2.11, 3.68, 2.54, 4.45, 2.52, 1.5, 1.99, 2.83, 3.17, 4.42, 3.15, 2.86, 7.12, 3.46, 2.66, 4.48, 4.65, 2.94, 3.98, 4.4, 3.98, 5.38, 4.1, 3.78, 2.51, 2.36, 3.22, 5.02, 2.54, 2.77, 8.36, 4.83, 14.83, 2.44, 2.86, 3.28, 3.66, 4.21, 11.61, 4.72, 3.34, 3.96]
l2_1310=[2.42, 5.1, 1.99, 4.64, 2.32, 5.2, 1.86, 5.33, 10.53, 2.14, 2.48, 2.92, 4.87, 3.72, 1.79, 3.25, 4.19, 7.88, 9.43, 2.29, 4.22, 1.85, 2.49, 4.82, 2.53, 2.3, 2.07, 3.54, 3.66, 2.71, 8.37, 2.4, 5.94, 6.43, 5.14, 3.3, 4.48, 8.63, 5.09, 5.82, 3.38, 1.51, 2.55, 8.12, 5.29, 5.91, 2.82, 9.65, 2.27, 4.39, 2.28, 10.02, 5.5, 3.54, 2.89, 2.79, 4.22, 4.55, 1.47, 2.81, 6.94, 9.74, 1.67, 2.61, 4.12, 3.04, 2.42, 1.93, 2.27, 1.71, 5.33, 4.67, 8.55, 2.08, 2.84, 2.95, 2.91, 2.64, 2.73, 3.58, 2.99, 4.23, 12.1, 4.44, 9.58, 3.65, 2.11, 4.18, 2.15, 2.35, 5.6, 6.01, 4.28, 3.55, 2.39, 6.35, 3.95, 3.3, 1.28, 2.8]

s2_1514=[5.5, 6.01, 4.87, 4.6, 3.3, 3.18, 6.81, 3.2, 3.41, 3.01, 2.86, 6.02, 4.03, 4.04, 2.86, 2.63, 2.53, 2.2, 4.42, 4.95, 3.11, 2.78, 3.78, 6.3, 2.92, 3.61, 5.02, 3.38, 4.59, 4.69, 3.2, 4.62, 2.66, 5.8, 3.65, 4.38, 1.93, 4.1, 3.55, 5.75, 3.39, 7.16, 3.73, 4.43, 5.77, 5.05, 7.45, 3.61, 4.4, 2.3, 3.78, 4.1, 4.46, 4.39, 3.57, 2.92, 3.2, 3.96, 2.48, 4.48, 4.47, 6.18, 2.45, 4.49, 1.95, 5.16, 4.11, 4.51, 4.24, 3.38, 6.05, 2.74, 2.29, 4.6, 3.2, 2.7, 5.09, 3.58, 5.82, 3.07, 4.44, 2.34, 3.88, 4.85, 5.76, 4.44, 7.07, 4.4, 3.53, 4.16, 3.2, 5.77, 5.14, 3.86, 6.28, 3.97, 3.78, 3.64, 5.75, 6.35]
l2_1514=[3.18, 1.91, 4.83, 5.56, 4.31, 4.42, 5.07, 4.73, 4.14, 4.02, 3.15, 5.17, 2.96, 3.21, 2.97, 4.62, 6.79, 5.1, 3.16, 5.4, 5.51, 6.75, 4.89, 4.15, 6.85, 2.35, 5.15, 3.4, 4.02, 4.81, 3.5, 4.33, 2.9, 3.71, 4.11, 4.18, 3.79, 2.43, 3.45, 6.58, 5.28, 6.93, 6.08, 5.67, 4.97, 2.11, 5.66, 4.97, 3.07, 4.93, 2.8, 2.63, 4.28, 3.23, 5.13, 3.92, 1.54, 6.1, 5.02, 6.39, 3.46, 4.42, 5.24, 4.3, 3.09, 3.59, 4.69, 4.59, 3.41, 3.93, 2.13, 4.81, 3.33, 2.39, 5.02, 4.84, 2.83, 5.08, 6.35, 3.11, 2.61, 2.58, 3.42, 2.38, 3.57, 5.48, 6.04, 2.97, 5.98, 6.8, 4.42, 2.45, 2.55, 4.84, 2.27, 5.54, 5.9, 3.76, 4.47, 3.13]

s2_157=[2.3, 18.81, 9.15, 11.61, 10.68, 6.6, 1.82, 6.74, 10.55, 8.83, 8.91, 8.45, 5.54, 6.72, 17.03, 4.43, 12.12, 19.46, 19.81, 13.11, 21.73, 7.92, 17.59, 4.9, 4.86, 15.19, 6.81, 6.81, 14.47, 18.77, 11.04, 12.25, 12.94, 12.2, 4.19, 16.93, 8.34, 4.71, 5.14, 11.53, 7.27, 14.48, 9.16, 16.66, 11.99, 13.35, 6.77, 12.06, 10.83, 10.58, 8.09, 14.68, 11.0, 12.58, 15.05, 18.84, 15.81, 9.01, 18.24, 6.65, 12.73, 13.34, 14.73, 13.54, 7.34, 14.22, 14.54, 10.48, 7.18, 7.68, 10.34, 13.29, 5.53, 9.15, 9.05, 11.1, 18.41, 8.6, 5.22, 8.91, 15.49, 14.09, 8.85, 3.68, 5.69, 12.79, 13.03, 6.87, 6.33, 7.49, 10.74, 17.68, 14.06, 15.27, 7.61, 14.64, 7.85, 8.7, 8.86, 23.15]
l2_157=[9.75, 14.26, 5.57, 14.96, 26.63, 3.9, 14.2, 9.2, 11.54, 14.06, 21.92, 10.58, 9.59, 10.01, 5.13, 12.81, 17.57, 17.07, 10.48, 8.1, 14.89, 17.24, 6.4, 15.24, 19.66, 10.44, 6.7, 9.45, 11.89, 8.93, 13.5, 5.81, 15.8, 4.72, 2.31, 15.11, 13.81, 11.59, 5.14, 6.09, 9.6, 19.86, 15.27, 10.12, 14.75, 14.9, 11.19, 8.28, 17.8, 13.63, 9.94, 4.89, 9.51, 10.21, 5.85, 7.12, 9.85, 13.18, 16.58, 10.3, 17.49, 16.64, 16.0, 10.84, 7.59, 5.31, 3.72, 13.46, 20.18, 9.43, 19.65, 11.14, 13.8, 7.49, 2.43, 11.96, 14.91, 10.36, 9.88, 5.13, 11.6, 12.55, 13.27, 14.71, 19.07, 11.88, 12.48, 16.12, 6.4, 14.35, 14.63, 14.34, 10.67, 8.11, 6.77, 3.85, 14.29, 13.51, 6.57, 18.52]

s1=[31.32, 15.97, 23.67, 29.12, 23.45, 29.49, 25.41, 28.1, 17.92, 20.12, 22.46, 19.2, 14.36, 10.22, 22.76, 29.22, 29.49, 19.95, 27.25, 26.46, 27.89, 19.96, 16.9, 23.7, 21.32, 19.88, 33.57, 10.41, 27.19, 27.47, 23.02, 20.26, 25.04, 26.71, 26.32, 16.81, 20.36, 31.76, 19.22, 29.66, 26.26, 21.6, 22.14, 20.17, 18.98, 20.11, 25.6, 26.36, 23.62, 35.27, 32.5, 24.41, 23.27, 33.68, 29.61, 19.29, 23.59, 29.09, 23.43, 25.11, 25.29, 28.62, 24.52, 15.03, 22.98, 19.78, 28.47, 24.68, 18.96, 21.82, 23.11, 19.74, 23.34, 26.02, 24.59, 27.45, 24.9, 22.21, 31.71, 17.49, 17.58, 25.64, 21.02, 17.78, 23.97, 23.17, 18.69, 18.78, 18.52, 29.34, 26.95, 21.16, 19.12, 26.12, 19.89, 25.02, 21.15, 20.25, 20.38, 24.23]
l1=[27.36, 23.53, 33.72, 19.12, 26.37, 33.03, 32.14, 23.36, 24.62, 34.06, 28.45, 17.43, 22.77, 10.03, 23.95, 21.02, 18.22, 21.54, 20.93, 22.87, 24.01, 18.0, 22.58, 18.73, 20.02, 23.01, 27.18, 22.21, 27.49, 17.65, 23.04, 23.81, 19.31, 19.98, 25.7, 20.99, 19.89, 27.18, 23.45, 21.45, 19.02, 26.0, 21.25, 27.52, 25.14, 24.22, 20.97, 23.63, 17.11, 27.57, 23.01, 24.15, 34.42, 29.0, 20.15, 28.92, 23.85, 24.8, 34.09, 24.61, 15.62, 12.85, 22.11, 25.78, 27.72, 25.27, 25.42, 22.86, 29.89, 29.45, 23.91, 20.36, 13.5, 16.95, 20.82, 19.99, 19.96, 18.72, 22.34, 22.77, 25.12, 33.56, 25.2, 16.44, 23.12, 16.89, 19.58, 19.47, 24.5, 20.35, 23.29, 20.85, 16.56, 23.26, 23.22, 18.07, 20.98, 27.66, 15.68, 15.01]
#五个源性能反而下降了
s5_82_107_1514_1310=[0.86, 4.9, 1.48, 1.81, 1.17, 1.96, 3.91, 1.65, 0.98, 1.55, 5.23, 1.05, 1.82, 1.61, 6.97, 7.26, 12.8, 7.16, 11.95, 1.95, 2.01, 2.01, 1.95, 0.94, 0.82, 2.65, 1.87, 1.32, 3.87, 7.79, 9.85, 4.22, 15.21, 3.37, 18.34, 9.08, 1.43, 1.03, 0.89, 32.4, 2.26, 0.53, 1.03, 1.08, 1.82, 1.1, 1.22, 1.35, 0.72, 2.96, 1.95, 1.11, 1.2, 0.79, 1.75, 1.5, 1.28, 4.03, 0.94, 1.12, 0.92, 4.52, 2.99, 6.73, 1.16, 4.92, 2.16, 1.01, 0.78, 1.78, 1.02, 4.17, 3.25, 0.6, 17.31, 1.61, 5.09, 3.1, 3.15, 1.68, 1.16, 8.8, 1.27, 1.26, 1.08, 1.33, 1.32, 10.53, 8.17, 1.33, 7.01, 0.89, 1.46, 0.93, 4.98, 1.03, 4.01, 2.15, 1.37, 8.21]
l5_82_107_1514_1310=[17.52, 70.39, 5.02, 2.7, 8.0, 1.22, 0.95, 33.99, 0.83, 1.28, 13.58, 2.63, 1.23, 1.3, 3.99, 2.49, 6.5, 1.38, 15.62, 0.86, 1.39, 4.03, 4.37, 3.53, 2.77, 3.44, 1.07, 1.24, 0.87, 0.53, 1.78, 8.39, 3.86, 0.59, 1.59, 16.22, 3.23, 7.87, 22.55, 2.5, 0.57, 11.58, 7.94, 1.2, 2.29, 0.65, 8.74, 2.83, 5.78, 1.04, 1.4, 1.71, 9.5, 0.83, 3.32, 2.14, 2.0, 0.78, 1.78, 1.79, 7.15, 0.65, 14.8, 2.42, 4.27, 1.0, 1.62, 2.36, 1.55, 4.07, 2.52, 24.81, 0.64, 2.64, 1.32, 1.16, 1.21, 5.73, 0.98, 1.88, 25.65, 13.79, 7.81, 1.32, 8.26, 9.33, 1.92, 1.4, 4.01, 3.47, 30.45, 1.05, 0.94, 1.29, 18.79, 2.7, 0.68, 1.14, 0.92, 0.66]


svr = [    
    s1,
    l1,
    #s2_82,
    #l2_82,
    #s2_107,
    #l2_107,
    s2_157,
    l2_157,
    #s2_1514,
    #l2_1514,
    #s2_1310,
    #l2_1310,
    s3_82_1514,
    l3_82_1514,
    s4_82_107_1514,
    l4_82_107_1514,
    #s5_82_107_1514_1310,
    #l5_82_107_1514_1310,
]

plt.figure(figsize=(10,10))
plt.boxplot(
    #x=svr,
    x=svr[1::2],
    #x=[svr[0],svr[2],svr[8],svr[10]],
    #x=[svr[1],svr[3],svr[13],svr[14]],
    notch=False,
    showmeans=True,
    meanline=True,
    patch_artist=False,
    widths=0.2,
    #boxprops={'color':'blue',},#设置箱体属性，填充色和边框色
    flierprops={'marker':'o','markerfacecolor':'#9999ff'},#设置异常值属性，点的形状、填充颜色和边框色
    meanprops={'linestyle':'dotted','color':'red'},#设置均值点的属性，点的颜色和形状
    medianprops={'linestyle':'-','color':'orange'},#设置中位数线的属性，线的类型和颜色
    labels=['n=1', 'n=2', 'n=3', 'n=4']
    )
plt.tight_layout()
plt.show()

