
import os
import time
import sys
sys.path.append('..')
from MultiSource.MultiBase import srcdata

# 记得执行脚本之前Batch.py也要加nohup


cmd = 'nohup python SPaRMTest.py --mode {} --n {} --traingap 200 --iter 400 --testgap 3 >>../img/train.log 2>&1 &'
#cmd = 'python SPaRMTest.py --mode {} --n {} --traingap 200 --iter 400 --testgap 3 '



opt = [
[2, 0],
[0, 0],
[0, 1],
[0, 2],
[0, 3],
[0, 4],
[0, 5],
[0, 6],
[0, 7],
[1, 0],
[1, 1],
[1, 2],
[1, 3],
[1, 4],
[1, 5],
[1, 6],
[1, 7],
]

for _ in range(20):
    for n in range(2):
        for m in range(len(srcdata)):
            c = cmd.format(n, m)
            os.system(c)
    os.system(cmd.format(2, 0))
    '''s = random.sample(opt, 2)
    time.sleep(60)
    for m in s:
        c = cmd.format(m[0], m[1])
        os.system(c)'''
    time.sleep(1200)
#os.system('shutdown -s -t 60')


