
import os
import random
import time

# 记得执行脚本之前也要加nohup

merge_exploration = [
'nohup python ../MultiSource/MultipleReversal.py --mode 0 --n 0 >>../img/train.log 2>&1 &',
'nohup python ../MultiSource/MultipleReversal.py --mode 0 --n 2 >>../img/train.log 2>&1 &',
'nohup python ../MultiSource/MultipleReversal.py --mode 0 --n 7 >>../img/train.log 2>&1 &',
'nohup python ../MultiSource/MultipleReversal.py --mode 1 --n 0 >>../img/train.log 2>&1 &',
'nohup python ../MultiSource/MultipleReversal.py --mode 1 --n 2 >>../img/train.log 2>&1 &',
'nohup python ../MultiSource/MultipleReversal.py --mode 1 --n 7 >>../img/train.log 2>&1 &',
]
planning = [
#'nohup python MSSETester.py --mode 2 --n 2 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 0 --n 0 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 0 --n 2 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
#'nohup python MSSETester.py --mode 0 --n 7 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 0 --n 8 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 1 --n 0 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 1 --n 2 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
#'nohup python MSSETester.py --mode 1 --n 7 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 1 --n 8 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
]


cmd = 'nohup python MSSETester.py --mode {} --n {} --traingap 200 --iter 150 --testgap 3 >>../img/train.log 2>&1 &'

'''
for _ in range(10):
    for me in planning:
        os.system(me)
    time.sleep(1200)
'''

tr = 500
it = 100
te = 10

opt = [
    [2, 0],
    [0, 0],
    [0, 2],
    [0, 7],
    [1, 0],
    [1, 2],
    [1, 7],
]

for _ in range(10):
    for m in opt:
        c = cmd.format(m[0], m[1])
        os.system(c)
    s = random.sample(opt, 2)
    time.sleep(60)
    for m in s:
        c = cmd.format(m[0], m[1])
        os.system(c)
    time.sleep(1440)



