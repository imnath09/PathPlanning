
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
'nohup python ../MultiSource/MultipleReversal.py --mode 1 --n 7 >>../img/train.log 2>&1 &',]
planning = [
'nohup python MSSETester.py --mode 2 --n 2 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 0 --n 0 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 0 --n 2 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 0 --n 7 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 0 --n 8 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 1 --n 0 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 1 --n 2 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 1 --n 7 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
'nohup python MSSETester.py --mode 1 --n 8 --traingap 200 --iter 200 --testgap 3 >>../img/train.log 2>&1 &',
]

for _ in range(10):
    for me in planning:
        os.system(me)
    time.sleep(1200)


'''
tr = 500
it = 100
te = 10

m=[0, 1]
n=[0, 2, 7]
for _ in range(1000):
    m = random.choice([0, 1])
    n = random.choice([0, 2, 7])
    c = planning.format(m, n, tr, it, te)
    os.system(c)
    time.sleep(15)
'''



