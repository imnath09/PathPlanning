
import os
import random
import time


cmd = 'nohup python MSSETester.py --mode {} --n {} --traingap {} --iter {} --testgap {} >>../img/train.log 2>&1 &'

tr = 500
it = 100
te = 10
for _ in range(1000):
    m = random.choice([0, 1])
    n = random.choice([0, 2, 7])
    c = cmd.format(m, n, tr, it, te)
    os.system(c)
    time.sleep(15)




