

from torch.utils.tensorboard import SummaryWriter
import os

def readfile(fname):
    if fname == '':
        return [None, None, None, None]
    with open(fname, 'r', encoding='utf-8') as file1:
        lines = file1.readlines()
        lines = [x.strip() for x in lines]
        test_rate = [float(x) for x in lines[0].split(',')]
        test_len = [float(x) for x in lines[1].split(',')]
        train_rate = [float(x) for x in lines[2].split(',')]
        train_len = [float(x) for x in lines[3].split(',')]
        test_rewards = [float(x) for x in lines[4].split(',')]#None#
        train_rewards = [float(x) for x in lines[5].split(',')]#None#
    return test_rate, test_len, train_rate, train_len, test_rewards, train_rewards



def dosth():
    b=[]
    for o,d,f in os.walk('d:\\code\\PathPlanning\\img\\200_150_3 jumpgap 500'):
        for ff in f:
            if ff.find('tr200')>=0 and ff.find('SP2_1514')>=0:
                fp = os.path.join(o,ff)
                b.append(fp)


    writer = SummaryWriter(log_dir='../img/tblog')
    files = [x for x in b]
    datas = [readfile(x) for x in files] # 数量是文本数量
    train_rates = [x[2][0:150] for x in datas]

    print(len(train_rates))

    for i in range(len(train_rates[0])):
        a={}
        for j in range(len(train_rates)):
            a[str(j)] = train_rates[j][i]
        writer.add_scalars(
            'trainrates',
            a,
            i
        )
    writer.close()


dosth()








