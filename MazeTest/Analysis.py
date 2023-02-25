
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fsize = (18, 9)
linestyle1= ['-', '--', '-.', ':',]
linestyles2 = ['-']*4
SOME_COLORS = (
    ('blue', '#1f77b4'),
    ('orange', '#ff7f0e'),
    ('green', '#2ca02c'),
    ('red', '#d62728'),
    ('purple', '#9467bd'),
    ('brown', '#8c564b'),
    ('pink', '#e377c2'),
    ('gray', '#7f7f7f'),
    ('olive', '#bcbd22'),
    ('cyan', '#17becf'),
)

scolors = [
    'blue','orange','yellow','green','red','pink','gray','purple','olive','cyan','brown'
]

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
        time_info = [float(x) for x in lines[6].split('[')[1].split(']')[0].split(',')]
    return test_rate, test_len, train_rate, train_len, test_rewards, train_rewards, time_info

def draw(title, datas, colors, labels, savename, linestyles = linestyle1):
    al = 0.7
    plt.figure(figsize=fsize)
    plt.suptitle(title)
    for i in range(len(datas)):
        plt.plot(
            datas[i],
            colors[i],
            linestyle = linestyles[i % 4],
            alpha = al,
            label = labels[i],
            )
    plt.ylabel(title)
    plt.xlabel('horizon')
    #plt.ylim(-0.05, 0.9)
    plt.grid(visible=True,axis='x')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def analyze(filename):
    test_rate, test_len, train_rate, train_len, test_rewards, train_rewards, time_info = readfile(filename)

    colors = ['r', 'b']
    draw('success rate', [test_rate, train_rate], colors, ['test rate', 'train rate'], '{}.rate.png'.format(filename), linestyles = linestyles2)

    draw('average length', [test_len, train_len], colors, ['test length', 'train length'], '{}.len.png'.format(filename), linestyles = linestyles2)

    draw('average cumulative reward', [test_rewards, train_rewards], colors, ['test reward', 'train reward'], '{}.reward.png'.format(filename), linestyles = linestyles2)

def analyze2(files):
    '''比较不同实验的六个数据'''
    datas = [readfile(x) for x in files] # 数量是文本数量
    #test_rate, test_len, train_rate, train_len, test_reward, train_reward, time_info = readfile(f1)

    #get_color = lambda : "#" + "%06x" % random.randint(0, 0xFFFFFF)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(len(datas))

    # 测试成功率
    test_rates = [x[0][0:sx] for x in datas]
    draw('test rates', test_rates, colors, files, '../img/test_rate.png')

    # 测试长度
    test_lens = [x[1][0:sx] for x in datas]
    draw('test length', test_lens, colors, files, '../img/test_length.png')

    # 训练成功率
    train_rates = [x[2][0:sx] for x in datas]
    draw('train rates', train_rates, colors, files, '../img/train_rate.png')

    # 训练长度
    train_lens = [x[3][0:sx] for x in datas]
    draw('train length', train_lens, colors, files, '../img/train_length.png')

    # 平均测试累积奖励
    test_rewards = [x[4][0:sx] for x in datas]
    draw('test reward', test_rewards, colors, files, '../img/test_reward.png')

    # 平均训练累积奖励
    train_rewards = [x[5][0:sx] for x in datas]
    draw('train reward', train_rewards, colors, files, '../img/train_reward.png')

    print('finish')

def SuccessRate(files, fname):
    '''把同一个实验的多个实例画进同一个图'''
    files = [x for x in files if x.find(fname) >= 0]
    datas = [readfile(x) for x in files] # 数量是文本数量
    #test_rate, test_len, train_rate, train_len, test_reward, train_reward = readfile(f1)

    #get_color = lambda : "#" + "%06x" % random.randint(0, 0xFFFFFF)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(len(datas))

    # 测试成功率
    train_rates = [x[2][0:sx] for x in datas]
    draw('train rates', train_rates, colors, files, '../img/{}/{}.png'.format(fpath, fname), linestyle1)

    #print('finish')

def Confidence_Interval(files, fname):
    '''把同一个实验的多个实例画进同一个图'''
    files = [x for x in files if x.find(fname) >= 0]
    datas = [readfile(x) for x in files]
    train_rates = [x[2] for x in datas] # 成功率
    m, y1, y2 = [], [], []
    xr = range(len(train_rates[0]))
    for i in xr:
        v = [x[i] for x in train_rates]
        avr = np.mean(v)
        se = stats.sem(v)
        m.append(avr)
        y2.append(avr + 1.96 * se)
        y1.append(avr - 1.96 * se)
    return xr, m, y1, y2

def draw_confidences(datas, exps):
    '''
    datas:所有的数据
    exps:所有的实验名
    '''
    i=0
    plt.figure(figsize=fsize)
    plt.suptitle('success rate')
    for e in exps:
        xr, m, y1, y2 = Confidence_Interval(datas, e)
        plt.fill_between(xr, y1, y2, facecolor=scolors[i], alpha=0.3)#color='red', 
        plt.plot(xr, m, color=scolors[i], label=e)
        i += 1
    plt.legend()
    plt.savefig('../img/{}/succr.png'.format(fpath))

def boxplot():
    pass#以后再搞了

if __name__ == '__main__':
    sx = -1
    fpath='14'
    all_data = []
    all_exp=[]
    for o,d,f in os.walk('d:\\code\\PathPlanning\\img\\'+fpath):
        for ff in f:
            if ff.find('tr200')>=0:
                fp = os.path.join(o,ff)
                all_data.append(fp)
            elif ff.find('.txt')>=0:
                all_exp.append(ff.strip('.txt')[0:])
        break

    draw_confidences(all_data, all_exp)

    ''' 
    for x in all_exp:
        SuccessRate(all_data,x)
    '''

    '''
    t = 'SP4_1514_107_82'
    a = [x for x in b if x.find(t) >= 0]
    analyze2(a)
    '''






