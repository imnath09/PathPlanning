
import os
import random
import matplotlib.pyplot as plt

fsize = (18, 9)
linestyle1= ['-', '--', '-.', ':',]
linestyles2 = ['-']*4

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

def draw(title, datas, colors, labels, savename, linestyles = linestyle1):
    al = 0.7
    plt.figure(figsize=fsize)
    plt.suptitle(title)
    for i in range(len(datas)):
        plt.plot(
            datas[i], colors[i], linestyle = linestyles[i % 4],
            label = labels[i], alpha = al)
    plt.ylabel(title)
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def analyze(filename):
    test_rate, test_len, train_rate, train_len, test_rewards, train_rewards = readfile(filename)

    colors = ['r', 'b']
    draw('success rate', [test_rate, train_rate], colors, ['test rate', 'train rate'], '{}.rate.png'.format(filename), linestyles = linestyles2)

    draw('average length', [test_len, train_len], colors, ['test length', 'train length'], '{}.len.png'.format(filename), linestyles = linestyles2)

    draw('average cumulative reward', [test_rewards, train_rewards], colors, ['test reward', 'train reward'], '{}.reward.png'.format(filename), linestyles = linestyles2)

def analyze2(files):
    datas = [readfile(x) for x in files] # 数量是文本数量
    #test_rate, test_len, train_rate, train_len, test_reward, train_reward = readfile(f1)

    #get_color = lambda : "#" + "%06x" % random.randint(0, 0xFFFFFF)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(len(datas))
    sx = 50

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

if __name__ == '__main__':
    def fn(a):
        return '../img/{}/data.txt'.format(a)
    def getdir():
        for o, d, f in os.walk('d:\\code\\pathplanning\\img'):
            return d
    s =[
 '06-17 16.24.46 tr1000it100ts10 ql',
 '06-17 16.25.42 tr1000it100ts10 spase4',
 '06-17 21.06.30 tr1000it100ts10 m2n4 1QLearning',
 '06-17 21.07.03 tr1000it100ts10 m1n1 2RFE',
 '06-17 21.08.34 tr1000it100ts10 m0n1 3SE',
 '06-17 21.09.02 tr1000it100ts10 m1n4 4SP',
 '06-17 21.09.30 tr1000it100ts10 m0n4 5SPASE',
 '06-17 23.04.29 tr1000it100ts10 m0n4 spase',
 '06-17 23.04.36 tr1000it100ts10 m1n4 sp',
 '06-17 23.04.39 tr1000it100ts10 m0n1 se',
 '06-17 23.04.42 tr1000it100ts10 m1n1 rfe',
 '06-17 23.04.45 tr1000it100ts10 m2n4 ql',
 '06-18 00.14.03 tr1000it100ts10 ql',
 '06-18 00.14.15 tr1000it100ts10 rfe',
 '06-18 00.14.18 tr1000it100ts10 se',
 '06-18 00.14.20 tr1000it100ts10 sp4',
 '06-18 00.14.23 tr1000it100ts10 spase4',
 '06-18 00.20.33 tr1000it100ts10 sp2',
 '06-18 00.23.02 tr1000it100ts10 spase3',
 '06-18 00.23.18 tr1000it100ts10 spase2',
 '06-18 00.23.36 tr1000it100ts10 spase2',
 '06-18 00.24.06 tr1000it100ts10 spase2',
 '06-18 00.24.20 tr1000it100ts10 spase2',
 '06-18 00.24.33 tr1000it100ts10 spase2',
 'QLearning tr1000it100ts10 06-18 17.28.37',
 'RFE tr1000it100ts10 06-18 17.32.12',
 'SE tr1000it100ts10 06-18 17.31.31',
 'SP2_1514 tr1000it100ts10 06-18 17.32.03',
 'SP4_82_107_1514 tr1000it100ts10 06-18 17.31.52',
 'SPaSE2_1514 tr1000it100ts10 06-18 17.31.14',
 'SPaSE4_82_107_1514 tr1000it100ts10 06-18 17.30.58']

    good =[
'06-17 23.04.36 tr1000it100ts10 m1n4 sp',
'06-18 00.20.33 tr1000it100ts10 sp2',
'SP2_1514 tr1000it100ts10 06-18 17.32.03',
'06-17 21.07.03 tr1000it100ts10 m1n1 2RFE',
'06-17 23.04.42 tr1000it100ts10 m1n1 rfe',
'RFE tr1000it100ts10 06-18 17.32.12',
'06-18 00.14.15 tr1000it100ts10 rfe',
]



    '''
    for x in rfe:
        analyze(x)
    for x in msse:
        analyze(x)
    for x in ql:
        analyze(x)
    '''
    #analyze2(rfe + msse + ql)
    #for x in ssss:
    #    analyze(x)

    #print(getdir())
    analyze2([fn(x) for x in good])
    #analyze2([rfe1[4], msse1[4], qlearning1[4]])
