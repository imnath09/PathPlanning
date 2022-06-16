
import random
import matplotlib.pyplot as plt

fsize = (18, 9)
linestyles = ['-', '--', '-.', ':',]
linestyles1 = ['-']*4

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

def draw(title, datas, colors, labels, savename):
    al = 0.7
    plt.figure(figsize=fsize)
    plt.suptitle(title)
    for i in range(len(datas)):
        plt.plot(
            datas[i], colors[i], linestyle = linestyles1[i % 4],
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
    draw('success rate', [test_rate, train_rate], colors, ['test rate', 'train rate'], '{}.rate.png'.format(filename))

    draw('average length', [test_len, train_len], colors, ['test length', 'train length'], '{}.len.png'.format(filename))

    draw('average cumulative reward', [test_rewards, train_rewards], colors, ['test reward', 'train reward'], '{}.reward.png'.format(filename))

def analyze2(files):
    datas = [readfile(x) for x in files] # 数量是文本数量
    #test_rate, test_len, train_rate, train_len, test_reward, train_reward = readfile(f1)

    #get_color = lambda : "#" + "%06x" % random.randint(0, 0xFFFFFF)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(len(datas))

    # 测试成功率
    test_rates = [x[0][0:50] for x in datas]
    draw('test rates', test_rates, colors, files, '../img/test_rate.png')

    # 测试长度
    test_lens = [x[1][0:50] for x in datas]
    draw('test length', test_lens, colors, files, '../img/test_length.png')

    # 训练成功率
    train_rates = [x[2][0:50] for x in datas]
    draw('train rates', train_rates, colors, files, '../img/train_rate.png')

    # 训练长度
    train_lens = [x[3][0:50] for x in datas]
    draw('train length', train_lens, colors, files, '../img/train_length.png')

    # 平均测试累积奖励
    test_rewards = [x[4][0:50] for x in datas]
    draw('test reward', test_rewards, colors, files, '../img/test_reward.png')

    # 平均训练累积奖励
    train_rewards = [x[5][0:50] for x in datas]
    draw('train reward', train_rewards, colors, files, '../img/train_reward.png')

    print('finish')

if __name__ == '__main__':
    def fn(a, b):
        return '../img/{}/{}.txt'.format(a, b)
    rfe = [
        fn('06-10 20.50.10 tr1000it100ts10 RFE', 'RFE'),
        fn('06-15 19.07.47 tr1000it100ts10 RFE', 'RFE'),
        fn('06-15 22.58.01 tr1000it100ts10 RFE', 'RFE'),#左
        fn('06-16 14.06.52 tr1000it100ts10 RFE', 'RFE'),#右
        fn('06-16 17.51.42 tr500it200ts10 RFE', 'RFE'),
    ]
    msse = [
        fn('06-09 20.29.40 tr1000it100ts10 MSSE', 'MSSE'),
        fn('06-15 20.17.54 tr1000it100ts10 MSSE', 'MSSE'),#右
        fn('06-15 22.57.38 tr1000it100ts10 MSSE', 'MSSE'),
        fn('06-16 14.06.34 tr1000it100ts10 MSSE', 'MSSE'),#左
        fn('06-16 17.52.00 tr500it200ts10 MSSE', 'MSSE'),
    ]
    qlearning = [
        fn('06-09 20.29.34 tr1000it100ts10 QLearning', 'QLearning'),
        fn('06-15 20.17.43 tr1000it100ts10 QLearning', 'QLearning'),
        fn('06-15 22.56.54 tr1000it100ts10 QLearning', 'QLearning'),#右
        fn('06-16 14.06.21 tr1000it100ts10 QLearning', 'QLearning'),#左
        fn('06-16 17.52.04 tr500it200ts10 QLearning', 'QLearning'),
    ]

    '''
    for x in rfe:
        analyze(x)
    for x in msse:
        analyze(x)
    for x in qlearning:
        analyze(x)
    '''
    #analyze2(rfe + msse + qlearning)
    #analyze2(rfe[2:] + msse[1::2] + qlearning[2:3])
    #analyze2(rfe[2:])
    #analyze2(msse[1::2])
    #analyze2(qlearning[2:])
    analyze2( msse[5:])
    analyze(msse[5])
