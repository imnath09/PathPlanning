
import random
import matplotlib.pyplot as plt

fsize = (18, 9)

def analyze(filename):
    with open(filename, 'r', encoding='utf-8') as file1:
        lines = file1.readlines()
        lines = [x.strip() for x in lines]
        test_rate1 = [float(x) for x in lines[0].split(',')]
        test_len1 = [float(x) for x in lines[1].split(',')]
        train_rate1 = [float(x) for x in lines[2].split(',')]
        train_len1 = [float(x) for x in lines[3].split(',')]

    x = range(1, 1 + len(test_rate1))
    plt.figure(figsize=fsize)
    plt.suptitle(filename)
    plt.plot(x, test_rate1, 'r', label = 'test rate')
    plt.plot(x, train_rate1, 'b', label = 'train rate')
    plt.ylabel('success rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}.rate.png'.format(filename))
    plt.close('all')

    plt.figure(figsize=fsize)
    plt.suptitle(filename)
    plt.plot(x, test_len1, 'r', label = 'test length')
    plt.plot(x, train_len1, 'b', label = 'train length')
    plt.ylabel('average length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}.len.png'.format(filename))
    plt.close('all')

def readfile(fname):
    if fname == '':
        return [None, None, None, None]
    with open(fname, 'r', encoding='utf-8') as file1:
        lines = file1.readlines()
        lines = [x.strip() for x in lines]
        test_rate1 = [float(x) for x in lines[0].split(',')]
        test_len1 = [float(x) for x in lines[1].split(',')]
        train_rate1 = [float(x) for x in lines[2].split(',')]
        train_len1 = [float(x) for x in lines[3].split(',')]
    return test_rate1, test_len1, train_rate1, train_len1

def analyze2(files):
    datas = [readfile(x) for x in files]
    #test_rate1, test_len1, train_rate1, train_len1 = readfile(f1)
    test_rates = [x[0][0:50] for x in datas]
    test_lens = [x[1][0:50] for x in datas]
    train_rates = [x[2][0:50] for x in datas]
    train_lens = [x[3][0:50] for x in datas]

    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(len(files))
    linestyles = ['-', '--', '-.', ':',]
    al = 0.7
    #get_color = lambda : "#" + "%06x" % random.randint(0, 0xFFFFFF)

    # 测试成功率
    plt.figure(figsize=fsize)
    plt.suptitle('test rate')
    for i in range(len(test_rates)):
        plt.plot(
            test_rates[i], colors[i], linestyle = linestyles[i % 4],
            label = files[i], alpha = al)
    plt.ylabel('test rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/test_rate.png')
    plt.close()
    # 测试长度
    plt.figure(figsize=fsize)
    plt.suptitle('test length')
    for i in range(len(test_lens)):
        plt.plot(
            test_lens[i], colors[i], linestyle = linestyles[i % 4],
            label = files[i], alpha = al)
    plt.ylabel('test length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/test_length.png')
    plt.close()
    # 训练成功率
    plt.figure(figsize=fsize)
    plt.suptitle('train rate')
    for i in range(len(train_rates)):
        plt.plot(
            train_rates[i], colors[i], linestyle = linestyles[i % 4],
            label = files[i], alpha = al)
    plt.ylabel('train rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/train_rate.png')
    plt.close()
    # 训练长度
    plt.figure(figsize=fsize)
    plt.suptitle('train length')
    for i in range(len(train_lens)):
        plt.plot(
            train_lens[i], colors[i], linestyle = linestyles[i % 4],
            label = files[i], alpha = al)
    plt.ylabel('train length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/train_length.png')
    plt.close()
    print('finish')

if __name__ == '__main__':
    rfe = [
        '../img/{}/{}.txt'.format('06-10 20.50.10 tr1000it100ts10 RFE', 'RFE'),
        '../img/{}/{}.txt'.format('06-15 19.07.47 tr1000it100ts10 RFE', 'RFE'),
        '../img/{}/{}.txt'.format('06-15 22.58.01 tr1000it100ts10 RFE', 'RFE'),#左
        '../img/{}/{}.txt'.format('06-16 14.06.52 tr1000it100ts10 RFE', 'RFE'),#右
    ]
    msse = [
        '../img/{}/{}.txt'.format('06-09 20.29.40 tr1000it100ts10 MSSE', 'MSSE'),
        '../img/{}/{}.txt'.format('06-15 20.17.54 tr1000it100ts10 MSSE', 'MSSE'),#右
        '../img/{}/{}.txt'.format('06-15 22.57.38 tr1000it100ts10 MSSE', 'MSSE'),
        '../img/{}/{}.txt'.format('06-16 14.06.34 tr1000it100ts10 MSSE', 'MSSE'),#左
    ]
    qlearning = [
        '../img/{}/{}.txt'.format('06-09 20.29.34 tr1000it100ts10 QLearning', 'QLearning'),
        '../img/{}/{}.txt'.format('06-15 20.17.43 tr1000it100ts10 QLearning', 'QLearning'),
        '../img/{}/{}.txt'.format('06-15 22.56.54 tr1000it100ts10 QLearning', 'QLearning'),#右
        '../img/{}/{}.txt'.format('06-16 14.06.21 tr1000it100ts10 QLearning', 'QLearning'),#左
    ]

    '''for x in rfe:
        analyze(x)
    for x in msse:
        analyze(x)
    for x in qlearning:
        analyze(x)'''
    #analyze2(rfe + msse + qlearning)
    analyze2(rfe[2:] + msse[1::2] + qlearning[2:3])
    #analyze2(rfe[2:])
    #analyze2(msse[1::2])
    #analyze2(qlearning[2:])
    # '-', '--', '-.', ':', 
