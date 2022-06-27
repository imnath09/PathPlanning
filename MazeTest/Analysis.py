
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
            datas[i],
            colors[i],
            linestyle = linestyles[i % 4],
            alpha = al,
            #label = labels[i],
            )
    plt.ylabel(title)
    plt.xlabel('horizon')
    plt.ylim(-0.05, 0.9)
    plt.grid(visible=True,axis='x')
    #plt.legend()
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
    sx = 150

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

def analyze3(files, fname):
    files = [x for x in files if x.find(fname) >= 0]
    datas = [readfile(x) for x in files] # 数量是文本数量
    #test_rate, test_len, train_rate, train_len, test_reward, train_reward = readfile(f1)

    #get_color = lambda : "#" + "%06x" % random.randint(0, 0xFFFFFF)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(len(datas))
    sx = 150

    # 测试成功率
    train_rates = [x[2][0:sx] for x in datas]
    draw('train rates', train_rates, colors, files, '../img/{}.png'.format(fname), linestyle1)

    print('finish')

if __name__ == '__main__':
    def getdir(d=''):
        if d == '':
            d = 'd:\\code\\pathplanning\\img'
        for o, d, f in os.walk(d):
            return o,d,f
    def fn(a):
        return '../img/{}/data.txt'.format(a)
    def fn1(a):
        return '../img/{}'.format(a)

    #l=getdir('d:\\code\\pathplanning\\img\\1000_100')[1]
    #l = [fn1(x) for x in l if x.find('tr200') >= 0]
    #analyze2(l)




    #for x in ssss:
    #    analyze(x)
    #analyze2([fn(x) for x in bad])

    b=[]
    for o,d,f in os.walk('d:\\code\\PathPlanning\\img\\200_150_2'):
        for ff in f:
            if ff.find('tr200')>=0:
                fp = os.path.join(o,ff)
                b.append(fp)

    analyze3(b,' RFE') #
    analyze3(b,' SE') #
    analyze3(b,' QLearning') #

    #analyze3(b,' SP2_107') # 差异不大
    analyze3(b,' SP2_1514') # 差异不大

    analyze3(b,' SP4_82_107_1514') # 差异极大（两级）
    #analyze3(b,' SP4_1814_1313_1514') # 差异极大（两级）

    #analyze3(b,' SPaSE2_82') # 差异不大
    #analyze3(b,' SPaSE2_157') # 差异不大
    #analyze3(b,' SPaSE2_107') # 差异不大
    #analyze3(b,' SPaSE2_1310') # 差异不大
    analyze3(b,' SPaSE2_1514') # 差异不大

    analyze3(b,' SPaSE4_82_107_1514') # 差异不大
    #analyze3(b,' SPaSE4_1814_1313_1514') # 差异不大

    '''analyze3(b,'nSPaSE4_82_107_1514') # 差异不大
    analyze3(b,'nSPaSE2_1514') # 差异不大
    analyze3(b,'nSP4_82_107_1514') # 差异极大（两级）
    analyze3(b,'nSP2_1514') # 差异极大（较好）
    analyze3(b,'nSE') # 差异不大
    analyze3(b,'nRFE') #
    analyze3(b,'nQLearning') #'''






