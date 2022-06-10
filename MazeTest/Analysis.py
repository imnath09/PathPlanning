
import matplotlib.pyplot as plt

fsize = (12, 9)

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

def analyze2(f1, f2, f3=''):
    test_rate1, test_len1, train_rate1, train_len1 = readfile(f1)
    test_rate2, test_len2, train_rate2, train_len2 = readfile(f2)
    test_rate3, test_len3, train_rate3, train_len3 = readfile(f3)

    # 测试成功率
    plt.figure(figsize=fsize)
    plt.suptitle('test rate')
    plt.plot(range(len(test_rate1)), test_rate1, 'r-', label = f1)
    plt.plot(range(len(test_rate2)), test_rate2, 'b-', label = f2)
    if not f3=='':
        plt.plot(range(len(test_rate3)), test_rate3, 'g-', label = f3)
    plt.ylabel('test rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/test_rate.png')
    # 测试长度
    plt.figure(figsize=fsize)
    plt.suptitle('test length')
    plt.plot(range(len(test_len1)), test_len1, 'r-', label = f1)
    plt.plot(range(len(test_len2)), test_len2, 'b-', label = f2)
    if not f3=='':
        plt.plot(range(len(test_len3)), test_len3, 'g-', label = f3)
    plt.ylabel('test length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/test_length.png')
    # 训练成功率
    plt.figure(figsize=fsize)
    plt.suptitle('train rate')
    plt.plot(range(len(train_rate1)), train_rate1, 'r-', label = f1)
    plt.plot(range(len(train_rate2)), train_rate2, 'b-', label = f2)
    if not f3=='':
        plt.plot(range(len(train_rate3)), train_rate3, 'g-', label = f3)
    plt.ylabel('train rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/train_rate.png')
    # 训练长度
    plt.figure(figsize=fsize)
    plt.suptitle('train length')
    plt.plot(range(len(train_len1)), train_len1, 'r-', label = f1)
    plt.plot(range(len(train_len2)), train_len2, 'b-', label = f2)
    if not f3=='':
        plt.plot(range(len(train_len3)), train_len3, 'g-', label = f3)
    plt.ylabel('train length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/train_length.png')
    print('finish')

if __name__ == '__main__':
    d1, t1 = '06-09 20.29.34 tr1000it100ts10\\06-09 20.29.40 tr1000it100ts10 MSSE', 'MSSE'
    d2, t2 = '06-09 20.29.34 tr1000it100ts10\\06-10 20.50.10 tr1000it100ts10 RFE', 'RFE'
    d3, t3 = '06-09 20.29.34 tr1000it100ts10\\06-09 20.29.34 tr1000it100ts10 QLearning', 'QLearning'
    ff1 = '../img/{}/{}.txt'.format(d1, t1)
    ff2 = '../img/{}/{}.txt'.format(d2, t2)
    ff3 = '../img/{}/{}.txt'.format(d3, t3)
    #analyze(ff1)
    #analyze(ff2)
    analyze2(ff1,  ff3)


