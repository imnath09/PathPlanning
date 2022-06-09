
import matplotlib.pyplot as plt

def analyze(filename):
    with open(filename, 'r', encoding='utf-8') as file1:
        lines = file1.readlines()
        lines = [x.strip() for x in lines]
        test_rate1 = [float(x) for x in lines[0].split(',')]
        test_len1 = [float(x) for x in lines[1].split(',')]
        train_rate1 = [float(x) for x in lines[2].split(',')]
        train_len1 = [float(x) for x in lines[3].split(',')]

    x = range(1, 1 + len(test_rate1))
    plt.figure(figsize=(10, 10))
    plt.suptitle(filename)
    plt.plot(x, test_rate1, 'r', label = 'test rate')
    plt.plot(x, train_rate1, 'b', label = 'train rate')
    plt.ylabel('success rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}.rate.png'.format(filename))
    plt.close('all')

    plt.figure(figsize=(10, 10))
    plt.suptitle(filename)
    plt.plot(x, test_len1, 'r', label = 'test length')
    plt.plot(x, train_len1, 'b', label = 'train length')
    plt.ylabel('average length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}.len.png'.format(filename))
    plt.close('all')

def analyze2(f1, f2):
    with open(f1, 'r', encoding='utf-8') as file1:
        lines = file1.readlines()
        lines = [x.strip() for x in lines]
        test_rate1 = [float(x) for x in lines[0].split(',')]
        test_len1 = [float(x) for x in lines[1].split(',')]
        train_rate1 = [float(x) for x in lines[2].split(',')]
        train_len1 = [float(x) for x in lines[3].split(',')]
    with open(f2, 'r', encoding='utf-8') as file2:
        lines = file2.readlines()
        lines = [x.strip() for x in lines]
        test_rate2 = [float(x) for x in lines[0].split(',')]
        test_len2 = [float(x) for x in lines[1].split(',')]
        train_rate2 = [float(x) for x in lines[2].split(',')]
        train_len2 = [float(x) for x in lines[3].split(',')]

    # 测试成功率
    plt.figure(figsize=(10,10))
    plt.suptitle('test rate')
    plt.plot(range(len(test_rate1)), test_rate1, 'r-', label = f1)
    plt.plot(range(len(test_rate2)), test_rate2, 'b-', label = f2)
    plt.ylabel('test rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/test_rate.png')
    # 测试长度
    plt.figure(figsize=(10,10))
    plt.suptitle('test length')
    plt.plot(range(len(test_len1)), test_len1, 'r-', label = f1)
    plt.plot(range(len(test_len2)), test_len2, 'b-', label = f2)
    plt.ylabel('test length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/test_length.png')
    # 训练成功率
    plt.figure(figsize=(10,10))
    plt.suptitle('train rate')
    plt.plot(range(len(train_rate1)), train_rate1, 'r-', label = f1)
    plt.plot(range(len(train_rate2)), train_rate2, 'b-', label = f2)
    plt.ylabel('train rate')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/train_rate.png')
    # 训练长度
    plt.figure(figsize=(10,10))
    plt.suptitle('train length')
    plt.plot(range(len(train_len1)), train_len1, 'r-', label = f1)
    plt.plot(range(len(train_len2)), train_len2, 'b-', label = f2)
    plt.ylabel('train length')
    plt.xlabel('horizon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/train_length.png')
    print('finish')

if __name__ == '__main__':
    analyze('../img/06-09 17.19.50 MSSE tr200it500ts10/MSSE.txt')
    analyze('../img/06-09 17.19.50 QLearning tr200it500ts10/QLearning.txt')
    analyze2(
        '../img/06-09 17.19.50 MSSE tr200it500ts10/MSSE.txt',
        '../img/06-09 17.19.50 QLearning tr200it500ts10/QLearning.txt'
    )


