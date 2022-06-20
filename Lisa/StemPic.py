import matplotlib.pyplot as plt

from data2 import *

def getVIPColor(value):
    if value < 2:
        return 0
    if value < 3:
        return 1
    return 2

COLOR = [['#C0D9D9','#4D4DFF','#0000FF'], ['#EAADEA','#FF6EC7','#FF0000']]

if __name__ == '__main__':
    #plt.rcParams['font.sans-serif'] = ['SimHei']
    #plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

    LENGTH = len(VIP)
    if (not len(compound_name) == LENGTH
     or not len(LOG_FOLDCHANGE) == LENGTH
     or not len(symbol) == LENGTH):
        print('长度有问题')

    linecolors = []
    markercolors = []
    plt.figure(figsize=(14, 10))
    for i in range(LENGTH):
        #plt.annotate(symbol[i], xy=(LOG_FOLDCHANGE[i], i - 0.35))

        if LOG_FOLDCHANGE[i] < 0:
            linecolors.append('blue')
            markercolors.append(COLOR[0][getVIPColor(VIP[i])])
            plt.annotate(
                symbol[i], xy=(LOG_FOLDCHANGE[i], i),
                xytext=(LOG_FOLDCHANGE[i] - 0.5, i - 0.25))
        else:
            linecolors.append('red')
            markercolors.append(COLOR[1][getVIPColor(VIP[i])])
            plt.annotate(
                symbol[i], xy=(LOG_FOLDCHANGE[i], i),
                xytext=(LOG_FOLDCHANGE[i] + 0.3, i - 0.25))

    plt.hlines(
        y=compound_name, xmin=0, xmax=LOG_FOLDCHANGE,
        color=linecolors, alpha=1, linewidth=4)

    plt.vlines(
        x=0, ymin=-1, ymax=LENGTH, color='Gray'
    )

    plt.scatter(LOG_FOLDCHANGE, compound_name, color=markercolors, s=240, alpha=0.6)

    m1 = max(LOG_FOLDCHANGE)
    print(m1)
    m2 = min(LOG_FOLDCHANGE)
    print(m2)
    m3 = max(abs(m1), abs(m2))
    print(m3)

    plt.xlim(-m3 - 1, m3 + 1)
    plt.ylim(-1, LENGTH)
    s=20
    plt.xlabel('log2 Fold Change', size = s)
    plt.ylabel('MS2 name', size = s)
    plt.yticks(fontproperties='Arial', size=s)#设置大小及加粗
    plt.xticks(fontproperties='Arial', size=s)

    plt.tight_layout()

    plt.savefig(filename)
    plt.show()
    plt.close()


