
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fsize = (4, 3)
linestyle1= ['-', '--', '-.', ':',]
linestyles = ['-']*4
SOME_COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'olive': '#bcbd22',
    'cyan': '#17becf',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
}
#scolors = ['b', 'g', 'r', 'y', 'c', 'm', 'k', ]
colorslist = list(SOME_COLORS.values())

def get_file_dict(d):
    fileDicts = dict()
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.find('.txt') > 0:
                ff = f.split(' ')[0].split('_')[0]
                if ff in fileDicts:
                    fileDicts[ff].append(os.path.join(root, f))
                else:
                    fileDicts[ff] = [os.path.join(root, f)]
    return fileDicts

def read_file(fname):
    with open(fname, 'r', encoding='utf-8') as file1:
        lines = [x.strip() for x in file1.readlines()]
        test_rate, train_rate, test_rewards, train_rewards, = (
            [float(x) for x in lines[0].split(',')],
            [float(x) for x in lines[1].split(',')],
            [float(x) for x in lines[2].split(',')],
            [float(x) for x in lines[3].split(',')],
        )
        wall_e, wall_i, wall_c = [float(x) for x in lines[4].split(',')][0: 3]
        episode_e, episode_i, episode_c = [int(x) for x in lines[5].split(',')][0:3]
        #episode_c *= 200

    return [[test_rate, train_rate, test_rewards, train_rewards],
            [wall_e, wall_i, wall_c],
            [episode_e, episode_i, episode_c]]

def get_data(files):
    [[test_rates, train_rates, test_rewards, train_rewards],
     [wall_es, wall_is, wall_cs],
     [episode_es, episode_is, episode_cs]] = [[[], [], [], []],
                                              [[], [], []],
                                              [[], [], []]]
    datas = [read_file(x) for x in files] # 数量是文本数量
    for x in datas:
        [[test_rate, train_rate, test_reward, train_reward],
         [wall_e, wall_i, wall_c],
         [episode_e, episode_i, episode_c]] = x
        test_rates.append(test_rate)
        train_rates.append(train_rate)
        test_rewards.append(test_reward)
        train_rewards.append(train_reward)
        wall_es.append(wall_e)
        wall_is.append(wall_i)
        wall_cs.append(wall_c)
        episode_es.append(episode_e)
        episode_is.append(episode_i)
        episode_cs.append(episode_c)
    '''a=[
        [test_rates, train_rates, test_rewards, train_rewards],
        [wall_es, wall_is, wall_cs],
        [episode_es, episode_is, episode_cs]
    ]
    for x in a:
        for y in x:
            print(y)'''
    return [[test_rates, train_rates, test_rewards, train_rewards],
            [wall_es, wall_is, wall_cs],
            [episode_es, episode_is, episode_cs]]

def get_interval(v):
    avr = np.mean(v)
    se = stats.sem(v)
    z = 1.96 * se
    return avr, z#np.std(v)

def confidence_Interval(data):
    '''把同一个实验的多个实例画进同一个图'''
    #files = [x for x in files if x.find(exp_name) >= 0]
    #datas = [readfile(x) for x in files]
    #train_rates = [x[2] for x in datas] # 成功率
    m, y1, y2, td = [], [], [], []
    xr = range(len(data[0]))
    for i in xr:
        v = [x[i] for x in data]
        avr,z=get_interval(v)
        m.append(avr)
        y2.append(avr + z)
        y1.append(avr - z)
        td.append(np.std(v))
    return xr, m, y1, y2, td

def draw_curves(title, datas, colors, labels, savename, linestyles = linestyle1):
    '''把一堆实验曲线画在一个图里'''
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

def DrawEveryExperimentCurve(fileDicts : dict):
    '''比较某一【类】实验(比如SPARM4_**)的六个数据'''
    for exp_name, files in fileDicts.items():
        #get_color = lambda : "#" + "%06x" % random.randint(0, 0xFFFFFF)
        get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
        colors = get_colors(len(files))
        [[test_rates, train_rates, test_rewards, train_rewards],
         [wall_es, wall_is, wall_cs],
         [episode_es, episode_is, episode_cs]] = get_data(files)
        # 测试成功率
        draw_curves('test_rate', test_rates, colors, files, '../img/{}_test_rate.png'.format(exp_name))
        # 训练成功率
        draw_curves('train_rate', train_rates, colors, files, '../img/{}_train_rate.png'.format(exp_name))
        # 测试奖励
        draw_curves('test_rewards', test_rewards, colors, files, '../img/{}_test_rewards.png'.format(exp_name))
        # 训练奖励
        draw_curves('train_rewards', train_rewards, colors, files, '../img/{}_train_rewards.png'.format(exp_name))
        # 平均测试累积奖励
        #draw('test reward', wallclock, colors, files, '../img/{}_wallclock.png'.format(exp_name))
        # 平均训练累积奖励
        #draw('train reward', episodes, colors, files, '../img/{}_episodes.png'.format(exp_name))
        print('curve finish')

def DrawConfidences(fileDicts : dict, index, title):
    '''3volatility and 3confidences'''
    i = 0
    plt.figure(figsize=(8, 6))
    #plt.suptitle(title)
    for exp_name in fileDicts:
        files = fileDicts[exp_name]
        '''[
        [test_rates, train_rates, test_rewards, train_rewards],
        [wall_es, wall_is, wall_cs],
        [episode_es, episode_is, episode_cs]
        ] = get_data(files)'''
        fd = get_data(files)
        data = fd[0][index] # 其中一个：[test_rates, train_rates, test_rewards, train_rewards]
        eps = fd[2] # [episode_es, episode_is, episode_cs]
        me = int(np.mean(np.array(eps[0]) + np.array(eps[1])) / 200)
        #print(exp_name, me)
        xr, m, y1, y2, td = confidence_Interval(data)
        gap = 1*(i + 1)
        plt.fill_between(range(me, me + len(m)), np.array(td) + gap, [gap] * len(xr), facecolor=colorslist[i], alpha=0.3)#color='red', 
        plt.plot(range(me, me + len(m)), np.array(td) + gap, color=colorslist[i])
        plt.plot(range(me, me + len(m)), [gap] * len(xr), color=colorslist[i])#, label=exp_name)
        i += 1
    plt.yticks([])
    #plt.legend()
    plt.tight_layout()
    plt.savefig('../img/3volatility {}.png'.format(title))
    plt.close()

    i = 0
    plt.figure(figsize=(8, 6))
    #plt.suptitle(title)
    for exp_name in fileDicts:
        files = fileDicts[exp_name]
        '''[
        [test_rates, train_rates, test_rewards, train_rewards],
        [wall_es, wall_is, wall_cs],
        [episode_es, episode_is, episode_cs]
        ] = get_data(files)'''
        fd = get_data(files)
        data = fd[0][index] # 其中一个：[test_rates, train_rates, test_rewards, train_rewards]
        eps = fd[2] # [episode_es, episode_is, episode_cs]
        me = int(np.mean(np.array(eps[0]) + np.array(eps[1])) / 200)
        #print(exp_name, me)
        xr, m, y1, y2, td = confidence_Interval(data)
        plt.fill_between(range(me, me + len(m)), y1, y2, facecolor=colorslist[i], alpha=0.3)#color='red', 
        plt.plot(range(me, me + len(m)), m, color=colorslist[i], label=exp_name)
        i += 1
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/3confidences {}.png'.format(title))
    plt.close()

    #print('confidence finish')

def DrawEveryExperimentBar(fileDicts : dict):
    for exp_name, files in fileDicts.items():
        #DrawBar(exp_name, files)
        [[test_rates, train_rates, test_rewards, train_rewards],
         [wall_es, wall_is, wall_cs],
         [episode_es, episode_is, episode_cs]] = get_data(files)
        x = files
        d=[np.array(episode_es)+np.array(episode_is)+np.array(episode_cs),
           np.array(episode_es)+np.array(episode_is),
           episode_es]
        e=[None,None,None]
        draw_bars(x,heights=d,yerrs=e,labels=['cvg','merge','expand'],title=exp_name,iname='../img/4{}episode.png'.format(exp_name))#,ylim=(0, 65000))
        d=[np.array(wall_es)+np.array(wall_is)+np.array(wall_cs),
           np.array(wall_es)+np.array(wall_is),
           wall_es]
        draw_bars(x,heights=d,yerrs=e,labels=['cvg','merge','expand'],title=exp_name,iname='../img/4{}wallclock.png'.format(exp_name))#,ylim=(0,3000))
        print('bar finish')

def DrawMainData(fileDicts : dict, index, title):
    xticks=[]
    mean_e,err_e,expand=[],[],[]#expand
    mean_i,err_i,inner=[],[],[]#inner
    mean_c,err_c,cvg=[],[],[]#convergence
    mean_m,err_m,merge=[],[],[]#merge
    mean_t,err_t,total=[],[],[]#total
    for exp_name, files in fileDicts.items():
        [expands, inners, convergences] = get_data(files)[index][0: 3] # [wall_es, wall_is, wall_cs]或者[episode_es, episode_is, episode_cs]
        edata=get_interval(expands)
        mean_e.append(edata[0])
        err_e.append(edata[1])
        expand.append(expands)

        edata=get_interval(inners)
        mean_i.append(edata[0])
        err_i.append(edata[1])
        inner.append(inners)

        edata=get_interval(convergences)
        mean_c.append(edata[0])
        err_c.append(edata[1])
        cvg.append(convergences)

        edata=np.array(expands) + np.array(inners)
        mean_m.append(round(np.mean(edata), 2))
        err_m.append(round(np.std(edata), 2))
        merge.append(edata)

        edata=np.array(expands) + np.array(inners) + np.array(convergences)
        mean_t.append(round(np.mean(edata), 2))
        err_t.append(round(np.std(edata), 2))
        total.append(edata)

        xticks.append(exp_name)
    print(xticks)
    print('merge', title, mean_m, err_m)
    print('total',title,mean_t,err_t)
    draw_a_bar(xticks,mean_m,err_m,xticks,'../img/1errorbar merge {}.png'.format(title))
    draw_a_bar(xticks,mean_t,err_t,xticks,'../img/1errorbar total {}.png'.format(title))
    #draw_a_bar(xticks,[mean_t,'mean_m'],[err_t,'err_m'],['total','merge'],title,'../img/1errorbar {} total.png'.format(title))
    #draw_box(xticks,merge,'../img/2box merge {}.png'.format(title))
    #draw_box(xticks,total,'../img/2box total {}.png'.format(title))
    draw_scatter(xticks, merge, mean_m, '../img/2scatter merge {}.png'.format(title))
    draw_scatter(xticks, total, mean_t, '../img/2scatter total {}.png'.format(title))

def draw_scatter(xticks, data, means, fname):
    plt.figure(figsize=fsize)
    r = range(len(xticks))
    for i in r:
        plt.scatter(
            x=[i] * len(data[i]), y=data[i], marker='+', c = colorslist[i]
            #s=abs(scattersiize * r), c=r, edgecolors='black', cmap='bwr', vmin=-1.0, vmax=1.0, linewidths=0.4
            )
        plt.scatter(
            x=i, y=means[i], marker='o', c='r', s=40
        )
    #plt.xticks(rotation=45, ticks=r, labels=xticks, ha='right')
    plt.xticks(ticks=[])#,color='w')
    #plt.ylim([0, 1800])
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def draw_box(labels,x,fname):
    plt.figure(figsize=fsize)
    plt.boxplot(
        x=x,
        notch=False,
        showmeans=True,
        meanline=True,
        patch_artist=False,
        widths=0.1,
        #boxprops={'color':'blue',},#设置箱体属性，填充色和边框色
        flierprops={'marker':'+','markerfacecolor':'#9999ff'},#设置异常值属性，点的形状、填充颜色和边框色
        meanprops={'linestyle':'dotted','color':'red'},#设置均值点的属性，点的颜色和形状
        medianprops={'linestyle':'-','color':'orange'},#设置中位数线的属性，线的类型和颜色
        labels=labels,#[0::1],
        whis=0.8,
        )
    plt.xticks(rotation = 45, horizontalalignment = 'right')
    #plt.grid(visible=True,axis='y')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def draw_bars(x, heights,yerrs,labels,title,iname):
    plt.figure(figsize=(12, 9))
    plt.suptitle(title)
    for i in range(len(heights)):
        plt.bar(x, height=heights[i],yerr=yerrs[i],color=colorslist,width=0.5,label=labels[i])
    plt.xticks(ha='right',rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(iname)
    plt.close()

def draw_a_bar(x, heights,yerrs,labels,iname):
    plt.figure(figsize=fsize)
    plt.bar(x, height=heights,yerr=yerrs,color=colorslist,width=0.5,label=labels)
    #plt.bar([0.5, 1.5, 2.5, 3.5, 4.5, 2.8, 3.8, 4.8], height=heights,yerr=yerrs,color=colorslist,width=0.5,label=labels)
    #plt.bar_label(pl, label_type='edge', labels = labels)
    plt.xticks(ticks=[])#ha='right',rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(iname)
    plt.close()

if __name__ == '__main__':
    dir = '../img/'
    fileDict = get_file_dict(dir)
    DrawMainData(fileDict,1,'wallclock')#画挂钟时间置信区间
    DrawMainData(fileDict,2,'episode')#画幕置信区间
    DrawConfidences(fileDict, 3, 'train_rewards')#画奖励曲线
    #DrawConfidences(fileDict, 1, 'train_rate')#画奖励曲线
    #DrawEveryExperimentBar(fileDict)
    #DrawEveryExperimentCurve(fileDict)#画实验曲线

    '''en='SPaRM4_1514_158_125'
    fs=fileDict[en]
    #print(en,fs)
    d=get_data(fs)
    print(d[1])
    #DrawEveryExperimentBar(en,fs)'''

    '''
    a = [x for x in b if x.find(t) >= 0]
    '''






