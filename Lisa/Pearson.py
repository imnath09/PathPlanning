
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

def heat(table1, table2, figname=''):
    l1 = table1.index.__len__()
    l2 = table2.index.__len__()
    l = 15 # max(l1, l2)
    fsize = 9.5
    scattersiize = 650
    fig = plt.figure(figsize=(l, l))
    #fig.set_figheight(l)
    #fig.set_figwidth(l)

    rframe = pd.DataFrame(columns=table2.index.values,dtype=float)
    pframe = pd.DataFrame(columns=table2.index.values)
    ntbl = np.full((l1, l2), 0.0)
    for i in range(l2):
        r1 = range(i, l2) if (l1 == l2) else range(l1)
        for j in r1:
            f2 = table2.iloc[i]
            f1 = table1.iloc[j]
            r, p = pearsonr(f1, f2)
            #r = round(r, 2)
            #ntbl[i][j] = r

            rrrr=''
            if p < 0.001:
                text = '{}{}'.format(rrrr, '***')
            elif p < 0.01:
                text = '{}{}'.format(rrrr, '**')
            elif p < 0.05:
                text = '{}{}'.format(rrrr, '*')
            else:
                text = str(rrrr)

            plt.annotate(text=text, xy=(i, j), ha='center', va='center', fontsize=fsize)
            #fontsize ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
            plt.hlines(
                y=j-0.5, xmin=i-0.5, xmax=i+0.5,
                color='black', linewidth=0.3
            )
            plt.vlines(
                x=i+0.5, ymin=j-0.5, ymax=j+0.5,
                color='black', linewidth=0.3
            )
            plt.scatter(
                x=i, y=j, s=abs(scattersiize * r), c=r,
                edgecolors='black',
                cmap='bwr', marker='o', vmin=-1.0, vmax=1.0,
                linewidths=0.4
            )
    #plt.scatter(x=0, y=0, s=abs(scattersiize * 1), c=r,edgecolors='black',cmap='bwr', marker='o', vmin=-1.0, vmax=1.0,linewidths=0.4)

    plt.imshow(ntbl, cmap='bwr', vmin=-1.0, vmax=1.0)
    #plt.xlim(-1, l2)
    #plt.ylim(-1, l1)

    #plt.xticks(range(l2), labels=['' for i in range(l2)], fontsize=fsize, rotation=45)
    #for i in range(l2):
    #    plt.annotate(text=table2.index.values[i],xy=(i, -0.5), rotation=45)
    plt.xticks(range(l2), labels=table2.index.values, fontsize=fsize, rotation=45, ha='right')
    
    plt.yticks(range(l1), labels=table1.index.values, fontsize=fsize)
    plt.tight_layout()
    #plt.colorbar()

    plt.savefig('{} {}.png'.format(figname, l))
    plt.show()
    plt.close()

'''def scatter(t1, t2, figname=''):
    l1 = len(t1.index)
    l2 = len(t2.index)
    l = max(l1, l2)
    fig = plt.figure()
    fig.set_figheight(l2)
    fig.set_figwidth(l2)
    plt.grid(axis='both')

    r2 = range(l2)
    for i in r2:
        r1 = range(i + 1) if (l1 == l2) else range(l1)
        for j in r1:
            f1 = t1.iloc[j]
            f2 = t2.iloc[i]
            r, p = pearsonr(f1, f2)
            r = round(r, 2)
            if p < 0.001:
                text = '{}{}'.format(r, '***')
            elif p < 0.01:
                text = '{}{}'.format(r, '**')
            elif p < 0.05:
                text = '{}{}'.format(r, '*')
            else:
                text = str(r)

            plt.scatter(
                x=i+0.5, y=j+0.5, s=abs(4000*r), c=r,
                edgecolors='black',
                cmap='bwr', marker='o', vmin=-1.0, vmax=1.0)
            plt.annotate(text=text, xy=(i+0.5, j+0.5), ha='center', va='center', fontsize='large')
            #fontsize ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
    plt.xlim(0, l2)
    plt.ylim(0, l)
    #ticks = []
    #for v in t1.index.values:
    #    ticks.append(v.rjust(50, '-'))
    plt.yticks(np.arange(l1), labels=t1.index.values)#, rotation=0)
    #ticks = []
    #for v in t2.index.values:
    #    ticks.append(v.rjust(50, '-'))
    plt.xticks(np.arange(l2), labels=t2.index.values, rotation=90)
    plt.tight_layout()
    plt.savefig(figname)
    #plt.show()'''



if __name__ == '__main__':
    n2 = 'before treatment metabolomics'
    n1 = '1'
    f1 = pd.read_excel(n1+'.xlsx', index_col='id')#, dtype='float')
    f2 = pd.read_excel(n2+'.xlsx', index_col='id',)# dtype='float')
    f3 = pd.concat([f1, f2])

    heat(f1, f2, n1 + '-' + n2)
    #heat(f3, f3, 'hzunei')
    '''
    c1 = f1.columns.values
    print(c1)
    c2 = f2.columns.values
    print(c2)
    if (c1 == c2).all():
        print('ok')
    else:
        print('no')
    print(f3.index.values.__len__(), f3.index.values)
    '''
    #scatter(f1, f2, 'szujian.png')
    #scatter(f3, f3, 'szunei.png')





    print('finish')

#'3-methoxy-4-hydroxyphenylethyleneglycol sulfate'