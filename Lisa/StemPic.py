from grpc import composite_channel_credentials
import matplotlib.pyplot as plt


def getVIPColor(value):
    if value < 2:
        return 0
    if value < 3:
        return 1
    return 2

fc=[
    1.269842457,1.28644567,1.349503274,1.355517106,1.502673049,1.549416168,1.576242671,
    1.70551587,1.917934975,2.046491381,2.129830103,3.356057363,2.52853E-08,1.34215E-07,
    2.00809E-07,4.88247E-07,0.166426372,0.201538948,0.229213216,0.364829198,0.442211474,0.589456539]


compound_name=[
'norophthalmic acid','lactic acid','6-methyl-3,5-heptadien-2-one','pc(15:0/15:0)','5-hydroxyindoleacetic acid','3-(4-hydroxyphenyl)-1-propanol',
'norvaline','phenylalanyl-isoleucine','loganin','galactose 1','5-methylcytidine','24,25-dihydrolanosterol','phosphate','glutaraldehyde 3',
'3-methylamino-1,2-propanediol 1','dodecanol','l-dopa 1','phloretin','3-hydroxypyridine','deoxyinosine','succinic acid','serine 2',]

LOG_FOLDCHANGE=[
0.34464952,0.36339053,0.432428477,0.43884332,0.587531142,0.631724699,0.656489663,0.770208179,0.939553808,1.033152591,1.090738351,1.746767375,
-25.23712683,-22.82894663,-22.24767231,-20.96588425,-2.587044036,-2.310869427,-2.125237867,-1.454706898,-1.177191636,-0.762542648,]

symbol=['*','**','**','*','**','**','*','***','*','*','***','**','***','***','***','***','***','**','***','*','***','***',]

VIP=[
1.67235302,1.557064739,2.442948775,2.646055716,2.794324423,2.337351252,1.871753511,3.226494075,1.516749529,1.962701713,2.98995506,
2.432254973,3.555942578,4.012584639,3.93369637,3.577524727,3.947246238,1.58424691,4.287011887,1.710888834,3.015698167,2.503207836]

COLOR = [['#C0D9D9','#4D4DFF','#0000FF'], ['#EAADEA','#FF6EC7','#FF0000']]

if __name__ == '__main__':
    #plt.rcParams['font.sans-serif'] = ['SimHei']
    #plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

    linecolors = []
    markercolors = []
    for i in range(len(symbol)):
        plt.annotate(symbol[i], xy=(LOG_FOLDCHANGE[i], i))

        if LOG_FOLDCHANGE[i] < 0:
            linecolors.append('blue')
            markercolors.append(COLOR[0][getVIPColor(VIP[i])])
        else:
            linecolors.append('red')
            markercolors.append(COLOR[1][getVIPColor(VIP[i])])

    plt.hlines(
        y=compound_name, xmin=0, xmax=LOG_FOLDCHANGE,
        color=linecolors, alpha=1, linewidth=1)

    plt.vlines(
        x=0, ymin=0, ymax=len(compound_name), color='Pink'
    )

    plt.scatter(LOG_FOLDCHANGE, compound_name, color=markercolors, s=100, alpha=0.6)

    plt.xlim(-26,26)
    plt.xlabel('log2 Fold Change')
    plt.ylabel('MS2 name')

    plt.tight_layout()

    plt.show()


