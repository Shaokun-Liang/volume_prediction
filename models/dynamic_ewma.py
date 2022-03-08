import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  os

def alpha(volatility):
    # ewma中衰减系数alpha的确定
    lower_bond = 0.1
    upper_bond = 0.8
    coeff = 0.8
    a = max(lower_bond, min(1 - coeff * volatility, upper_bond))
    return  a


def ewma_prediction(x_ti, alpha, j):
    # volume=df1.loc[data1['bin_volume']!=0,'bin_volume']
    ewmas = []

    for i in range(len(x_ti)):
        if i < j:
            ewmas.append(x_ti[i])  # ewmas.append(xti[i])
        else:
            ewma = 0

            for k in range(1, j + 1):
                ewma = alpha * ewma + (1 - alpha) * x_ti[i - k]
            ewmas.append(ewma)
    return ewmas


def dynamic_ewma_prediction(x_ti, j=4):
    # volume=df1.loc[data1['bin_volume']!=0,'bin_volume']
    ewmas = []

    for i in range(len(x_ti)):
        if i < j:
            ewmas.append(x_ti[i])  # ewmas.append(xti[i])
        else:
            ewma = 0

            for k in range(1, j + 1):
                windows = x_ti[i - j - 1:i - 1]
                windows = (windows - np.mean(windows)) / np.std(windows, ddof=1)
                volatility = np.std(windows)
                alpha1 = alpha(volatility)
                ewma = alpha1 * ewma + (1 - alpha1) * x_ti[i - k]
            ewmas.append(ewma)
    return ewmas

def bin_mape(x_ti, x_ti_hat):
    if len(x_ti)!=len(x_ti_hat):
        print("lenth error!")
        print((len(x_ti)))
        print((len(x_ti_hat)))
        return None
    else:
        mape = []
        for i in range(len(x_ti)):
            if x_ti[i]<1e-5:
                mape_bin = np.nan
                mape.append(mape_bin)
            else:
                mape_bin = abs((x_ti[i]- x_ti_hat[i]) / x_ti[i])*100
                mape.append(mape_bin)

    return mape


home='/home/sliang/JupyterLab/input_data/high_volume_stock/'
filenames=os.listdir(home)
#filenames=['688388_XSHG_25_daily.csv','688128_XSHG_25_daily.csv']
mape_all=pd.DataFrame()
for file in filenames:
    index = file[:6]
    df=pd.read_csv(home+file)
    bin_vol=df['bin_volume']
    j=3
    pred=dynamic_ewma_prediction(bin_vol,j)
    mape= bin_mape(bin_vol,pred) #mape is a list

    pred = pd.DataFrame({index + '_pred': pred})
    true = pd.DataFrame({index +'_ture': bin_vol})
    pred.to_csv('/home/sliang/JupyterLab/new_result/dynamic_ewma/' + index + '_dynamic_ewma_pred.csv', index=False)
    true.to_csv('/home/sliang/JupyterLab/new_result/dynamic_ewma/' + index + '_dynamic_ewma_true.csv', index=False)

    mape_df = pd.DataFrame({index: mape}) #给mape命名
    mape_all=pd.concat([mape_all,mape_df],axis=1)
mape_all.to_csv('/home/sliang/JupyterLab/new_result/dynamic_ewma/dyEWMA_mape_%d_windows.csv'%j,index=False)
xtick=mape_all.columns
ax=sns.boxplot(data=mape_all.iloc[:,:],width=0.8,showmeans=False,showfliers=False)
ax.set_xticklabels(labels = xtick,rotation = 45,fontsize = 10)
plt.xlabel('ticker') # 放大横轴名称
plt.ylabel('(%)')
plt.title('dynamic_EWMA_error_windows=%d'%j)
fig=ax.get_figure()
plt.show()
plt.tight_layout()
fig.savefig('/home/sliang/JupyterLab/result/dynamic_ewma/dynamic_ewma_boxplot_%dwindows.jpg'%j)





