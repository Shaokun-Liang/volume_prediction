import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

#1.读取股票对应的profile行，2.全部split取第三个 3.最后每只股票得到48个bin的profile
#4.两两合并成为24个bin的profile 5.整理成母单包含所有bin，母单包含两个bin的预测形式6.预测未来一个月即可
#在写导入文件的时候，一定不要随心所欲，要先看一下dataframe的结构

def read_profile(index, profile_csv):
    df = profile_csv.loc[profile_csv.iloc[:, 0] == index, 11:]
    df = df.T
    df = df.reset_index(drop=True)
    df.columns = [0]  # df.T转置之后是dataframe不是series，df.colmuns讲df转为dataframe，df[0]是第一列，而非第一行

    for i in range(48):
        df.iloc[i, :] = str(df.iloc[i, :]).split('|')[2]
        df.iloc[i, 0] = float(df.iloc[i, 0])

    df1 = df[0]
    df2 = [5.000000000000] * 48
    df2 = pd.DataFrame(df2)
    df2 = df2[0]

    bin_profile = []

    # for j in range(24):
    # bin_profile[j]=df1.iloc[2*j:2*j+2].sum(axis=0)
    # bin_profile=pd.Series(bin_profile)

    bintime = (4 * 60 - 3) / 24
    value_timeweighted = 0
    i = 0
    while i <= 47:
        if bintime - df2[i] > 0:
            bintime = bintime - df2[i]
            value_timeweighted = value_timeweighted + (df2[i] / 5) * df1[i]
            i = i + 1
        elif bintime - df2[i] < 0:
            value_timeweighted = value_timeweighted + (bintime / 5) * df1[i]
            df2[i] = 5 - bintime
            bin_profile.append(value_timeweighted)
            bintime = (4 * 60 - 3) / 24
            value_timeweighted = 0
        else:
            value_timeweighted = value_timeweighted + (df2[i] / 5 * df1[i])
            bin_profile.append(value_timeweighted)
            bintime = (4 * 60 - 3) / 24
            value_timeweighted = 0
            i = i + 1
    return bin_profile


def read_date(date_str):
    date_str=str(date_str)

    year=date_str.split('-')[0]
    month=date_str.split('-')[1]
    day=date_str.split('-')[2]
    date=datetime.date(int(year),int(month),int(day))
    return date

def read_pred_month(df):
    start_t=datetime.date(2021,6,1)
    end_t=datetime.date(2021,6,30)
    df.loc[:,'date']=df.loc[:,'date'].map(read_date)
    df_pred=df.loc[(df['date'] <= end_t) &(start_t<=df['date'])].reset_index(drop=True)
    return  df_pred




def ESV_prediction_gena(bin_volume,PF,binnum,big_order=1):
    total_t = len(bin_volume) // binnum

    total_j = binnum
    prediction = []
    PF=pd.Series(PF)

    for i in range(len(bin_volume)):
        t=i//binnum
        j=i%binnum

        if j<2:
            continue
        else:
            #m = (j+1) % big_order  #母单中，第m个子单
            m=1
            v0=PF[j-1-m:j-1].sum() ### PF index[0,24] 而j最大是24超出PF范围，所以对PF索引整体-1与daily对应
            v1=PF.cumsum()[j-2]
            v2=PF[j-1]
            v3=PF[j-1-m:j-1-m+big_order].sum()####

            dailybin_vol = bin_volume[t * binnum:(t + 1) * binnum].reset_index(drop=True)
            Rv=dailybin_vol[j-1-m:j-1].sum()  #####
            Market_v = dailybin_vol.cumsum()[j - 1]
            if v0==0:
                ESV=Market_v/v1 *v2
            else:
                ESV=(1-(v0/v3))*(Market_v/v1 *v2)+(v0/v3)*(Rv/v0*v2)


        prediction.append(ESV)
    return prediction






def ESV_prediction(index,bin_volume,binnum,lag_t):

    total_t=len(bin_volume)//binnum

    total_j=binnum
    prediction=[]

    for t in range (lag_t,total_t):
        for j in range(total_j):
            if j==0:
                continue
            else:

                dailybin_vol=bin_volume[t * binnum :(t+1)*binnum].reset_index(drop=True)

                Market_v =dailybin_vol.cumsum()[j-1]

                histrical_df =bin_volume[(t-lag_t)* binnum:t*binnum].reset_index(drop=True) #姑且认为历史数据是前1天的数据
                cum_profiles=0
                for i in range(lag_t):
                    hisrical_daily=histrical_df[i*binnum:(i+1)*binnum].reset_index(drop=True)
                    cum_profile=hisrical_daily.cumsum()[j-1]
                    cum_profiles += cum_profile
                v1=cum_profiles/lag_t  #前21天累计成交量比例的平均


                hisrical_bin=histrical_df[j::25]
                v2=hisrical_bin.mean()  #前21天同bin成交量的比例的平均

                ESV= Market_v/v1 *v2
            prediction.append(ESV)

    return  prediction

def extract_xti(df1,binnum):

    x_ti1 = df1.loc[df1['bin_num']!=0, 'bin_profile'].reset_index(drop=True)
    x_ti2 = x_ti1[21*(binnum-1):].reset_index(drop=True)  # loc[21*24:,'bin_volume'] 与loc[21*24:,bin_volume] 结果数据类型不同
    return x_ti2

def cal_mape(true, pred):
    #分母为0，长度对不上
    if len(true)!=len(pred):
        print("lenth error!")
        print((len(true)))
        print((len(pred)))
        return None
    else:
        n=len(true)
        mape=sum(np.abs((true -pred) / true)) / n *100
        return mape

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


def line_plot(x_ti, x_ti_hats,index):
    t=10
    x_ti_hats = pd.Series(x_ti_hats)
    x_ti = pd.Series(x_ti)
    df=pd.concat([x_ti,x_ti_hats],axis=1)
    df.columns=['Ture','pred']
    ax1=sns.lineplot(data=x_ti_hats[:24 * t], linewidth=1.5, alpha=0.8)
    ax1=sns.lineplot(data=x_ti[:24 * t], linewidth=1.5, alpha=0.8)
    mape = cal_mape(x_ti, x_ti_hats)

    ax1.text(x=120, y=x_ti[0:24 * t].max(), s='mape= %f' % mape)

    figure=ax1.get_figure()
    figure.savefig('/home/sliang/JupyterLab/result/system/'+index+'system_mape.jpg',dpi=800) #设置字体大小与格式
    return 0

def box_plot(data):

    ax=sns.boxplot(data=data.iloc[:,:])
    fig=ax.get_figure()
    fig.savefig('/home/sliang/JupyterLab/result/system/system_mape.jpg',dpi=800)

    return 0


binnum=24+1
home='/home/sliang/JupyterLab/input_data/money/'
filenames=os.listdir(home)
#filenames=['688388_XSHG_25_daily.csv','688128_XSHG_25_daily.csv']
mape_all=pd.DataFrame()
profile_csv=pd.read_csv('/home/sliang/Profile.csv',header=None)

for file in filenames:
    index=file[:6]
    press=file[7:11]
    profile=read_profile(index+'.'+press, profile_csv)
    df=pd.read_csv(home+file)
    df_need=read_pred_month(df)
    bin_money=df_need['bin_money']
    big_order =2

    df_pred=ESV_prediction_gena(bin_money, profile, binnum, big_order)
    df_true = df_need.loc[(df_need['bin_num'] != 0)&(df_need['bin_num'] != 1), 'bin_money'].reset_index(drop=True)

    mape = bin_mape(df_true, df_pred)
    mape_df = pd.DataFrame({index: mape})

    # list1=sorted(mape)
    # quantile =list1[int(len(mape)*0.95)]# 计算volume95%分位数
    # mape_df = mape_df.loc[mape_df[index] <= quantile].reset_index(drop= True)
    mape_all=pd.concat([mape_all,mape_df],axis=1)

    #line_plot(original_vol,predict_vol, index)
mape_all.to_csv('/home/sliang/JupyterLab/result/system/system_mape_%d_row.csv'%big_order)
xtick=mape_all.columns
ax=sns.boxplot(data=mape_all.iloc[:,:],width=0.8,showmeans=True,showfliers=False)
ax.set_xticklabels(labels = xtick,rotation = 45,fontsize = 10)
plt.xlabel('ticker') # 放大横轴名称
plt.ylabel('(%)')
#plt.ylim(0,100)
plt.title('system_error_parentlist=%d'%big_order)
fig=ax.get_figure()
plt.show()
plt.tight_layout()
fig.savefig('/home/sliang/JupyterLab/result/system/system_boxplot_%d_row.jpg'%big_order)




# binnum=24+1
# home='/home/sliang/JupyterLab/input_data/money/'
# #filenames=os.listdir(home)
# filenames=['688388_XSHG_25_daily.csv','688128_XSHG_25_daily.csv']
# mape_all=pd.DataFrame()
# for file in filenames:
#     index=file[:6]
#     lag_t = 21
#
#     df = pd.read_csv(home+file)
#     bin_mon=df.loc[:,'bin_money']
#     daily_mon =df.loc[:,'daily_money']
#     df.loc[:,'bin_profile']=bin_mon/daily_mon
#     bin_PF=df.loc[:,'bin_profile'].reset_index(drop=True)
#
#     predict_vol = ESV_prediction(bin_PF,binnum,lag_t)
#     predict_vol = pd.Series(predict_vol)
#     original_vol=extract_xti(df,binnum)
#
#     mape =bin_mape(original_vol,predict_vol)
#     mape_df = pd.DataFrame({index: mape})
#
#     list1=sorted(mape)
#     quantile =list1[int(len(mape)*0.95)]# 计算volume95%分位数
#     mape_df = mape_df.loc[mape_df[index] <= quantile].reset_index(drop= True)
#     mape_all=pd.concat([mape_all,mape_df],axis=1)
#
#     #line_plot(original_vol,predict_vol, index)
# mape_all.to_csv('/home/sliang/JupyterLab/result/system/system_mape.csv')
# ax=sns.boxplot(data=mape_all.iloc[:,:],width=0.8)
#
# ax.set_xticklabels(labels = [],rotation = 45,fontsize = 15) # 放大横轴坐标并逆时针旋转45°
#
# fig=ax.get_figure()
#
# fig.savefig('/home/sliang/JupyterLab/result/system/system_boxplot.jpg')













    