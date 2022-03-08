import pandas as pd
import os
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
home='/home/sliang/JupyterLab/input_data/final_data/'
filenames=os.listdir(home)
#filenames=['000001_XSHG_25_daily.csv','688128_XSHG_25_daily.csv']
mape_all=pd.DataFrame()
binnum=24+1
#获取被填补series的index
#获取填补series的dataframe
#根据dataframe中index获取填补index
#利用np.array(index)将填补df与被填补df的index映射一致(不能用reindex，会出现大量空值，需要用set_index)
# df_fill = pd.DataFrame(df_fill)
#     index=df.loc[df['bin_num']==1,'daily_volume'].index.tolist()
#     index = (np.array(index)-1).tolist()
#     df_fill.loc[:,'idx'] = index
#填补
#简化版本df.fillna()
for file in filenames:
    index=file[:6]
    df=pd.read_csv(home+file)
    #count1=df['daily_volume'].isnull().sum(axis=0)
    df.fillna(method="bfill",inplace=True)

    #df.loc[:,'Phi']=df['bin_volume']/df['daily_volume']






    #print(df)

# mape_all=pd.read_csv('/home/sliang/JupyterLab/result/system/system_mape.csv')
# mape_all=mape_all.set_index('Unnamed: 0')
# mape_all=mape_all.reset_index(drop=True)
# xtick=mape_all.columns
# print(xtick)
# print(mape_all.head(5))
# ax=sns.boxplot(data=mape_all.iloc[:,:],width=0.8)
#
# ax.set_xticklabels(labels = xtick,rotation = 45,fontsize = 10) # 放大横轴坐标并逆时针旋转45°
#
# plt.xlabel('ticker') # 放大横轴名称
# plt.title('system_pred')
# fig=ax.get_figure()
# plt.show()
# plt.tight_layout()
# fig.savefig('/home/sliang/JupyterLab/result/system/system_boxplot.jpg')
# home='/home/sliang/JupyterLab/input_data/final_data/'
# filenames=os.listdir(home)
# #filenames=['688388_XSHG_25_daily.csv','688128_XSHG_25_daily.csv']
# mape_all=pd.DataFrame()
# binnum=24+1
# for file in filenames:
#     index=file[:6]
#     df_need=pd.read_csv(home+file)
#     df_true = df_need.loc[df_need['bin_num']!=0,'bin_volume'].reset_index(drop=True)
#     df_pred = df_true[1:].reset_index(drop=True) #待遇测
#     df_true1= df_true[:-1].reset_index(drop=True) #预测
#     mape = bin_mape(df_pred, df_true1)
#     mape_df = pd.DataFrame({index: mape})
#     mape_all = pd.concat([mape_all, mape_df], axis=1)
# #mape_all.to_csv('/home/sliang/JupyterLab/result/system/system_mape_%d_row.csv'%big_order)
# xtick=mape_all.columns
# ax=sns.boxplot(data=mape_all.iloc[:,:],width=0.8,showmeans=False,showfliers=False)
# ax.set_xticklabels(labels = xtick,rotation = 45,fontsize = 10)
# plt.xlabel('ticker') # 放大横轴名称
# plt.ylabel('(%)')
# #plt.ylim(0,100)
# plt.title('volume_error_lag=1')
# fig=ax.get_figure()
# plt.show()
# plt.tight_layout()
# fig.savefig('/home/sliang/JupyterLab/result/system/lag_boxplot_row.jpg')
