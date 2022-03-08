import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def mapeTest(df1, df2, index):


    def coefficient_estim(data1, data2):

        d1_1 = data2.loc[:, 'eta_t1']

        d1_1 = np.log(d1_1)  # 传入η_t-1

        d5_1 = data2.loc[:, 'xt_eta_1']

        d5_1 = np.log(d5_1)  # 传入x_t-1^((η))

        d2 = data1.loc[:, 'Phi']  # 传入φ_j
        logd2 = np.log(d2)

        d3_1 = data2.loc[:, 'mu_ti_1']
        d3_1 = np.log(d3_1)  # 传入μ_ti-1

        d4_1 = data2.loc[:, 'xti_mu_1']
        d4_1 = np.log(d4_1)  # 传入x_ti-1^((μ))

        d6 = data1.loc[:, 'bin_volume']  # 传入x_tau
        d6 = np.log(d6)

        '''df1=data2.loc[:,'a_tau0'].map(lambda x:1)#传入向量a_tau的第1列
        df2=np.log(data2.loc[:,'eta_t1']) #传入向量a_tau的第2列
        df3=np.log(data2.loc[:,'xt_eta_1'])#传入向量a_tau的第3列
        df4=np.log(data2.loc[:,'mu_ti_1']) #传入向量a_tau的第4列  
        df5=np.log(data2.loc[:,'xti_mu_1']) #传入向量a_tau的第5列'''
        # b系数矩阵
        b0 = logd2 - d6
        b11 = d6 - logd2
        b1 = b11.sum(axis=0)
        b21 = -d1_1 * b0
        b2 = b21.sum(axis=0)
        b31 = -d5_1 * b0
        b3 = b31.sum(axis=0)
        b41 = -d3_1 * b0
        b4 = b41.sum(axis=0)
        b51 = -d4_1 * b0
        b5 = b51.sum(axis=0)

        # w系数矩阵第一行
        w11 = len(df1)
        w12 = d1_1.sum(axis=0)
        w13 = d5_1.sum(axis=0)
        w14 = d3_1.sum(axis=0)
        w15 = d4_1.sum(axis=0)
        # w系数矩阵第二行
        w21 = d1_1.sum(axis=0)
        x22 = d1_1 * d1_1
        w22 = x22.sum(axis=0)
        x23 = d1_1 * d5_1
        w23 = x23.sum(axis=0)
        x24 = d1_1 * d3_1
        w24 = x24.sum(axis=0)
        x25 = d1_1 * d4_1
        w25 = x25.sum(axis=0)
        # w系数矩阵第三行
        w31 = d5_1.sum(axis=0)
        x32 = d5_1 * d1_1
        w32 = x32.sum(axis=0)
        x33 = d5_1 * d5_1
        w33 = x33.sum(axis=0)
        x34 = d5_1 * d3_1
        w34 = x34.sum(axis=0)
        x35 = d5_1 * d4_1
        w35 = x35.sum(axis=0)
        # w系数矩阵第四行
        w41 = d3_1.sum(axis=0)
        x42 = d3_1 * d1_1
        w42 = x42.sum(axis=0)
        x43 = d3_1 * d5_1
        w43 = x43.sum(axis=0)
        x44 = d3_1 * d3_1
        w44 = x44.sum(axis=0)
        x45 = d3_1 * d4_1
        w45 = x45.sum(axis=0)
        # w系数矩阵第五行
        w51 = d4_1.sum(axis=0)
        x52 = d4_1 * d1_1
        w52 = x52.sum(axis=0)
        x53 = d4_1 * d5_1
        w53 = x53.sum(axis=0)
        x54 = d4_1 * d3_1
        w54 = x54.sum(axis=0)
        x55 = d4_1 * d4_1
        w55 = x55.sum(axis=0)

        w = np.matrix([[w11, w12, w13, w14, w15], [w21, w22, w23, w24, w25],
                       [w31, w32, w33, w34, w35], [w41, w42, w43, w44, w45],
                       [w51, w52, w53, w54, w55]])
        b = np.matrix([b1, b2, b3, b4, b5]).T

        result = np.linalg.solve(w, b)  # (α_0^(μ)+α_0^((η)),β_1^((η)),α_1^((η)),β_1^((μ)),α_1^((μ)))
        result = result.tolist()
        return result




    def extract_xti(df1):

        x_ti1 = df1.loc[df1['bin_num']!=0, 'bin_volume']


        x_ti2 = x_ti1[21*24:].reset_index(drop=True)  # loc[21*24:,'bin_volume'] 与loc[21*24:,bin_volume] 结果数据类型不同

        return x_ti2





    def cmem(v, coef):
        '''
        coef[0]:alpha_0_(eta)
        coef[1]:beta_1_(eta)
        coef[2]:alpha_1_(eta)
        coef[3]:beta_1_(mu)
        coef[4]:alpha_1(mu)
        v[0]:d1_1(#传入η_t-1)
        v[1]:d5_1(#传入x_t-1^((η)))
        v[2]:d3_1(传入μ_ti-1)
        v[3]:d4_1(#传入x_ti-1^((μ)))
        v[4]:ln(Phi)
        v:eta_t-1,xt-1_eta,mu_ti-1,xti-1_mu,Phi,需要输入的变量值，array
        coef:gmm估计出来的系数,list
        '''
        x_ti_hat = v[4] + coef[0][0] + coef[1][0] * v[0] + coef[2][0] * v[1] + coef[3][0] * v[2] + coef[4][0] * v[3]
        return x_ti_hat

    def prediction(df1, df2):

        # rolling:用前21天数据训练模型，用前一个bin的数据做预测
        # return：返回从第22天到最后一天每个bin的预测结果,list

        x_ti_hats = []
        df2a = df2.loc[:, 'eta_t1']
        df2a = np.log(df2a)

        df2b = df2.loc[:, 'xt_eta_1']
        df2b = np.log(df2b)

        df2c = df2.loc[:, 'mu_ti_1']
        df2c = np.log(df2c)

        df2d = df2.loc[:, 'xti_mu_1']
        df2d = np.log(df2d)

        df1a = np.log(df1['Phi'])

        for i in range(21, int(len(df1['Phi']) / 25)):  # 范围从第22天到最后一天
            df11 = df1[(i - 21) * 25:i * 25].reset_index(drop=True)  # 取t的前21天
            df21 = df2[(i - 21) * 25:i * 25].reset_index(drop=True)

            coef = coefficient_estim(data1=df11, data2=df21)

            for j in range(25):
                if df1['bin_num'][i * 25 + j] == 0:
                    continue
                else:
                    v = [0, 0, 0, 0, 0]
                    v[0] = df2a[i * 25 + j]
                    v[1] = df2b[i * 25 + j]
                    v[2] = df2c[i * 25 + j]
                    v[3] = df2d[i * 25 + j]
                    v[4] = df1a[i * 25 + j]
                    x_ti_hat = cmem(v, coef)  # 调用
                    x_ti_hats.append(x_ti_hat)

        return x_ti_hats




    def cal_mape(x_ti, x_ti_hats):

        # x_ti:all real bin volume,array or list(lenth = bin_num*t)
        # x_ti_hat:predicted bin volume,array or list

        mape = 0
        x_ti = list(x_ti)
        x_ti_hats = list(x_ti_hats)
        if len(x_ti) != len(x_ti_hats):
            print("lenth error!")
        else:
            for i in range(len(x_ti)):
                if x_ti[i] == 0:
                    print("Error:0 appears in the denominator")
                    break
                elif x_ti_hats[i] == None:
                    continue
                else:
                    mape = mape + abs((x_ti_hats[i] - x_ti[i]) / x_ti[i])
            mape /= len(x_ti)

        return mape

    def cal_mape_error(x_ti, x_ti_hats):

        # x_ti:all real bin volume,array or list(lenth = bin_num*t)
        # x_ti_hat:predicted bin volume,array or list

        mape = abs((x_ti_hats - x_ti) / x_ti)*100
        return mape

    def daily_mape(x_ti, x_ti_hat, t):
        mape = []
        for i in range(1, t + 1):
            x_ti_daily = x_ti[(i - 1) * 24:i * 24 - 1]
            x_ti_hat_daily = x_ti_hat[(i - 1) * 24:i * 24 - 1]
            mape_daily = cal_mape(x_ti_daily, x_ti_hat_daily)
            mape.append(mape_daily)
        return mape

    def bin_mape(x_ti, x_ti_hat):
        if len(x_ti) != len(x_ti_hat):
            print("lenth error!")
            print((len(x_ti)))
            print((len(x_ti_hat)))
            return None
        else:
            mape = []
            for i in range(len(x_ti)):
                if x_ti[i] < 1e-5:
                    mape_bin = np.nan
                    mape.append(mape_bin)
                else:
                    mape_bin = abs((x_ti[i] - x_ti_hat[i]) / x_ti[i]) * 100
                    mape.append(mape_bin)

        return mape

    def CMEM_linear_plot(x_ti, x_ti_hats, t=10):
        plt.figure(10)
        x_ti_hats = pd.Series(x_ti_hats)
        x_ti = pd.Series(x_ti)
        sns.lineplot(data=x_ti_hats[:24 * t], linewidth=1.5, alpha=0.8, label='x_ti_hats')
        sns.lineplot(data=x_ti[:24 * t], linewidth=1.5, alpha=0.8, label='x_ti')
        mape = cal_mape(x_ti[0:24 * t], x_ti_hats[0:24 * t])

        plt.text(x=120, y=x_ti[0:240].max(), s='mape= %f' % mape)
        plt.xlabel('bin number')
        plt.ylabel('bin_volume')
        plt.legend()
        plt.title(index + '(CMEM)')
        # plt.savefig('C:/Users/50302/Desktop/组会/volume prediction/result/cmem_fig/'+index+'cmem_mape.jpg',dpi=800) #设置字体大小与格式
        plt.show()
        return 0

    df1.loc[df1['bin_volume'] == 0, 'bin_volume'] = 1
    df2.loc[df2['eta_t1'] == 0, 'eta_t1'] = 1
    df2.loc[df2['xt_eta_1'] == 0, 'xt_eta_1'] = 1
    df2.loc[df2['mu_ti_1'] == 0, 'mu_ti_1'] = 1
    df2.loc[df2['xti_mu_1'] == 0, 'xti_mu_1'] = 1
    df1 = df1[22 * 25:].reset_index(drop=True)  # 除去前22天数据
    df2 = df2[22 * 25:].reset_index(drop=True)

    # α_0^(μ) α_1^((μ)) β_1^((μ)) x0 x1 x2
    # α_0^((η)) α_1^((η)) β_1^((η)) x3 x4 x5
    '''#η_t==d1[t]      t<=N  #表示天数
    #φ_j==d2[j]         j<=48 #表示bin的个数
    #μ_ti==d3[i]
    #x_ti^((μ)) ==d4[i] i<=N*48 #表示样本个数
    #x_t^((η)) ==d5[t]  t<=N
    #x_t==d6[i]'''
    x_ti_hats= np.exp(prediction(df1, df2))

    x_ti = extract_xti(df1)
    pred = pd.DataFrame({index + '_pred': x_ti_hats})
    true = pd.DataFrame({index + '_ture': x_ti})
    pred.to_csv('/home/sliang/JupyterLab/CMEM/test2/' + index + '_CMEM_pred.csv', index=False)
    true.to_csv('/home/sliang/JupyterLab/CMEM/test2/' + index + '_CMEM_true.csv', index=False)


    all_mape=bin_mape(x_ti, x_ti_hats)
    return all_mape
# df1=pd.read_csv('/home/sliang/JupyterLab/CMEM/gmm/derivative/000001_XSHG_25_derivative.csv')
# df2=pd.read_csv('/home/sliang/JupyterLab/CMEM/gmm/gmm/000001_XSHG_25_gmm.csv')
# a1= mapeTest(df1, df2, '000001')
file1_path = r'/home/sliang/JupyterLab/CMEM/gmm/derivative/'
file2_path = r'/home/sliang/JupyterLab/CMEM/gmm/gmm/'
gmm = os.listdir(file2_path)
der = os.listdir(file1_path)
df_mape = pd.DataFrame()
for left, right in zip(der, gmm):
    df1 = pd.read_csv(file1_path + left, engine='python')
    df2 = pd.read_csv(file2_path + right, engine='python')
    index = left[:6]
    df_mape =pd.concat([df_mape,pd.DataFrame({index:mapeTest(df1,df2,index)})],axis=1)

df_mape.to_csv('/home/sliang/JupyterLab/CMEM/test2/mape.csv', index=False)
xtick=df_mape.columns
ax=sns.boxplot(data=df_mape.iloc[:,:],width=0.8,showmeans=False,showfliers=False)
ax.set_xticklabels(labels = xtick,rotation = 45,fontsize = 10)
plt.xlabel('ticker') # 放大横轴名称
plt.ylabel('(%)')
plt.title('CMEM_error')
fig=ax.get_figure()
plt.show()
plt.tight_layout()
fig.savefig('/home/sliang/JupyterLab/CMEM/CMEM_mape_boxplot_nofliers.jpg')


# sns.set_style("white")  #背景风格
# box = sns.boxplot(data=df_mape,width = 0.5,color = 'mediumslateblue')
# scatter_fig = box.get_figure()
# scatter_fig.savefig('/home/sliang/JupyterLab/CMEM/box.jpg', dpi = 800)