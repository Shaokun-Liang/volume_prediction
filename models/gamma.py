from sklearn.utils import stats
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import  os


def calc_rho_t(idx,df):  # idxz没看懂
    df_phi = df['Phi']
    if idx <= 0:
        return 0
    else:
        return df_phi.cumsum()[idx - 1]


def  calc_alpha_t(idx, model_params={'aw_mu': 0.7, 'aw_sigma': 0.4}):
    if idx <= 0:
        return 0
    steps = 25
    time_weight = idx / steps
    alpha_weight = stats.norm.pdf(time_weight, loc=model_params['aw_mu'], scale=model_params['aw_sigma'])
    if time_weight > 0.75:
        alpha_weight = alpha_weight if alpha_weight > 1 else 1
    alpha_t = idx * alpha_weight
    return alpha_t


def calc_prior_params(prior_mu, prior_sigma):
    prA = prior_mu ** 2 / prior_sigma ** 2 + 2
    prB = prior_mu * (prA - 1)
    return (prA, prB)


def calc_post_params(prior_params, alpha_t, rho_t, cum_volume):
    if rho_t <= 0:
        return prior_params
    prA, prB = prior_params
    poA = prA + alpha_t
    poB = (cum_volume * alpha_t / rho_t) + prB
    return (poA, poB)


def calc_cum_volume(idx, min_volume):
    if idx <= 0:
        return 0
    else:
        cum_volume = min_volume.cumsum()[idx - 1]
        return cum_volume


def calc_cum_profile(idx, min_volume):
    if idx <= 0:
        return 0
    else:
        cum_profile = min_volume.cumsum()[idx - 1] / min_volume.sum()
        return cum_profile


def calc_gamma_volume(idx, profile, cum_volume, cum_profile, prior_mu, prior_sigma, model_params):
    prior_params = calc_prior_params(prior_mu, prior_sigma)
    rho_t = calc_rho_t(idx, profile)
    alpha_t = calc_alpha_t(idx, model_params)
    poA, poB = calc_post_params(prior_params, alpha_t, rho_t, cum_volume)
    gamma_vol = poB / (poA - 1)
    tpc_gamma = cum_volume / (cum_volume + (1 - rho_t) * gamma_vol)
    tpc_error = tpc_gamma - cum_profile
    return gamma_vol, tpc_gamma, tpc_error


def beyesian_prediction_bin(df1):
    V_T_hats = []
    df_daily = df1.loc[:,'daily_volume'].reset_index(drop=True)
    df_phi = df1.loc[:, 'Phi'].reset_index(drop=True)
    df_bin = df1.loc[:, 'bin_volume'].reset_index(drop=True)
    
    
    for i in range(21, int(len(df_bin) / 25)):
        df_eta = df_daily[(i - 21) * 25:i  * 25].reset_index(drop=True)

        df_referPhi = df_phi[(i - 1) * 25:i  * 25].reset_index(drop=True)
        df_rv = df_bin[i * 25:(i + 1) * 25].reset_index(drop=True)
        df_Rphi=df_phi[i * 25:(i + 1) * 25].reset_index(drop=True)
        for j in range(25):
            if j == 0:
                continue
            else:
                #                    df_cum_profile1=df_cum_profile[j::25]
                df_conpress_daily = df_eta.loc[1::25].reset_index(drop=True)
                mu_V = np.mean(df_conpress_daily)
                sigma_V = np.std(df_conpress_daily, ddof=1)

                K = 1 + (mu_V ** 2 / sigma_V ** 2)

                V_t = df_rv.cumsum()[j-1]  # 没清零

                rho_t =df_Rphi.cumsum()[j-1]  # 没清零，改成21day平均
                phi_t=df_referPhi.cumsum()[j]
                alpha_t = calc_alpha_t(j)  # 已经清零
                V_T = (V_t / rho_t + (mu_V - (V_t / rho_t)) / (1 + (alpha_t / K)))*phi_t
                V_T_hats.append(V_T)
    V_T_hats1 = []
    for m in range(int(len(V_T_hats) / 24)):
        V_T_hats2 = V_T_hats[m * 24:(m + 1) * 24]
        for j in range(24):
            if j == 0:
                V_T_hats1.append(V_T_hats2[j])
            else:
                V_T_hats1.append(V_T_hats2[j] - V_T_hats2[j-1])

    return V_T_hats1
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
#filenames=['000001_XSHG_25_daily.csv','688128_XSHG_25_daily.csv']
mape_all=pd.DataFrame()
binnum=24+1
for file in filenames:
    index=file[:6]
    df=pd.read_csv(home+file)
    df.fillna(method="bfill",inplace=True)
    df.loc[:,'Phi']=df['bin_volume']/df['daily_volume']

    pred=beyesian_prediction_bin(df)

    df=df[21*25:].reset_index(drop=True)
    true=df.loc[df['bin_num']!=0,'bin_volume'].reset_index(drop=True)

    mape=bin_mape(pred,true)

    pred=pd.DataFrame({index + '_pred': pred})
    true=pd.DataFrame({index +'_ture': true})
    pred.to_csv('/home/sliang/JupyterLab/new_result/gamma/'+index+'_gamma_pred.csv', index=False)
    true.to_csv('/home/sliang/JupyterLab/new_result/gamma/'+index+'_gamma_true.csv', index=False)
    mape_df = pd.DataFrame({index: mape})
    mape_all=pd.concat([mape_all,mape_df],axis=1)

mape_all.to_csv('/home/sliang/JupyterLab/new_result/dynamic_ewma/gamma_mape.csv',index=False)
xtick=mape_all.columns
ax=sns.boxplot(data=mape_all.iloc[:,:],width=0.8,showmeans=False,showfliers=False)
ax.set_xticklabels(labels = xtick,rotation = 45,fontsize = 10)
plt.xlabel('ticker') # 放大横轴名称
plt.ylabel('(%)')
plt.title('gamma_error')
fig=ax.get_figure()
plt.show()
plt.tight_layout()
fig.savefig('/home/sliang/JupyterLab/result/beysian/gamma_mape_boxplot_nofliers.jpg')




