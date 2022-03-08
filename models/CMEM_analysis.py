def predict_muti(df1, df2):
    mu_ti_hats = []
    df2c = df2.loc[df2['mu_ti_1'] != 0, 'mu_ti_1']
    df2c = np.log(df2c)

    df2d = df2.loc[df2['xti_mu_1'] != 0, 'xti_mu_1']
    df2d = np.log(df2d)
    for i in range(22, int(len(df1['Phi']) / 25)):  # 范围从第22天到最后一天
        df11 = df1[(i - 21) * 25:i * 25].reset_index(drop=True)  # 取t的前21天
        df21 = df2[(i - 21) * 25:i * 25].reset_index(drop=True)
        coef = coefficient_estim(data1=df11, data2=df21)

        df22 = df21.loc[df2['mu_ti'] != 0, 'mu_ti']
        df22 = np.log(df22)
        mean = df22.mean(axis=0)
        a0_mu = (1 - coef[3][0] - coef[4][0]) * mean

        for j in range(25):
            if df1['bin_num'][i * 25 + j] == 0:
                continue
            if df2['xt_eta_1'][i * 25 + j] == 0:
                continue
            if df2['eta_t1'][i * 25 + j] == 0:
                continue
            if df2['mu_ti_1'][i * 25 + j] == 0:
                continue
            if df2['xti_mu_1'][i * 25 + j] == 0:
                continue

            else:
                v = [0, 0]
                v[0] = df2c[i * 25 + j]
                v[1] = df2d[i * 25 + j]
                mu_ti_hat = a0_mu + coef[3][0] * v[0] + coef[4][0] * v[1]  # 调用
                mu_ti_hats.append(mu_ti_hat)
    return mu_ti_hats


def extract_mu_ti(df1):
    print(len(df1))
    x_ti1 = df1.loc[df1['bin_num'] != 0, 'mu_ti']
    x_ti2 = x_ti1.reset_index()
    x_ti3 = x_ti2.loc[22 * 24:, 'mu_ti']  # loc[21*24:,'bin_volume'] 与loc[21*24:,bin_volume] 结果数据类型不同
    x_ti3 = np.log(x_ti3)
    x_ti = x_ti3.values.tolist()
    return x_ti


def predict_eta_ti(df1, df2):
    eta_ti_hats = []
    df2a = df2.loc[df2['eta_t1'] != 0, 'eta_t1']
    df2a = np.log(df2a)

    df2b = df2.loc[df2['xt_eta_1'] != 0, 'xt_eta_1']
    df2b = np.log(df2b)
    for i in range(22, int(len(df1['Phi']) / 25)):  # 范围从第22天到最后一天
        df11 = df1[(i - 21) * 25:i * 25].reset_index(drop=True)  # 取t的前21天
        df21 = df2[(i - 21) * 25:i * 25].reset_index(drop=True)
        coef = coefficient_estim(data1=df11, data2=df21)
        df22 = df21.loc[df2['eta_t'] != 0, 'eta_t']
        df22 = np.log(df22)
        mean = df22.mean(axis=0)
        a0_eta = (1 - coef[1][0] - coef[2][0]) * mean

        for j in range(25):
            if df1['bin_num'][i * 25 + j] == 0:
                continue
            if df2['xt_eta_1'][i * 25 + j] == 0:
                continue
            if df2['eta_t1'][i * 25 + j] == 0:
                continue
            if df2['mu_ti_1'][i * 25 + j] == 0:
                continue
            if df2['xti_mu_1'][i * 25 + j] == 0:
                continue

            else:
                v = [0, 0]
                v[0] = df2a[i * 25 + j]
                v[1] = df2b[i * 25 + j]
                eta_ti_hat = a0_eta + coef[1][0] * v[0] + coef[2][0] * v[1]  # 调用
                eta_ti_hats.append(eta_ti_hat)
    return eta_ti_hats


def extract_log_eta_ti(df1):
    print(len(df1))
    x_ti1 = df1.loc[df1['bin_num'] != 0, 'eta_t']
    x_ti2 = x_ti1.reset_index()
    x_ti3 = x_ti2.loc[22 * 24:, 'eta_t']  # loc[21*24:,'bin_volume'] 与loc[21*24:,bin_volume] 结果数据类型不同
    x_ti3 = np.log(x_ti3)
    x_ti = x_ti3.values.tolist()
    return x_ti


def extract_eta_ti(df1):
    print(len(df1))
    x_ti1 = df1.loc[df1['bin_num'] != 0, 'eta_t']
    x_ti2 = x_ti1.reset_index()
    x_ti3 = x_ti2.loc[22 * 24:, 'eta_t']  # loc[21*24:,'bin_volume'] 与loc[21*24:,bin_volume] 结果数据类型不同
    x_ti = x_ti3.values.tolist()
    return x_ti


def extract_eta_ti_all(df1):
    print(len(df1))

    x_ti1 = df1.loc[22 * 24:, 'eta_t']  # loc[21*24:,'bin_volume'] 与loc[21*24:,bin_volume] 结果数据类型不同
    x_ti3 = x_ti1.reset_index(drop=True)
    x_ti3 = x_ti3 * 25
    x_ti = x_ti3.values.tolist()
    return x_ti
