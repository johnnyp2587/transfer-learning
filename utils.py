import pandas as pd
import numpy as np
import torch
from empyrical import sharpe_ratio, max_drawdown, annual_return, annual_volatility
from empyrical import sortino_ratio, downside_risk, value_at_risk, tail_ratio
from scipy.stats import skew, kurtosis

def calmar_ratio(x):
    return annual_return(x).values/-max_drawdown(x)

def compute_performance_metrics(df_returns):
    '''

    :param df_returns:
    :return:
    '''

    # metrics to compute
    pf_metrics = [sharpe_ratio, calmar_ratio, max_drawdown, annual_return, annual_volatility,
                  sortino_ratio, downside_risk, value_at_risk, tail_ratio, skew, kurtosis]
    pf_metrics_labels = ["SR", "CR", "MDD", "ANN_RET", "ANN_VOL", "SortR", "DownRisk", "VaR", "TailR", "Skew", "Kurt"]

    # compute performance metric
    df_metrics = pd.DataFrame(index=range(df_returns.shape[1]), columns=pf_metrics_labels)
    for (pf, pf_label) in zip(pf_metrics, pf_metrics_labels):
        df_metrics[pf_label] = np.array(pf(df_returns))
    df_metrics.index = df_returns.columns

    return df_metrics


def get_data(data_config, problem_config, model_config):
    '''

    :return:
    '''
    Xtrain_tasks, Xval_tasks, Xtest_tasks = {}, {}, {}
    for region in data_config["region"]:
        # pre-allocation
        region_task_paths = [t + "_all_assets_data.pkl.gz" for t in data_config[region]]
        Xtrain_tasks[region], Xval_tasks[region], Xtest_tasks[region] = {}, {}, {}

        for (tk_path, tk) in zip(region_task_paths, data_config[region]):
            # get data
            df = pd.read_pickle(data_config["data_path"] + tk_path)
            df_train = df.iloc[:-(problem_config["val_period"] + problem_config["holdout_period"])]
            if problem_config["val_period"] != 0:
                df_val = df.iloc[-(problem_config["val_period"] + problem_config[
                    "holdout_period"]):-problem_config["holdout_period"]]
            else:
                df_val = df.iloc[:-(problem_config["val_period"] + problem_config["holdout_period"])]
            df_test = df.iloc[-problem_config["holdout_period"]:]

            # transform in tensor
            Xtrain_tasks[region][tk] = torch.from_numpy(df_train.values).to(model_config["device"])
            Xval_tasks[region][tk] = torch.from_numpy(df_val.values).to(model_config["device"])
            Xtest_tasks[region][tk] = torch.from_numpy(df_test.values).to(model_config["device"])
            print(region, tk, Xtrain_tasks[region][tk].size())

    return Xtrain_tasks, Xval_tasks, Xtest_tasks


def calc_tcosts(signal):
    slip = 0.0005 * 0.00
    bp = 0.0020 * 0.00
    tc = (torch.abs(signal[:, 1:, :] - signal[:, :-1, :]) * (bp + slip))
    tc = torch.cat([torch.zeros(signal.size(0), 1, signal.size(2)).double(), tc], dim=1)
    return tc
