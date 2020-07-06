import pandas as pd
import no_transfer_linear, no_transfer_lstm
import global_linear_linear, global_linear_lstm, global_lstm_linear, global_lstm_lstm
import torch
import utils
import pickle
import numpy as np
import random

# data params
manualSeed = 999999999
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_config = {"data_path": ".\\Tasks\\",
               "region": ["Americas", "Europe", "Asia and Pacific", "MEA"],
               "Europe": ["Europe_AEX", "Europe_ASE", "Europe_ATX", "Europe_BEL20", "Europe_BUX",
                          "Europe_BVLX", "Europe_CAC", "Europe_CYSMMAPA", "Europe_DAX", "Europe_HEX",
                          "Europe_IBEX", "Europe_ISEQ", "Europe_KFX", "Europe_OBX", "Europe_OMX",
                          "Europe_SMI", "Europe_UKX", "Europe_VILSE", "Europe_WIG20", "Europe_XU100",
                          "Europe_SOFIX", "Europe_SBITOP", "Europe_PX", "Europe_CRO"],
               "Asia and Pacific": ["Asia and Pacific_AS51", "Asia and Pacific_FBMKLCI", "Asia and Pacific_HSI",
                                    "Asia and Pacific_JCI", "Asia and Pacific_KOSPI", "Asia and Pacific_KSE100",
                                    "Asia and Pacific_NIFTY", "Asia and Pacific_NKY", "Asia and Pacific_NZSE50FG",
                                    "Asia and Pacific_PCOMP", "Asia and Pacific_STI", "Asia and Pacific_SHSZ300",
                                    "Asia and Pacific_TWSE"],
               "Americas": ["Americas_IBOV", "Americas_MEXBOL", "Americas_MERVAL", "Americas_SPTSX", "Americas_SPX",
                            "Americas_RTY"],
               "MEA": ["MEA_DFMGI", "MEA_DSM", "MEA_EGX30", "MEA_FTN098", "MEA_JOSMGNFF",
                       "MEA_KNSMIDX", "MEA_KWSEPM", "MEA_MOSENEW", "MEA_MSM30", "MEA_NGSE30", "MEA_PASISI",
                       "MEA_SASEIDX", "MEA_SEMDEX", "MEA_TA-35", "MEA_TOP40"],
               "additional_data_path": "_all_assets_data.pkl.gz"
               }

# problem params
problem_config = {"export_path": ".\\",
                  "val_period": 0,  # if val is 0, then its results are the same as training
                  "holdout_period": 252 * 3,
                 }

# model params
model_config = {"tsteps": 10,
                "tasks_tsteps": 0,  # [len(data_config[x]) for x in data_config["region"]] - right
                "batch_size": 32,
                "seq_len": 252,
                "transfer_strat": "global_lstm_lstm",
                "device": torch.device("cuda"),
                "export_losses": False,
                "no_transfer_linear": {"opt_lr": 0.001,
                                       "amsgrad": True,
                                       "export_weights": False
                                       },
                "no_transfer_lstm": {"opt_lr": 0.001,
                                     "amsgrad": True,
                                     "export_model": False,
                                     "out_nhi": 50,
                                     "nlayers": 2,
                                     "drop_rate": 0.1
                                    },
                "global_linear_linear": {"opt_lr": 0.01,
                                         "amsgrad": True,
                                         "export_weights": False,
                                         "in_transfer_dim": 200,
                                         "out_transfer_dim": 200
                                         },
                "global_lstm_linear": {"opt_lr": 0.01,
                                       "amsgrad": True,
                                       "export_model": False,
                                       "in_transfer_dim": 5,
                                       "out_transfer_dim": 5,
                                       "nlayers": 2,
                                       "drop_rate": 0.1
                                       },
                "global_linear_lstm": {"opt_lr": 0.01,
                                       "amsgrad": True,
                                       "export_model": False,
                                       "in_transfer_dim": 5,
                                       "out_transfer_dim": 5,
                                       "in_nlayers": 2,
                                       "out_nlayers": 2,
                                       "out_nhi": 10,
                                       "drop_rate": 0.1
                                       },
                "global_lstm_lstm": {"opt_lr": 0.01,
                                     "amsgrad": True,
                                     "export_model": False,
                                     "in_transfer_dim": 5,
                                     "out_transfer_dim": 5,
                                     "in_nlayers": 2,
                                     "out_nlayers": 2,
                                     "nlayers": 2,
                                     "out_nhi": 10,
                                     "drop_rate": 0.1,
                                     "drop_rate_transfer": 0.1
                                     }
                }

# main routine

# pre-allocation
export_label = "valperiod_" + str(problem_config["val_period"]) + "_testperiod_" + str(problem_config["holdout_period"]) + \
               "_tsteps_" + str(model_config["tsteps"]) + "_tksteps_" + str(model_config["tasks_tsteps"]) + "_batchsize_" + \
               str(model_config["batch_size"]) + "_seqlen_" + str(model_config["seq_len"]) + "_transferstrat_" + \
               model_config["transfer_strat"] + "_lr_" + str(model_config[model_config["transfer_strat"]]["opt_lr"])
data_config["export_label"] = export_label
problem_config["export_label"] = export_label
model_config["export_label"] = export_label
model_config["export_path"] = problem_config["export_path"]

# get data
Xtrain_tasks, Xval_tasks, Xtest_tasks = utils.get_data(data_config, problem_config, model_config)

# set model
if model_config["transfer_strat"] == "no_transfer_linear":
    transfer_trad_strat = no_transfer_linear.NoTransferLinear(Xtrain_tasks, model_config)
    add_label = [""] * len(data_config["region"])

elif model_config["transfer_strat"] == "no_transfer_lstm":
    transfer_trad_strat = no_transfer_lstm.NoTransferLSTM(Xtrain_tasks, model_config)
    add_label = ["_nhi_" + str(model_config["no_transfer_lstm"]["out_nhi"]) +
                 "_nlayers_" + str(model_config["no_transfer_lstm"]["nlayers"]) +
                 "dpr" + str(model_config["no_transfer_lstm"]["drop_rate"]) for x in data_config["region"]]

elif model_config["transfer_strat"] == "global_linear_linear":
    transfer_trad_strat = global_linear_linear.GlobalLinearLinear(Xtrain_tasks, model_config)
    add_label = ["_indim_" + str(model_config["global_linear_linear"]["in_transfer_dim"]) +
                 "_outdim_" + str(model_config["global_linear_linear"]["out_transfer_dim"]) for x in data_config["region"]]

elif model_config["transfer_strat"] == "global_lstm_linear":
    transfer_trad_strat = global_lstm_linear.GlobalLSTMLinear(Xtrain_tasks, model_config)
    add_label = ["_indim_" + str(model_config["global_lstm_linear"]["in_transfer_dim"]) +
                 "_outdim_" + str(model_config["global_lstm_linear"]["out_transfer_dim"]) +
                 "_inlay_" + str(model_config["global_lstm_linear"]["nlayers"]) +
                 "dpr" + str(model_config["global_lstm_linear"]["drop_rate"]) for x in data_config["region"]]

elif model_config["transfer_strat"] == "global_linear_lstm":
    transfer_trad_strat = global_linear_lstm.GlobalLinearLSTM(Xtrain_tasks, model_config)
    add_label = ["_indim_" + str(model_config["global_linear_lstm"]["in_transfer_dim"]) +
                 "_outdim_" + str(model_config["global_linear_lstm"]["out_transfer_dim"]) +
                 "_inlay_" + str(model_config["global_linear_lstm"]["in_nlayers"]) +
                 "_outlay_" + str(model_config["global_linear_lstm"]["out_nlayers"]) +
                 "_lindim_" + str(model_config["global_linear_lstm"]["out_nhi"]) +
                 "dpr" + str(model_config["global_linear_lstm"]["drop_rate"]) for x in data_config["region"]]

elif model_config["transfer_strat"] == "global_lstm_lstm":
    transfer_trad_strat = global_lstm_lstm.GlobalLSTMLSTM(Xtrain_tasks, model_config)
    add_label = ["_indim_" + str(model_config["global_lstm_lstm"]["in_transfer_dim"]) +
                 "_outdim_" + str(model_config["global_lstm_lstm"]["out_transfer_dim"]) +
                 "_inlay_" + str(model_config["global_lstm_lstm"]["in_nlayers"]) +
                 "_outlay_" + str(model_config["global_lstm_lstm"]["out_nlayers"]) +
                 "_odim_" + str(model_config["global_lstm_lstm"]["out_nhi"]) +
                 "_ltr_" + str(model_config["global_lstm_lstm"]["nlayers"]) +
                 "_dpr_" + str(model_config["global_lstm_lstm"]["drop_rate"]) +
                 "_dtr_" + str(model_config["global_lstm_lstm"]["drop_rate_transfer"]) for x in data_config["region"]]

# additional labelling
to_add_label = {}
for (lab, region) in zip(add_label, data_config["region"]):
    to_add_label[region] = lab

# train model
import time
start=time.time()
transfer_trad_strat.train()
print(time.time()-start)

# get signals
Xtrain_signal = transfer_trad_strat.predict(Xtrain_tasks)
Xval_signal = transfer_trad_strat.predict(Xval_tasks)
Xtest_signal = transfer_trad_strat.predict(Xtest_tasks)

# compute results
k = True
for region in data_config["region"]:
    region_task_paths = [t + "_all_assets_data.pkl.gz" for t in data_config[region]]

    z = True
    for (tk, tk_path) in zip(data_config[region], region_task_paths):

        # get signal
        pred_train = Xtrain_signal[region][tk].cpu()
        pred_val = Xval_signal[region][tk].cpu()
        pred_test = Xtest_signal[region][tk].cpu()

        # get target
        Ytrain = Xtrain_tasks[region][tk].view(1, -1, Xtrain_tasks[region][tk].size(1))[:, 1:].cpu()
        Yval = Xval_tasks[region][tk].view(1, -1, Xval_tasks[region][tk].size(1))[:, 1:].cpu()
        Ytest = Xtest_tasks[region][tk].view(1, -1, Xtest_tasks[region][tk].size(1))[:, 1:].cpu()

        # compute returns
        df_train_ret = pred_train.mul(Ytrain)[0].cpu().numpy() - utils.calc_tcosts(pred_train)[0].cpu().numpy()
        df_val_ret = pred_val.mul(Yval)[0].cpu().numpy() - utils.calc_tcosts(pred_val)[0].cpu().numpy()
        df_test_ret = pred_test.mul(Ytest)[0].cpu().numpy() - utils.calc_tcosts(pred_test)[0].cpu().numpy()

        # get performance metrics
        df = pd.read_pickle(data_config["data_path"] + tk_path)
        df_train_ret = pd.DataFrame(df_train_ret, columns=df.columns)
        df_train_metrics = utils.compute_performance_metrics(df_train_ret)
        df_train_metrics["exchange"] = tk

        df_val_ret = pd.DataFrame(df_val_ret, columns=df.columns)
        df_val_metrics = utils.compute_performance_metrics(df_val_ret)
        df_val_metrics["exchange"] = tk

        df_test_ret = pd.DataFrame(df_test_ret, columns=df.columns)
        df_test_metrics = utils.compute_performance_metrics(df_test_ret)
        df_test_metrics["exchange"] = tk

        if z:
            all_df_train_metrics = df_train_metrics.copy()
            all_df_val_metrics = df_val_metrics.copy()
            all_df_test_metrics = df_test_metrics.copy()
            z = False
        else:
            all_df_train_metrics = pd.concat([all_df_train_metrics, df_train_metrics], axis=0)
            all_df_val_metrics = pd.concat([all_df_val_metrics, df_val_metrics], axis=0)
            all_df_test_metrics = pd.concat([all_df_test_metrics, df_test_metrics], axis=0)

    # export results
    all_df_train_metrics["region"] = region
    all_df_train_metrics["set"] = "train"
    all_df_val_metrics["region"] = region
    all_df_val_metrics["set"] = "val"
    all_df_test_metrics["region"] = region
    all_df_test_metrics["set"] = "test"

    pd.concat([all_df_train_metrics, all_df_val_metrics, all_df_test_metrics], axis=0).to_csv(
        problem_config["export_path"] + region + "_" + problem_config["export_label"] + to_add_label[region] + ".csv")
    pickle.dump(model_config, open(problem_config["export_path"] + region + "_" + problem_config["export_label"] +
                                         to_add_label[region] + "_modelconfig.pkl.gz", "wb"))
    if model_config["export_losses"]:
        pickle.dump(transfer_trad_strat.losses[region], open(problem_config["export_path"] + region + "_" +
                                                             problem_config["export_label"] + to_add_label[region] +
                                                             "_losses.pkl.gz", "wb"))
