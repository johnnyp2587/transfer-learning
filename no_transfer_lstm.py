import numpy as np
import torch
from torch import nn


class NoTransferLSTM:

    def __init__(self, x_tasks, model_config):
        # general params
        self.criterion = self.avg_sharpe_ratio  # nn.MSELoss().cuda()
        self.Xtrain_tasks = x_tasks
        self.tsteps = model_config["tsteps"]
        self.tasks_tsteps = model_config["tasks_tsteps"]
        self.batch_size = model_config["batch_size"]
        self.seq_len = model_config["seq_len"]
        self.device = model_config["device"]
        self.export_path = model_config["export_path"]
        self.export_label = model_config["export_label"]

        # model params
        self.opt_lr = model_config["no_transfer_lstm"]["opt_lr"]
        self.amsgrad = model_config["no_transfer_lstm"]["amsgrad"]
        self.export_weights = model_config["no_transfer_lstm"]["export_model"]
        self.nhi = model_config["no_transfer_lstm"]["out_nhi"]
        self.nlayers = model_config["no_transfer_lstm"]["nlayers"]

        # set learning model per transfer
        self.mtl_list = self.Xtrain_tasks.keys()
        self.sub_mtl_list = {}
        self.model_lin_dict, self.model_lstm_dict, self.opt_dict, self.signal_layer, self.losses = {}, {}, {}, {}, {}
        for tk in self.mtl_list:
            # pre-allocation
            self.model_lin_dict[tk], self.signal_layer[tk], self.model_lstm_dict[tk], \
            self.opt_dict[tk], self.losses[tk] = {}, {}, {}, {}, {}
            self.sub_mtl_list[tk] = self.Xtrain_tasks[tk].keys()

            for sub_tk in self.sub_mtl_list[tk]:

                # parameters
                self.losses[tk][sub_tk] = []
                nin = self.Xtrain_tasks[tk][sub_tk].shape[1]  # number of inputs
                nout = self.Xtrain_tasks[tk][sub_tk].shape[1]  # number of inputs

                # Linear
                nhi, nlayers = self.nhi, self.nlayers
                self.model_lstm_dict[tk][sub_tk] = nn.LSTM(nin, nhi, nlayers, batch_first=True).double().to(self.device)
                self.model_lin_dict[tk][sub_tk] = nn.Linear(nhi, nout).double().to(self.device)
                self.signal_layer[tk][sub_tk] = nn.Tanh().to(self.device)

                # optimizer
                self.opt_dict[tk][sub_tk] = torch.optim.Adam(list(self.model_lstm_dict[tk][sub_tk].parameters()) +
                                                             list(self.model_lin_dict[tk][sub_tk].parameters()) +
                                                             list(self.signal_layer[tk][sub_tk].parameters()),
                                                             lr=self.opt_lr, amsgrad=self.amsgrad)
                print(tk, sub_tk, self.model_lstm_dict[tk][sub_tk], self.model_lin_dict[tk][sub_tk],
                      self.signal_layer[tk][sub_tk], self.opt_dict[tk][sub_tk])

    def train(self):

        for i in range(self.tsteps):
            for tk in self.mtl_list:
                for sub_tk in self.sub_mtl_list[tk]:
                    # Fetch batches
                    start_ids = np.random.permutation(list(range(
                        self.Xtrain_tasks[tk][sub_tk].size(0) - self.seq_len - 1)))[:self.batch_size]
                    XYbatch = torch.stack([self.Xtrain_tasks[tk][sub_tk][i:i + self.seq_len + 1] for i in start_ids],
                                          dim=0)
                    Ytrain = XYbatch[:, 1:, :]  # For all batches, one-step ahead pred
                    Xtrain = XYbatch[:, :-1, :]  # For all batches, one-step ahead pred

                    # Reset gradient and hidden when starting a new sequence
                    self.opt_dict[tk][sub_tk].zero_grad()

                    # forward pass
                    (hidden, cell) = self.get_hidden(self.batch_size, self.nlayers, self.nhi)
                    hidden, cell = self.model_lstm_dict[tk][sub_tk](Xtrain, (hidden, cell))
                    preds = self.signal_layer[tk][sub_tk](self.model_lin_dict[tk][sub_tk](hidden))

                    # loss
                    loss = self.criterion(preds, Ytrain)
                    self.losses[tk][sub_tk].append(loss.item())

                    # gradient + optimization
                    loss.backward()
                    self.opt_dict[tk][sub_tk].step()

            # iter training
            if (i % 100) == 1:
                print(i)

        if self.export_weights:
            for tk in self.mtl_list:
                for sub_tk in self.sub_mtl_list[tk]:
                    torch.save(self.model_lin_dict[tk][sub_tk], self.export_path + tk + "_" + sub_tk + "_" +
                               self.export_label + "_linnotransferlstm.pt")
                    torch.save(self.model_lstm_dict[tk][sub_tk], self.export_path + tk + "_" + sub_tk + "_" +
                               self.export_label + "_notransferlstm.pt")

    def predict(self, x_test):

        y_pred = {}
        for tk in self.mtl_list:
            y_pred[tk] = {}
            for sub_tk in self.sub_mtl_list[tk]:
                 # we still need a batch dim, but it's just 1
                 xflat = x_test[tk][sub_tk].view(1, -1, x_test[tk][sub_tk].size(1))
                 with torch.autograd.no_grad():
                     (hidden, cell) = self.get_hidden(1, self.nlayers, self.nhi)
                     hidden, _ = self.model_lstm_dict[tk][sub_tk](xflat[:, :-1], (hidden, cell))
                     y_pred[tk][sub_tk] = self.signal_layer[tk][sub_tk](self.model_lin_dict[tk][sub_tk](hidden))

        return y_pred

    def avg_sharpe_ratio(self, output, target):
        slip = 0.0005 * 0.00
        bp = 0.0020 * 0.00
        rets = torch.mul(output, target)
        tc = (torch.abs(output[:, 1:, :] - output[:, :-1, :]) * (bp + slip))
        tc = torch.cat([torch.zeros(output.size(0), 1, output.size(2)).double().to(self.device), tc], dim=1)
        rets = rets - tc
        avg_rets = torch.mean(rets)
        vol_rets = torch.std(rets)
        loss = torch.neg(torch.div(avg_rets, vol_rets))
        return loss.mean()

    def get_hidden(self, batch_size, nlayers, nhi):
        return (torch.zeros(nlayers, batch_size, nhi).double().to(self.device),
                torch.zeros(nlayers, batch_size, nhi).double().to(self.device))
