import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from scipy.io import wavfile
import numpy as np
import torch.nn.functional as F  # 用relu激活函数
import torch.optim as optim  # 优化器
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import KFold
eps = np.finfo(np.float).eps

# ov1设为100  ov2 epoch设为200 ov3 400
Overlap_num = 'ov1'
Feat_name = 'MFCC+NAP'

class PitayaFan(Dataset):
    def __init__(self, mode, overlap_num, feat_name):
        self.spec_path = 'G:/SED数据集无噪/' + overlap_num + '/' + feat_name + '/' + 'feat/'
        self.label_path = 'G:/SED数据集无噪/' + overlap_num + '/' + feat_name + '/' + 'label/'
        self.mode = mode
        self.feat_name = feat_name
        test_file = []
        train_file = []
        self.nb_sed = 8
        for i, spec_name in enumerate(os.listdir(self.spec_path)):
            if 'train' in spec_name:
                train_file.append(spec_name)
            elif 'test' in spec_name:
                test_file.append(spec_name)

        self.spec = []
        self.label = []
        if self.mode == 'train':
            if self.feat_name == 'NAP' or self.feat_name == 'NAP改进':
                for name in train_file:
                    filename = name.split('.')[0]
                    spec1 = np.loadtxt(self.spec_path + name).transpose()
                    spec1 = spec1[:, ::-1]
                    self.spec.append(spec1)
                    self.label.append(np.load(self.label_path + filename + '.npy'))
            else:
                for name in train_file:
                    filename = name.split('.')[0]
                    spec1 = np.load(self.spec_path + name)
                    # SAI加这一行
                    # spec1[spec1 < 0] = 0
                    self.spec.append(spec1)
                    self.label.append(np.load(self.label_path + filename + '.npy'))
        else:
            if self.feat_name == 'NAP' or self.feat_name == 'NAP改进':
                for name in test_file:
                    filename = name.split('.')[0]
                    spec1 = np.loadtxt(self.spec_path + name).transpose()
                    spec1 = spec1[:, ::-1]
                    self.spec.append(spec1)
                    self.label.append(np.load(self.label_path + filename + '.npy'))
            else:
                for name in test_file:
                    filename = name.split('.')[0]
                    self.spec.append(np.load(self.spec_path + name))
                    self.label.append(np.load(self.label_path + filename + '.npy'))

        self.spec = torch.from_numpy(np.array(self.spec))  # （N，1363,129）
        self.label = torch.from_numpy(np.array(self.label))  # （N,1363,129）
        if feat_name == '幅度谱':
            self.spec = self.spec.view(-1, 129)  # （1363,129）
        elif '+' in feat_name:
            self.spec = self.spec.view(-1, 106)
        else:
            self.spec = self.spec.view(-1, 53)
        self.label = self.label.view(-1, 8)
        self.len = len(self.spec)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.spec[index], self.label[index]


class PitayaSELD(torch.nn.Module):
    def __init__(self, Q, K=8, P=32):
        super(PitayaSELD, self).__init__()
        self.P = P
        self.K = K
        self.Q = Q
        # self.batchNorm = torch.nn.BatchNorm1d(P, affine=False)
        self.convs1 = torch.nn.Conv2d(in_channels=1, out_channels=P, kernel_size=(3, 3), padding=(1, 1))
        self.convs2 = torch.nn.Conv2d(in_channels=P, out_channels=P, kernel_size=(3, 3), padding=(1, 1))
        self.convs3 = torch.nn.Conv2d(in_channels=P, out_channels=P, kernel_size=(3, 3), padding=(1, 1))
        self.Pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 2))
        self.Pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 2))
        self.Pooling3 = torch.nn.MaxPool2d(kernel_size=(1, 2))
        self.Grus = torch.nn.GRU(self.Q, 16, 1, batch_first=True, bidirectional=True)
        # self.dropout = torch.nn.Dropout(p=0.5)

        # =================SED================
        self.sed_fc0 = torch.nn.Linear(P * 16 * 2, P * 8)
        self.sed_fc1 = torch.nn.Linear(P * 8, P * 2)
        self.sed_fc2 = torch.nn.Linear(P * 2, K)
        # =================DOA===============
        # self.doa_fc0 = torch.nn.Linear(64*32, 64)
        # self.doa_fc1 = torch.nn.Linear(64, 3 * K)

    def forward(self, y):
        batch_size = y.size(0)
        y = y.unsqueeze(1)
        y = y.unsqueeze(1)
        # y = self.Pooling1(F.relu(self.batchNorm(self.convs1(y))))
        # y = self.Pooling2(F.relu(self.batchNorm(self.convs2(y))))
        # y = self.Pooling3(F.relu(self.batchNorm(self.convs3(y))))

        y = self.Pooling1(F.relu(self.convs1(y)))
        y = self.Pooling2(F.relu(self.convs2(y)))
        y = self.Pooling3(F.relu(self.convs3(y)))
        y = y.view(batch_size, -1, self.Q)
        y = y.permute(1, 0, 2)
        y, _ = self.Grus(y)
        y = torch.tanh(y)
        y = y.permute(1, 0, 2)
        y = y.view(batch_size, -1)
        # y = self.dropout(y)
        y = torch.sigmoid(self.sed_fc0(y))
        # y = self.dropout(y)
        y = torch.sigmoid(self.sed_fc1(y))
        # y = self.dropout(y)
        y = self.sed_fc2(y)
        return y


def trainModel():
    model.train()
    train_loss_sum = 0
    for i, (paras, sed_label) in enumerate(train_loader):
        paras = paras.to(torch.float32)
        paras = paras.to(device)
        sed_label = sed_label.float()
        sed_label = sed_label.to(device)
        optimizer.zero_grad()

        out_sed = model(paras)
        loss = criterion_sed(out_sed, sed_label)
        loss.backward()
        train_loss_sum += loss.item()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    model.eval()
    val_loss_sum = 0
    get_sed = []
    fact_sed = []
    with torch.no_grad():
        for i, (paras, sed_label) in enumerate(val_loader):
            paras = paras.to(torch.float32)
            paras = paras.to(device)
            sed_label = sed_label.float()
            sed_label = sed_label.to(device)
            out_sed = model(paras)
            loss = criterion_sed(out_sed, sed_label)
            val_loss_sum += loss.item()

            if i < len(val_loader) - 1:
                # print(np.array(out_sed.cuda().cpu()).shape)  # [512,8] [178,8]
                get_sed.append(np.array(out_sed.cuda().cpu()))
                fact_sed.append(np.array(sed_label.cuda().cpu()))

            else:
                get_sed_last = np.array(out_sed.cuda().cpu())
                fact_sed_last = np.array(sed_label.cuda().cpu())

    return train_loss_sum / len(train_loader), val_loss_sum / len(
        val_loader), get_sed, fact_sed, get_sed_last, fact_sed_last


def compute_sed_scores(y, x):
    TP = ((2 * x - y) == 1).sum()  # (y→sed_pred, x→sed_gt)
    Nref, Nsys = x.sum(), y.sum()
    prec = float(TP) / float(Nsys + eps)  # prec=TP/(TP+FP)
    recall = float(TP) / float(Nref + eps)  # recall=TP/(TP+TN)
    f1_score = 2 * prec * recall / (prec + recall + eps)

    FP = np.logical_and(x == 0, y == 1).sum()
    FN = np.logical_and(x == 1, y == 0).sum()

    S = np.minimum(FP, FN)
    D = np.maximum(0, FN - FP)
    I = np.maximum(0, FP - FN)
    ER = (S + D + I) / (Nref + 0.0)
    return ER, f1_score


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


if __name__ == "__main__":
    trainset = PitayaFan(mode='train', overlap_num=Overlap_num, feat_name=Feat_name)
    # testset = PitayaFan(mode='test', overlap_num=Overlap_num, feat_name=Feat_name)
    threshold = 0.5
    num_seg = 1363    # 1363
    if Feat_name == '幅度谱':
        num = 16
    elif '+' in Feat_name:
        num = 13
    else:
        num = 6
    BATCH_SIZE = 256
    SavePath = 'G:/SED数据集无噪/' + Overlap_num + '/' + Feat_name + '/'
    # early_stopping = EarlyStopping(save_path=SavePath)
    model = PitayaSELD(Q=num)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)
    print(model)
    criterion_sed = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    N_EPOCHS = 100
    train_loss = np.zeros(N_EPOCHS + 1)
    val_loss = np.zeros(N_EPOCHS + 1)
    sed_loss = np.zeros((N_EPOCHS + 1, 2))
    epoch_F1_loss = np.zeros(N_EPOCHS + 1)
    best_F1 = 0
    split = 4
    kf = KFold(n_splits=split)
    for epoch in range(1, N_EPOCHS):
        print("========================================================")
        print("epoch: ", epoch)
        loss_tr = 0
        loss_val = 0
        er = 0
        fscore = 0
        get_sed = []
        ft_sed = []
        get_sed_lt = []
        ft_sed_lt = []
        sed_pred = []
        sed_gt = []
        start = time.time()
        for i, (train_index, val_index) in enumerate(kf.split(trainset)):
            train_fold = torch.utils.data.dataset.Subset(trainset, train_index)
            val_fold = torch.utils.data.dataset.Subset(trainset, val_index)
            train_loader = DataLoader(dataset=train_fold, batch_size=BATCH_SIZE, shuffle=False)
            val_loader = DataLoader(dataset=val_fold, batch_size=BATCH_SIZE, shuffle=False)
            tr_los, val_los, get_sed, ft_sed, get_sed_lt, ft_sed_lt = trainModel()
            sed_pred = np.vstack((reshape_3Dto2D(np.array(get_sed)) > threshold, get_sed_lt > threshold))
            sed_gt = np.vstack((reshape_3Dto2D(np.array(ft_sed)) > threshold, ft_sed_lt > threshold))
            er_seg, fscore_seg = compute_sed_scores(sed_pred, sed_gt)
            loss_tr += tr_los
            loss_val += val_los
            er += er_seg
            fscore += fscore_seg
        loss_tr /= split
        loss_val /= split
        er /= split
        fscore /= split
        train_loss[epoch] = loss_tr
        val_loss[epoch] = loss_val
        sed_loss[epoch] = (er, fscore * 100)
        print("save model!")
        torch.save(model.state_dict(), SavePath + "model_" + str(epoch) + ".pth")
        print("save train_loss!")
        np.save(SavePath + "train_loss_" + str(epoch) + ".npy", np.array(train_loss))
        print("save val_loss!")
        np.save(SavePath + "val_loss_" + str(epoch) + ".npy", np.array(val_loss))
        print("save score!")
        np.save(SavePath + "sed_score_" + str(epoch) + ".npy", np.array(sed_loss))
        if fscore >= best_F1:
            best_F1 = fscore
            best_epoch = epoch
        print(
            'epoch: %d, time: %.2fs, tr_loss: %f, val_loss: %f, '
            'F1_score: %.1f, ER: %.2f, ' %
            (
                epoch, time.time() - start, train_loss[epoch], val_loss[epoch],
                sed_loss[epoch, 1], sed_loss[epoch, 0]
            )
        )
        print('Best epoch:', best_epoch)
        print('Best F1:', best_F1 * 100)

