import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

eps = np.finfo(np.float).eps

Overlap_num = 'ov1'
Feat_name = 'MFCC+GFCC'
path = "G:/SED数据集无噪/" + Overlap_num + '/' + Feat_name + '/'
N_EPOCH = 99
# 四折  SAI 77.6 0.25  GFCC 78.1 0.25  yuanSAI 78.1 0.26
# ov3 SAI 56.0 0.49  GFCC 60.6 0.43  yuanNAP 78.2 0.247(99.1 0.01)
def testModel():
    # model.eval()
    # print("testing trained model...")
    test_loss_sum = 0
    get_sed = []
    fact_sed = []
    with torch.no_grad():
        for i, (paras, sed_label) in enumerate(test_loader):
            paras = paras.to(device)
            paras = paras.float()
            sed_label = sed_label.to(device)
            # model.eval()
            out_sed = model(paras)
            # out_sed = out_sed.reshape(-1, 8)
            loss = criterion_sed(out_sed.double(), sed_label.double())
            test_loss_sum += loss.item()

            if i < len(test_loader) - 1:
                # print(np.array(out_sed.cuda().cpu()).shape)  # [512,8] [178,8]
                get_sed.append(np.array(out_sed.cuda().cpu()))
                fact_sed.append(np.array(sed_label.cuda().cpu()))

            else:
                get_sed_last = np.array(out_sed.cuda().cpu())
                fact_sed_last = np.array(sed_label.cuda().cpu())
    return test_loss_sum / len(test_loader), get_sed, fact_sed, get_sed_last, fact_sed_last


class PitayaFan(Dataset):
    def __init__(self, mode, overlap_num, feat_name):
        self.spec_path = path + 'feat/'
        self.label_path = path + 'label/'
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
                    self.spec.append(np.load(self.spec_path + name))
                    self.label.append(np.load(self.label_path + filename + '.npy'))
        else:
            if self.feat_name == 'NAP' or self.feat_name == 'NAP改进':
                for name in test_file:
                    filename = name.split('.')[0]
                    spec1 = np.loadtxt(self.spec_path + name).transpose()
                    self.spec.append(spec1)
                    self.label.append(np.load(self.label_path + filename + '.npy'))
            else:
                for name in test_file:
                    filename = name.split('.')[0]
                    spec1 = np.load(self.spec_path + name)
                    # SAI加这一行
                    # spec1[spec1 < 0] = 0
                    self.spec.append(spec1)
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


if Feat_name == '幅度谱':
    num = 16
elif '+' in Feat_name:
    num = 13
else:
    num = 6
model = PitayaSELD(Q=num)

BATCH_SIZE = 256
testset = PitayaFan(mode='test', overlap_num=Overlap_num, feat_name=Feat_name)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)
print(model)
criterion_sed = torch.nn.BCEWithLogitsLoss()
threshold = 0.5
best_epoch = 0
best_fscore = 0
loss_test = []
for epoch in range(N_EPOCH):
    epoch += 1
    print("================================================================")
    val_sed_score = np.load(path + 'sed_score_' + str(epoch) + '.npy')
    train_loss = np.load(path + 'train_loss_' + str(epoch) + '.npy')
    val_loss = np.load(path + 'val_loss_' + str(epoch) + '.npy')
    model.load_state_dict(torch.load(path + 'model_' + str(epoch) + '.pth'))
    print("epoch", epoch)
    print(path + 'model_' + str(epoch) + '.pth')
    print("val_fscore", val_sed_score[epoch][1])
    print("val_er", val_sed_score[epoch][0])
    test_loss, get_sed_ls, ft_sed_ls, get_sed_last, fact_sed_last = testModel()
    loss_test.append(test_loss)
    sed_pred = np.vstack((reshape_3Dto2D(np.array(get_sed_ls)) > threshold, get_sed_last > threshold))
    sed_gt = np.vstack((reshape_3Dto2D(np.array(ft_sed_ls)) > threshold, fact_sed_last > threshold))
    er, f_score = compute_sed_scores(sed_pred, sed_gt)
    print("test_fscore:", f_score * 100)
    print("test_er:", er)
    if f_score >= best_fscore:
        best_fscore = f_score
        best_epoch = epoch
    print("best_epoch:", best_epoch)
    print("bast_fscore:", best_fscore * 100)

val_er = []
val_score = []
for i, (er, f_score) in enumerate(val_sed_score):
    if i < N_EPOCH:
        val_er.append(er)
        val_score.append(f_score)

plt.figure(1)
plt.plot(val_er)
plt.title("ER")
plt.show()
plt.figure(2)
plt.plot(val_score)
plt.title("Fscore")
plt.show()

plt.figure(3)
plt.title("loss")
plt.plot(train_loss[:N_EPOCH])
plt.plot(val_loss[:N_EPOCH])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(['train_loss', 'eval_loss'])
plt.show()

plt.figure(5)
plt.title("loss")
plt.plot(loss_test)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()