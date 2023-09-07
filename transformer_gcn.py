import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
import math
from torch.autograd import Variable
import scipy
import pickle
import os.path
from scipy import io
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import sys


use_cuda = torch.cuda.is_available()


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(float(TP), float(FP), float(TN), float(FN))

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.squeeze(input)
        adj = torch.squeeze(adj)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BrainNetCNN(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(BrainNetCNN, self).__init__()
        self.gc1 = GraphConvolution(246,100)
        self.gc2 = GraphConvolution(100,10)

        self.trans = torch.nn.Transformer(d_model=2460,nhead=4,num_encoder_layers=2,num_decoder_layers=2)
        self.temporalFC1 = torch.nn.Linear(2460,500)
        self.temporalFC2 = torch.nn.Linear(500,1)

        self.sm = torch.nn.Softmax(dim=1)

        self.fc_class = torch.nn.Linear(10,2,bias=False)
        self.fc1 = torch.nn.Linear(246,60)
        self.fc2 = torch.nn.Linear(60,1)



    def forward(self, x, adj):

        intermediate = torch.empty(1,246,10,15)
        for i in range(15):
            map = x[:,:,:,:,i]
            out = F.leaky_relu(self.gc1(map, adj),negative_slope=0.1)
            out = F.leaky_relu(self.gc2(out, adj), negative_slope=0.1)
            intermediate[:,:,:,i] = out
        trans_reshape = intermediate.view(intermediate.size(0),intermediate.size(3),intermediate.size(1)*intermediate.size(2))
        Attn = self.trans(trans_reshape,trans_reshape)

        Attn = F.leaky_relu(self.temporalFC1(Attn),negative_slope=0.1)
        Attn = F.leaky_relu(self.temporalFC2(Attn), negative_slope=0.1)

        Attn = self.sm(Attn)
        Attn = torch.squeeze(Attn)

        classifier_in = intermediate.view(intermediate.size(0),intermediate.size(1),intermediate.size(3),intermediate.size(2))
        out = F.leaky_relu(self.fc_class(classifier_in), negative_slope=0.1)
        out = out.view(out.size(0),out.size(1),out.size(3),out.size(2))
        out = torch.matmul(out,Attn)
        out = torch.squeeze(out)

        ann_out = F.leaky_relu(self.fc1(out.view(out.size(1),out.size(0))),negative_slope=0.1)
        ann_out = F.leaky_relu(self.fc2(ann_out),negative_slope=0.1)

        out = torch.add(out,ann_out.view(ann_out.size(1),ann_out.size(0)))


        return out, ann_out


import torch.utils.data.dataset


class GoldMSI_LSD_Dataset_test(torch.utils.data.Dataset):
    def __init__(self, transform=False, class_balancing=False, index=0):
        self.transform = transform

        stringX = 'x_d' + str(index) + '.mat'
        X_str = 'x_d'

        stringA = 'A_GCN.mat'
        A_str = 'A_GCN'

        stringY = 'y' + str(index) + '.mat'
        Y_str = 'y'

        x = scipy.io.loadmat(stringX, mdict=None)
        x = x[X_str]

        y = scipy.io.loadmat(stringY, mdict=None)
        y = y[Y_str]

        A = scipy.io.loadmat(stringA, mdict=None)
        A = A[A_str]

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.A = torch.FloatTensor(np.expand_dims(A, 1).astype(np.float32))
        self.Y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx],self.A[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


class GoldMSI_LSD_Dataset_train(torch.utils.data.Dataset):
    def __init__(self, transform=False, class_balancing=False, index=0, fold=1):
        self.transform = transform

        folds_index = range(fold)
        folds_index = [x + 1 for x in folds_index]
        del [folds_index[index - 1]]
        count = 0
        for i in folds_index:
            stringX = 'x_d' + str(i) + '.mat'
            X_str = 'x_d'

            stringA = 'A_GCN.mat'
            A_str = 'A_GCN'

            stringY = 'y' + str(i) + '.mat'
            Y_str = 'y'

            A_partial = scipy.io.loadmat(stringA, mdict=None)
            A_partial = A_partial[A_str]

            x_partial = scipy.io.loadmat(stringX, mdict=None)
            x_partial = x_partial[X_str]

            y_partial = scipy.io.loadmat(stringY, mdict=None)
            y_partial = y_partial[Y_str]


            if count == 0:
                x = x_partial
                A = A_partial
                y = y_partial

            else:
                x = np.concatenate((x, x_partial), axis=0)
                A = np.concatenate((A,A_partial), axis=0)
                y = np.concatenate((y, y_partial), axis=0)

            count = count + 1


        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.A = torch.FloatTensor(np.expand_dims(A, 1).astype(np.float32))
        self.Y = torch.tensor(y, dtype=torch.long)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.A[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


total_pred = []
total_acc = []
total_auc = []
total_sens = []
total_spec = []
total_precision = []

test_index = int(sys.argv[1])
lr = float(sys.argv[2])
nbepochs = int(sys.argv[3])
BATCH_SIZE = int(sys.argv[4])
class_0 = float(sys.argv[5])
class_1 = float(sys.argv[6])
Alpha = float(sys.argv[7])


trainset = GoldMSI_LSD_Dataset_train(index=test_index, fold=14)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
testset = GoldMSI_LSD_Dataset_test(index=test_index)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

net = BrainNetCNN(trainset.X)
if use_cuda:
    net = net.cuda(0)
    net = torch.nn.DataParallel(net(), device_ids=[0])

momentum = 0.9
wd = 0.00005  ## Decay for L2 regularization


def init_weights_he(m):

    if type(m) == torch.nn.Linear:
        fan_in = net.dense1.in_features
        he_lim = np.sqrt(6) / fan_in
        m.weight.data.uniform_(-he_lim, he_lim)
        print(m.weight)


class_weight = torch.FloatTensor([class_0, class_1])
criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
optimizer2 = torch.optim.Adam(net.parameters(), lr=lr)

def train(epoch, alpha=1.0,idx=1.0):
    net.train()
    print(epoch)
    for batch_idx, (X,A,Y) in enumerate(trainloader):
        if use_cuda:
            X,A, Y = X.cuda(),A.cuda(), Y.cuda()
        optimizer2.zero_grad()

        X, A, Y = Variable(X),Variable(A), Variable(Y)
        out, ann_out = net(X,A)

        Y = Y.view(Y.size(0) * Y.size(1), 1)
        Y = np.squeeze(Y)
        Y = Variable(Y)

        loss_backprop = torch.zeros((1, 1), requires_grad=True)
        loss = criterion((out), Y)

        total_loss.append(loss)
        loss_epilepsy = torch.zeros((1, 1), requires_grad=True)
        for i in range(246):
            if Y[i] == 1:
                loss_epilepsy = torch.add(-out[i,1], loss_epilepsy)

        asym_string = "Contralateral loss: " +str(alpha*torch.mean(loss_epilepsy))
        cross_entropy_string = "Crossentropy loss: " +str(loss)
        #print(cross_entropy_string)
        #print(asym_string)

        loss = loss + alpha * torch.mean(loss_epilepsy)
        loss.backward()
        #print(loss)
        optimizer2.step()
    return loss_backprop

def test(alpha=1.0):
    net.eval()
    test_loss = 0
    running_loss = 0.0

    total_out = []

    for batch_idx, (X,A, Y) in enumerate(testloader):

        if use_cuda:
            X, A, Y = X.cuda(), A.cuda(), Y.cuda()

        with torch.no_grad():
            if use_cuda:
                X,A, Y = X.cuda(), A.cuda(), Y.cuda()
            optimizer.zero_grad()
            X, A, Y = Variable(X), Variable(A), Variable(Y)
            out, ann_out = net(X, A)
            out = out.cpu()
            out = out.data.numpy()
            total_out.append(out)


    return total_out, ann_out

cont = scipy.io.loadmat('BNA_cont.mat', mdict=None)
cont = cont['cont']
sparse_loss = []
class_loss = []
total_loss = []
for epoch in range(nbepochs):
    train(epoch,alpha=Alpha,idx=cont)

#torch.save(net.state_dict(),'/home/naresh/naresh/PycharmProjects/Epilepsy_BNA_MNI/DeepEZ.pkl')
#filter1 = net.fc2.weight.data.numpy()

stringY = 'y' + str(test_index) + '.mat'
Y_str = 'y'
y = scipy.io.loadmat(stringY, mdict=None)
y = y[Y_str]

out, ann_out = test()
out = np.array(out)
out = np.squeeze(out)
pred = np.argmax(out, 1)
mystring = "/home/nnandak1/scratch4-avenka14/Dynamic_EZ_localization/Outputs/fold"+str(test_index)+"delta"+str(class_1)+"outputs.pickle"
f = open(mystring, "wb")
pickle.dump(out, f)
pickle.dump(pred, f)
f.close()

