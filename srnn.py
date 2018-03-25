import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

from torch.autograd import Variable as Var

import pretrainedmodels

class SRnnCell(nn.Module):
    def init_para(self):
        self.U.weight.data.copy_(torch.eye(self.state_dim,self.state_dim))  #identity matrix, equivalent to pre-linear average
        if self.srnn_type in [2,3]:
            self.Wz.weight.data.zero_()
            self.Uz.weight.data.zero_()
        if self.srnn_type == 3:
            self.Wr.weight.data.zero_()
            self.Ur.weight.data.zero_()

    def __init__(self,input_dim, state_dim, srnn_type, input_ln_4init=None):
        super(SRnnCell, self).__init__()
        # const
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.srnn_type = srnn_type
        # components
        self.U = nn.Linear(state_dim, state_dim, bias=False)  # Uh
        if self.srnn_type in [2,3]:
            self.Wz = nn.Linear(input_dim, state_dim, bias=False)  # forget gate: z_t =  Sigmoid( W_z*CNN(x) + U_z*h_{t-1} )
            self.Uz = nn.Linear(state_dim, state_dim, bias=False)

        if self.srnn_type == 3:
            self.Wr = nn.Linear(input_dim, state_dim, bias=False)  # reset gate: z_r =  Sigmoid( W_r*CNN(x) + U_r*h_{t-1} )
            self.Ur = nn.Linear(state_dim, state_dim, bias=False)

#        self.input_ln = nn.Linear(input_dim, state_dim) if input_ln_4init==None else input_ln_4init
        self.input_ln = input_ln_4init
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # init
        self.init_para()

# vanilla srnn cell forward
    def forward_vanilla(self,s,x):
        xs = self.input_ln(x) if self.input_ln!=None else x
        ss = self.U(s)
        s = self.relu(xs+ss)
        return s

# half-gru forward
    def forward_hgru(self, ht_1, x):  # ht_1 is previous state
        xs = self.input_ln(x) if self.input_ln!=None else x
        h_tp = self.relu(self.U(ht_1) + xs)
        z_t = self.sigmoid(self.Uz(ht_1) + self.Wz(xs))  # forget gate
        h_t = ht_1.mul(1-z_t) + h_tp.mul(z_t)  # mixed current state
        return h_t

# full gru forward
    def forward_gru(self, ht_1, x):  # ht_1 is previous state
        xs = self.input_ln(x) if self.input_ln!=None else x
        z_r = self.sigmoid(self.Ur(ht_1) + self.Wr(xs))  # reset gate
        h_tp = self.relu(self.U(ht_1.mul(z_r)) + xs)
        z_t = self.sigmoid(self.Uz(ht_1) + self.Wz(xs))  # forget gate
        h_t = ht_1.mul(1-z_t) + h_tp.mul(z_t)  # mixed current state
        return h_t

# srnn cell forward
    def forward(self, ht_1, x):
        if self.srnn_type==1:
            return self.forward_vanilla(ht_1, x)
        elif self.srnn_type==2:
            return self.forward_hgru(ht_1, x)
        elif self.srnn_type==3:
            return self.forward_gru(ht_1, x)
        else:
            assert(0)

# the following is for resnet only (obsolete)
def basenet_forward_resnet(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = x.view(x.size(0),x.size(1),-1)
    x = x.mean(2)
    return x

# the following uses pretrainedmodels's settings
def basenet_forward(self, x):
# get feature planes
    x = self.features(x)
# global average pooling
    x = x.view(x.size(0),x.size(1),-1)
    x = x.mean(2)
    return x

class SRnn(nn.Module):
    def _get_adapted_basenet(self, basenet, pretrained_base):
        if pretrained_base:
            net = pretrainedmodels.__dict__[basenet](num_classes=1000, pretrained='imagenet')
        else:
            net = pretrainedmodels.__dict__[basenet](num_classes=1000)

        if hasattr(net, 'fc') and net.fc!=None:  # only has fc (torchvision models)
            last_fc = net.fc
            del net.fc
        elif hasattr(net, 'last_linear'): # last_linear (pretrainedmodels)
            last_fc = net.last_linear
            if hasattr(net, 'fc'): del net.fc
            del net.last_linear
        elif hasattr(net, 'classifier'): # no fc and last_linear, but has classifier (dpn model in pretrainedmodels)
            wm_size = net.classifier.weight.size()
            last_fc = nn.Linear(wm_size[1], wm_size[0])
            last_fc.weight.data.copy_(net.classifier.weight.squeeze().data)
            last_fc.bias.data.copy_(net.classifier.bias.data)
            del net.classifier
        else:
            print('No fc layer found in the basenet');
            assert(0)

        net.__class__.forward = basenet_forward
        return net,last_fc

    def __init__(self, basenet, pretrained_base=False,train_cnn=False, mode=0, single_scale=0):
        super(SRnn, self).__init__()
        self.base_net,last_fc = self._get_adapted_basenet(basenet, pretrained_base)
        in_dim,out_dim = last_fc.in_features,last_fc.out_features
        self.feat_dim = in_dim
        self.state_dim = in_dim
        self.train_cnn = train_cnn
        self.single_scale = single_scale

        srnn_type = 0
        if mode==3: srnn_type = 1
        elif mode==4: srnn_type = 2
        elif mode==5: srnn_type = 3
        self.mode = mode

        self.srnn_cell = SRnnCell(self.feat_dim,self.state_dim,srnn_type,None)
        self.fc = last_fc
        self.softmax = nn.Softmax()

    def get_basenet(self):
        return self.base_net

    def get_and_remove_basenet(self):
        basenet = self.base_net
        del self.base_net   # de-register the basenet
        return basenet

# set to train or eval mode
    def train(self, mode=True):
        self.srnn_cell.train()
        self.fc.train()
        if mode and self.train_cnn:
            if 'base_net' in self.__dict__['_modules']:
                self.base_net.train()
        else:
            if 'base_net' in self.__dict__['_modules']:
                print('Set basenet to eval mode');
                self.base_net.eval()

# single scale
    def forward_single_scale(self, x_arr):
        return self.fc(self.base_net(x_arr[self.single_scale]))

# pre-softmax (logit ensemble, or vanilla srnn with idenity state transition)
    def forward_presoftmax_scale_ensemble(self, x_arr):
        batch = x_arr[0].size(0)
        s = torch.zeros(batch, 1000)
        s = s.type(x_arr[0].data.type())
        s = Var(s)
        for i,x in enumerate(x_arr):
            x.volatile = True
            s += self.fc(self.base_net(x))
        s.div_(len(x_arr))
        return s

# post-softmax (probability ensemble)
    def forward_postsoftmax_scale_ensemble(self, x_arr):
        batch = x_arr[0].size(0)
        s = torch.zeros(batch, 1000)
        s = s.type(x_arr[0].data.type())
        s = Var(s)
        for i,x in enumerate(x_arr):
            x.volatile = True
            s += self.softmax(self.fc(self.base_net(x)))
        s.div_(len(x_arr))
        return s

# srnn -
    def forward_srnn(self, x_arr):
        batch = x_arr[0].size(0)
        s = torch.zeros(1,self.state_dim)
        s = s.type(x_arr[0].data.type())
        s = Var(s)
        for x in x_arr:
            if 'base_net' in self.__dict__['_modules']:   # if there is basenet
                x = self.base_net(x)
            s = self.srnn_cell(s,x)
        return self.fc(s)

# forward function
    def forward(self, x_arr):
        if self.mode==0:
            return self.forward_single_scale(x_arr)
        elif self.mode==1:
            return self.forward_postsoftmax_scale_ensemble(x_arr)
        elif self.mode==2:
            return self.forward_presoftmax_scale_ensemble(x_arr)
        elif self.mode==3:
            return self.forward_srnn(x_arr)
        elif self.mode==4:
            return self.forward_srnn(x_arr)
        elif self.mode==5:
            return self.forward_srnn(x_arr)
        else:
            print('Mode not supported !')
            asset(0)

if __name__=='__main__':
    pass
