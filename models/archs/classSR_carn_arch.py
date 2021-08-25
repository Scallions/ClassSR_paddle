import functools
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.CARN_arch import CARN_M
import numpy as np
import time

class ClassSR(nn.Layer):
    def __init__(self, in_nc=3, out_nc=3):
        super(ClassSR, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = CARN_M(in_nc=in_nc, out_nc=out_nc, nf=36, scale=4, multi_scale=False, group=4)
        self.net2 = CARN_M(in_nc=in_nc, out_nc=out_nc, nf=52, scale=4, multi_scale=False, group=4)
        self.net3 = CARN_M(in_nc=in_nc, out_nc=out_nc, nf=64, scale=4, multi_scale=False, group=4)

    def forward(self, x,is_train):
        if is_train:
            for i in range(len(x)):
                # print(x[i].unsqueeze(0).shape)
                type = self.classifier(x[i].unsqueeze(0))
                p = F.softmax(type, axis=1)
                p1 = p[0][0]
                p2 = p[0][1]
                p3 = p[0][2]

                out1 = self.net1(x[i].unsqueeze(0))
                out2 = self.net2(x[i].unsqueeze(0))
                out3 = self.net3(x[i].unsqueeze(0))

                out = out1 * p1 + out2 * p2 + out3 * p3
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = paddle.concat((out_res, out), 0)
                    type_res = paddle.concat((type_res, p), 0)
        else:

            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))

                flag = paddle.max(type, 1)[1].squeeze()
                p = F.softmax(type, axis=1)
                #flag=np.random.randint(0,2)
                #flag=2
                if flag == 0:
                    out = self.net1(x[i].unsqueeze(0))
                elif flag==1:
                    out = self.net2(x[i].unsqueeze(0))
                elif flag==2:
                    out = self.net3(x[i].unsqueeze(0))
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = paddle.concat((out_res, out), 0)
                    type_res = paddle.concat((type_res, p), 0)

            return out_res, type_res

        return out_res,type_res

class Classifier(nn.Layer):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2D(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2D(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2D(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2D(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2D(128, 32, 1))
        arch_util.initialize_weights([self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2D(out.shape[2])(out)
        out = out.reshape((out.shape[0], -1))
        out = self.lastOut(out)
        return out


