import logging
from collections import OrderedDict
import paddle 
import paddle.nn as nn
from paddle.distributed import fleet
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss,class_loss_3class,average_loss_3class
from models.archs import arch_util
import cv2
import numpy as np
from utils import util
from data import util as ut
import os.path as osp
import os

logger = logging.getLogger('base')


class ClassSR_Model(BaseModel):
    def __init__(self, opt):
        super(ClassSR_Model, self).__init__(opt)

        self.patch_size = int(opt["patch_size"])
        self.step = int(opt["step"])
        self.scale = int(opt["scale"])
        self.name = opt['name']
        self.which_model = opt['network_G']['which_model_G']


        if opt['dist']:
            self.rank = 2
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        # TODO: 切换设备
        # self.netG = networks.define_G(opt).to(self.device)
        self.netG = networks.define_G(opt)

        if opt['dist']:
            self.netG = fleet.distributed_model(self.netG)
        # print network
        self.print_network()
        # TODO: change no load
        # self.load()

        if self.is_train:
            self.l1w = float(opt["train"]["l1w"])
            self.class_loss_w = float(opt["train"]["class_loss_w"])
            self.average_loss_w = float(opt["train"]["average_loss_w"])
            self.pf = opt['logger']['print_freq']
            self.batch_size = int(opt['datasets']['train']['batch_size'])
            self.netG.train()

            # loss
            # TODO: remove to device
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss()#.to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss()#.to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss()#.to(self.device)
            elif loss_type == 'ClassSR_loss':
                self.cri_pix = nn.L1Loss()#.to(self.device)
                self.class_loss = class_loss_3class()#.to(self.device)
                self.average_loss = average_loss_3class()#.to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0.
            optim_params = []
            if opt['fix_SR_module']:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    # if v.requires_grad and "class" not in k:
                    if not v.stop_gradient and "class" not in k:
                        # v.requires_grad=False
                        v.stop_gradient = True

            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                # if v.requires_grad:
                if not v.stop_gradient:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            # schedulers
            # TODO: scheduler
            # if train_opt['lr_scheme'] == 'MultiStepLR':
            #     for optimizer in self.optimizers:
            #         self.schedulers.append(
            #             lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
            #                                              restarts=train_opt['restarts'],
            #                                              weights=train_opt['restart_weights'],
            #                                              gamma=train_opt['lr_gamma'],
            #                                              clear_state=train_opt['clear_state']))
            if train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingDecay(train_opt['lr_G'],
                        train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
                    #optimizer._learning_rate = self.schedulers[-1]
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            # TODO: adam 参数设置
            self.optimizer_G = paddle.optimizer.Adam(learning_rate=self.schedulers[0], parameters=optim_params,
                                                     weight_decay=wd_G,
                                                     beta1=train_opt['beta1'], beta2=train_opt['beta2'])
            if opt['dist']:
                self.optimizer_G = fleet.distributed_optimizer(self.optimizer_G)
            self.optimizers.append(self.optimizer_G)

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        # TODO: to device
        self.var_L = data['LQ']#.to(self.device)
        self.LQ_path = data['LQ_path'][0]
        if need_GT:
            self.real_H = data['GT']#.to(self.device)  # GT
            self.GT_path = data['GT_path'][0]


    def optimize_parameters(self, step):
        self.optimizer_G.clear_grad()
        self.fake_H, self.type = self.netG(self.var_L, self.is_train)
        #print(self.type)
        l_pix = self.cri_pix(self.fake_H, self.real_H)
        class_loss=self.class_loss(self.type)
        average_loss=self.average_loss(self.type)
        loss = self.l1w * l_pix + self.class_loss_w * class_loss+self.average_loss_w*average_loss

        if step % self.pf == 0:
           self.print_res(self.type)

        loss.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['class_loss'] = class_loss.item()
        self.log_dict['average_loss'] = average_loss.item()
        self.log_dict['loss'] = loss.item()

    def test(self):
        self.netG.eval()
        self.var_L = cv2.imread(self.LQ_path, cv2.IMREAD_UNCHANGED)
        self.real_H = cv2.imread(self.GT_path, cv2.IMREAD_UNCHANGED)

        lr_list, num_h, num_w, h, w = self.crop_cpu(self.var_L, self.patch_size, self.step)
        gt_list=self.crop_cpu(self.real_H,self.patch_size*4,self.step*4)[0]
        sr_list = []
        index = 0

        psnr_type1 = 0
        psnr_type2 = 0
        psnr_type3 = 0

        for LR_img,GT_img in zip(lr_list,gt_list):

            if self.which_model=='classSR_3class_rcan':
                img = LR_img.astype(np.float32)
            else:
                img = LR_img.astype(np.float32) / 255.
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            # some images have 4 channels
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img = img[:, :, [2, 1, 0]]
            img = paddle.to_tensor(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).astype('float32').unsqueeze(0)
            with paddle.no_grad():
                srt, type = self.netG(img, False)

            if self.which_model == 'classSR_3class_rcan':
                sr_img = util.tensor2img(srt.detach()[0].astype('float32'), out_type=np.uint8, min_max=(0, 255))
            else:
                sr_img = util.tensor2img(srt.detach()[0].astype('float32'))
            sr_list.append(sr_img)

            if index == 0:
                type_res = type
            else:
                type_res = paddle.concat((type_res, type), 0)

            psnr=util.calculate_psnr(sr_img, GT_img)
            flag=paddle.argmax(type, 1).squeeze().numpy()
            if flag == 0:
                psnr_type1 += psnr
            if flag == 1:
                psnr_type2 += psnr
            if flag == 2:
                psnr_type3 += psnr

            index += 1

        self.fake_H = self.combine(sr_list, num_h, num_w, h, w, self.patch_size, self.step)
        if self.opt['add_mask']:
            self.fake_H_mask = self.combine_addmask(sr_list, num_h, num_w, h, w, self.patch_size, self.step,type_res)
        self.real_H = self.real_H[0:h * self.scale, 0:w * self.scale, :]
        self.num_res = self.print_res(type_res)
        self.psnr_res=[psnr_type1,psnr_type2,psnr_type3]


        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L
        out_dict['rlt'] = self.fake_H
        out_dict['num_res'] = self.num_res
        out_dict['psnr_res']=self.psnr_res
        if need_GT:
            out_dict['GT'] = self.real_H
        if self.opt['add_mask']:
            out_dict['rlt_mask']=self.fake_H_mask
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        # if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
        #     net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
        #                                      self.netG.module.__class__.__name__)
        # else:
        net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n.item()))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_classifier = self.opt['path']['pretrain_model_classifier']
        load_path_G_branch3 = self.opt['path']['pretrain_model_G_branch3']
        load_path_G_branch2= self.opt['path']['pretrain_model_G_branch2']
        load_path_G_branch1 = self.opt['path']['pretrain_model_G_branch1']
        load_path_Gs=[load_path_G_branch1,load_path_G_branch2,load_path_G_branch3]
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        if load_path_classifier is not None:
            logger.info('Loading model for classfier [{:s}] ...'.format(load_path_classifier))
            self.load_network_classifier_rcan(load_path_classifier, self.netG, self.opt['path']['strict_load'])
        if load_path_G_branch3 is not None and load_path_G_branch1 is not None and load_path_G_branch2 is not None:
            logger.info('Loading model for branch1 [{:s}] ...'.format(load_path_G_branch1))
            logger.info('Loading model for branch2 [{:s}] ...'.format(load_path_G_branch2))
            logger.info('Loading model for branch3 [{:s}] ...'.format(load_path_G_branch3))
            self.load_network_classSR_3class(load_path_Gs, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def crop_cpu(self,img,crop_sz,step):
        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)
        index = 0
        num_h = 0
        lr_list=[]
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                lr_list.append(crop_img)
        h=x + crop_sz
        w=y + crop_sz
        return lr_list,num_h, num_w,h,w

    def combine(self,sr_list,num_h, num_w,h,w,patch_size,step):
        index=0
        sr_img = np.zeros((h*self.scale, w*self.scale, 3), 'float32')
        for i in range(num_h):
            for j in range(num_w):
                sr_img[i*step*self.scale:i*step*self.scale+patch_size*self.scale,j*step*self.scale:j*step*self.scale+patch_size*self.scale,:]+=sr_list[index]
                index+=1
        sr_img=sr_img.astype('float32')

        for j in range(1,num_w):
            sr_img[:,j*step*self.scale:j*step*self.scale+(patch_size-step)*self.scale,:]/=2

        for i in range(1,num_h):
            sr_img[i*step*self.scale:i*step*self.scale+(patch_size-step)*self.scale,:,:]/=2
        return sr_img

    def combine_addmask(self, sr_list, num_h, num_w, h, w, patch_size, step, type):
        index = 0
        sr_img = np.zeros((h * self.scale, w * self.scale, 3), 'float32')

        for i in range(num_h):
            for j in range(num_w):
                sr_img[i * step * self.scale:i * step * self.scale + patch_size * self.scale,
                j * step * self.scale:j * step * self.scale + patch_size * self.scale, :] += sr_list[index]
                index += 1
        sr_img = sr_img.astype('float32')

        for j in range(1, num_w):
            sr_img[:, j * step * self.scale:j * step * self.scale + (patch_size - step) * self.scale, :] /= 2

        for i in range(1, num_h):
            sr_img[i * step * self.scale:i * step * self.scale + (patch_size - step) * self.scale, :, :] /= 2

        index2 = 0
        for i in range(num_h):
            for j in range(num_w):
                # add_mask
                alpha = 1
                beta = 0.2
                gamma = 0
                bbox1 = [j * step * self.scale + 8, i * step * self.scale + 8,
                         j * step * self.scale + patch_size * self.scale - 9,
                         i * step * self.scale + patch_size * self.scale - 9]  # xl,yl,xr,yr
                zeros1 = np.zeros((sr_img.shape), 'float32')

                if paddle.argmax(type, 1)[1].squeeze()[index2] == 0:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                      color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1),
                                         color=(0, 255, 0), thickness=-1)# simple green
                elif paddle.argmax(type, 1).squeeze()[index2] == 1:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                       color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1),
                                          color=(0, 255, 255), thickness=-1)# medium yellow
                elif paddle.argmax(type, 1).squeeze()[index2] == 2:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                       color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1),
                                          color=(0, 0, 255), thickness=-1)# hard red

                sr_img = cv2.addWeighted(sr_img, alpha, mask2, beta, gamma)
                # sr_img = cv2.addWeighted(sr_img, alpha, mask1, 1, gamma)
                index2+=1
        return sr_img

    def print_res(self, type_res):
        num0 = 0
        num1 = 0
        num2 = 0

        for i in paddle.argmax(type_res, 1).squeeze():
            if i == 0:
                num0 += 1
            if i == 1:
                num1 += 1
            if i == 2:
                num2 += 1

        return [num0, num1,num2]
