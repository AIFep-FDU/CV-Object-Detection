import os
import torch

import os
import numpy as np
import random

from uutils.augment import SSDAugmentation, SSDBaseTransform
from uutils.dataset import build_dataset, build_dataloader, CollateFunc
from uutils.utils import build_optimizer, build_lambda_lr_scheduler
from evaluator import build_evluator


def build_transform(args, is_train=False):
    ## SSD style transform
    if is_train:
        transform = SSDAugmentation(args.img_size)
    else:
        transform = SSDBaseTransform(args.img_size)
    return transform

## Trainer for general YOLO series
class YoloTrainer(object):
    def __init__(self, args, data_cfg, model_cfg, trans_cfg, device, model, criterion):
        # ------------------- basic parameters -------------------
        self.args = args
        self.epoch = 0
        self.best_map = -1.
        self.device = device
        self.criterion = criterion
        self.grad_accumulate = args.grad_accumulate
        self.clip_grad = 35
        self.heavy_eval = False

        # path to save model
        self.path_to_save = args.save_folder
        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- Hyperparameters refer to RTMDet ----------------------------
        self.optimizer_dict = {'optimizer': 'adamw', 'momentum': None, 'weight_decay': 5e-2, 'lr0': 0.001}
        self.lr_schedule_dict = {'scheduler': 'linear', 'lrf': 0.01}
        self.warmup_dict = {'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1}        

        # ---------------------------- Build Dataset & Model & Trans. Config ----------------------------
        self.data_cfg, self.model_cfg, self.trans_cfg  = data_cfg, model_cfg, trans_cfg

        # ---------------------------- Build Transform ----------------------------
        self.train_transform = build_transform( args=args, is_train=True )
        self.val_transform = build_transform( args=args, is_train=False )

        # ---------------------------- Build Dataset & Dataloader ----------------------------
        self.dataset, self.dataset_info = build_dataset(args, 'train', self.data_cfg, self.train_transform, is_train=True)
        self.train_loader = build_dataloader(args, self.dataset, self.args.batch_size, CollateFunc())

        # ---------------------------- Build Evaluator ----------------------------
        self.evaluator = build_evluator(args, 'val', self.val_transform, self.device)

        # ---------------------------- Build Grad. Scaler ----------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

        # ---------------------------- Build Optimizer ----------------------------
        self.optimizer_dict['lr0'] *= args.batch_size * self.grad_accumulate / 64
        self.optimizer, self.start_epoch = build_optimizer(self.optimizer_dict, model)

        # ---------------------------- Build LR Scheduler ----------------------------
        self.lr_scheduler, self.lf = build_lambda_lr_scheduler(self.lr_schedule_dict, self.optimizer, args.max_epoch)
        self.lr_scheduler.last_epoch = self.start_epoch - 1  # do not move



    def train(self, model):
        for epoch in range(self.start_epoch, self.args.max_epoch):
            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # eval one epoch
            model_eval = model
            if (epoch % self.args.eval_epoch) == 0 or (epoch == self.args.max_epoch - 1):
                self.eval(model_eval)


    def eval(self, model):
        # chech model
        model_eval = model 

        print('eval ...')
        # set eval mode
        model_eval.trainable = False
        model_eval.eval()

        # evaluate
        with torch.no_grad():
            self.evaluator.evaluate(model_eval)

        # save model
        cur_map = self.evaluator.map
        if cur_map > self.best_map:
            # update best-map
            self.best_map = cur_map
            # save model
            print('Saving state, epoch:', self.epoch)
            weight_name = '{}_best.pth'.format(self.args.model)
            checkpoint_path = os.path.join(self.path_to_save, weight_name)
            torch.save({'model': model_eval.state_dict(),
                        'mAP': round(self.best_map*100, 1),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': self.epoch,
                        'args': self.args}, 
                        checkpoint_path)                      

        # set train mode.
        model_eval.trainable = True
        model_eval.train()



    def train_one_epoch(self, model):
        print_freq = 30

        # basic parameters
        epoch_size = len(self.train_loader)

        # Train one epoch
        for iter_i, (images, targets) in enumerate(self.train_loader):
            ni = iter_i + self.epoch * epoch_size
                     
            # To device
            images = images.to(self.device, non_blocking=True).float()

            # Multi scale
            images, targets, img_size = self.rescale_image_targets(
                images, targets, self.model_cfg['stride'], self.args.min_box_size, self.model_cfg['multi_scale'])
    

            # Inference
            with torch.amp.autocast('cuda', enabled=self.args.fp16):
# =============================== 补全下面内容 ===============================
                pass
            # 利用model进行前向传播得到outputs
            # 通过criterion计算损失

# =============================== 补全上面内容 ===============================


            # Backward
            self.scaler.scale(losses).backward()

            # Optimize
            if ni % self.grad_accumulate == 0:
                if self.clip_grad > 0:
                    # unscale gradients
                    self.scaler.unscale_(self.optimizer)
                    # clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
                # optimizer.step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            if iter_i % print_freq == 0:
                print('Epoch: %d | Iter: %d/%d | Loss: %.4f | LR: %.6f' % (self.epoch, iter_i, epoch_size, losses.item(), self.optimizer.param_groups[0]['lr']))


        # LR Schedule
        self.lr_scheduler.step()



    def rescale_image_targets(self, images, targets, stride, min_box_size, multi_scale_range=[0.5, 1.5]):
        """
            Deployed for Multi scale trick.
        """
        if isinstance(stride, int):
            max_stride = stride
        elif isinstance(stride, list):
            max_stride = max(stride)

        # During training phase, the shape of input image is square.
        old_img_size = images.shape[-1]
        min_img_size = old_img_size * multi_scale_range[0]
        max_img_size = old_img_size * multi_scale_range[1]

        # Choose a new image size
        new_img_size = random.randrange(min_img_size, max_img_size + max_stride, max_stride)

        if new_img_size / old_img_size != 1:
            # interpolate
            images = torch.nn.functional.interpolate(
                                input=images, 
                                size=new_img_size, 
                                mode='bilinear', 
                                align_corners=False)
        # rescale targets
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            boxes = torch.clamp(boxes, 0, old_img_size)
            # rescale box
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= min_box_size)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]

        return images, targets, new_img_size

      