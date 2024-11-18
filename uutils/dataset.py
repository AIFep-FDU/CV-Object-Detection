import cv2
import random
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
import os

# VOC class names
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDataset(data.Dataset):
    def __init__(self, 
                 data_dir     :str,
                 split_dir    :str,
                 transform    = None,
                 is_train     :bool = False,
                 ):

        self.is_train = is_train
        self.target_transform = VOCAnnotationTransform()
        self.transform = transform

        self.root = data_dir
        self._annopath = osp.join(self.root, 'Annotations', '%s.xml')
        self._imgpath = osp.join(self.root, 'JPEGImages', '%s.jpg')

        self.ids = list()
        assert split_dir in ['train', 'val', 'trainval', 'test']
        txt_path = osp.join(self.root, 'ImageSets', 'Main', split_dir+'.txt')
        for line in open(txt_path):
            self.ids.append(line.strip())
        self.dataset_size = len(self.ids)



    def __getitem__(self, index):
        mosaic = False
        image, target = self.load_image_target(index)
        image, target, deltas = self.transform(image, target, mosaic)
        return image, target, deltas

    def __len__(self):
        return self.dataset_size
    
    def load_image_target(self, index):
        image = self.pull_image(index)
        height, width, channels = image.shape

        anno = self.pull_anno(index)
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }
        return image, target

    def pull_image(self, index):
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        return image

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self.target_transform(anno)
        return anno
    

def build_dataset(args, split_dir, data_cfg, transform, is_train=False):
    # ------------------------- Basic parameters -------------------------
    data_dir = args.root
    dataset_info = {
        'num_classes': data_cfg['num_classes'],
        'class_names': data_cfg['class_names'],
        'class_indexs': data_cfg['class_indexs']
    }

    # ------------------------- Build dataset -------------------------
    ## VOC dataset
    dataset = VOCDataset(
        data_dir = data_dir, split_dir = split_dir,
        transform = transform, is_train = is_train,
    )
    return dataset, dataset_info


## build dataloader
def build_dataloader(args, dataset, batch_size, collate_fn=None):

    sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    return dataloader


## collate_fn for dataloader
class CollateFunc(object):
    def __call__(self, batch):
        targets = []
        images = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)

        images = torch.stack(images, 0) # [B, C, H, W]

        return images, targets