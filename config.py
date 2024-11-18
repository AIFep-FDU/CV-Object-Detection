DATA_CFG = {
    'data_name': 'VOCdevkit',
    'num_classes': 20,
    'class_indexs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'class_names': ('aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor'),
}
YOLOV1_CFG = {
    # ---------------- Model config ----------------
    ## Backbone
    'backbone': 'resnet18',
    'pretrained': True,
    'stride': 32,  # P5
    'max_stride': 32,
    ## Neck
    'neck': 'sppf',
    'neck_act': 'lrelu',
    'neck_norm': 'BN',
    'neck_depthwise': False,
    'expand_ratio': 0.5,
    'pooling_size': 5,
    ## Head
    'head': 'decoupled_head',
    'head_act': 'lrelu',
    'head_norm': 'BN',
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': False,
    # ---------------- Data process config ----------------
    ## Input
    'multi_scale': [0.5, 1.5], # 320 -> 960
    'trans_type': 'ssd',
    # ---------------- Loss config ----------------
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    # ---------------- Trainer config ----------------
    'trainer_type': 'yolo',
}

TRANS_CONFIG = {
    'aug_type': 'ssd',
    'use_ablu': False,
    # Mosaic & Mixup are not used for SSD-style augmentation
    'mosaic_prob': 0.0,
    'mixup_prob':  0.0,
    'mosaic_type': 'yolov5',
    'mixup_type':  'yolov5',
    'mixup_scale': [0.5, 1.5]   # "mixup_scale" is not used for YOLOv5MixUp, just for YOLOXMixup
}