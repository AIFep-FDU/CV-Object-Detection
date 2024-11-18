import random
import numpy as np
import torch
from omegaconf import OmegaConf

# from utils.dataset import VOCDataset
from trainer import build_transform
from evaluator import build_evluator
from config import DATA_CFG, YOLOV1_CFG, TRANS_CONFIG
from yolov1.build import build_model

def fix_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    

if __name__ == '__main__':
    args = OmegaConf.load('args.yaml')

    if torch.cuda.is_available():
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    fix_random_seed(args.seed)

    data_cfg, model_cfg, trans_cfg = DATA_CFG, YOLOV1_CFG, TRANS_CONFIG

    model = build_model(args, model_cfg, device, data_cfg['num_classes'], False)
    ckpt = torch.load('weights/yolov1_best.pth', map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device).eval()


    # trainer = YoloTrainer(args, data_cfg, model_cfg, trans_cfg, device, model, criterion)
    # trainer.train(model)
    # del trainer
    val_transform = build_transform( args=args, is_train=False )
    evaluator = build_evluator(args, 'test', val_transform, device)
    evaluator.test(model)
    