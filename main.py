import argparse
import os
from datetime import datetime
import numpy as np
import torch
from data.dataloaders import get_loaders
from model_trains import *


if __name__ == '__main__':
    seed = 1337
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientnet_b0', help='model_name',
                        choices=['uaps', 'dct', 'cct', 'baseline', 'nlc', 'mobilenetv3_small', 'efficientnet_b0', 'resnet18'
                                 'u_net', 'fastsurfacenet', 'fdsnet', 'bisenet', 'edrnet', 'enet', 'fastcnn'])
    parser.add_argument('--benchmark', type=str, default='carpet', help='dataset',
                        choices=['KolektorSDD', 'KolektorSDD2', 'carpet', 'hazelnut', 'MT', 'CrackForest', 'Crack500', 'CDD', 'DAGM1', 'DAGM2', 'DAGM3', 'DAGM4', 'DAGM5', 'DAGM6', 'DAGM7', 'DAGM8', 'DAGM9', 'DAGM10'])
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--dataset_root_path', type=str, default='C:/wrd/IndustryNetData/Data/carpet')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--mode', type=str, default='total-sup', choices=['total-sup', 'semi-sup'])
    parser.add_argument('--batch_size', type=int, default=6)

    # semi-supervised parameter
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=150.0, help='consistency_rampup')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.4)
    args = parser.parse_args()

    logtime = datetime.now().__format__('_%m%d_%H%M%S')
    log_path = args.log_path+'/'+args.benchmark+'/'+args.model+logtime+'/'

    # semi-supervised model train
    if args.mode == 'semi-sup':
        loaders = get_loaders(args.benchmark, args.dataset_root_path, args.batch_size, args.mode, args.unlabeled_ratio)
        # build model
        if args.model == 'baseline':
            NetWork = SemiSupBaselineTrain(args.epochs, args.benchmark, args.model, log_path)
        elif args.model == 'uaps':
            NetWork = UAPSTrain(args.epochs, args.benchmark, args.model, log_path, args.consistency)
        elif args.model == 'dct':
            NetWork = DCTTrain(args.epochs, args.benchmark, args.model, log_path, args.consistency)
        elif args.model == 'cct':
            NetWork = CCTTrain(args.epochs, args.benchmark, args.model, log_path, args.consistency)
        elif args.model == 'uamt':
            NetWork = UAMTTrain(args.epochs, args.benchmark, args.model, log_path, args.consistency)
        elif args.model == 'mt':
            NetWork = MTTrain(args.epochs, args.benchmark, args.model, log_path, args.consistency)
        elif args.model == 'nlc':
            NetWork = NLCTrain(args.epochs, args.benchmark, args.model, log_path, args.consistency)
        else:
            raise print('Please input correct model!')

        # run!
        NetWork.run(loaders['train'], loaders['unlabeled'], loaders['val'], loaders['test'])

    # total-supervised model train
    if args.mode == 'total-sup':
        loaders = get_loaders(args.benchmark, args.dataset_root_path, args.batch_size, args.mode)
        if args.model == 'fastsurfacenet':
            NetWork = FastSurfaceTrain(args.epochs, args.benchmark, args.model, log_path, int(args.epochs*0.2), args.base_lr)
        elif args.model in ['u_net', 'bisenet', 'edrnet', 'enet', 'fastcnn', 'fdsnet']:
            NetWork = BaseLineTrain(args.epochs, args.benchmark, args.model, log_path, args.base_lr)
        elif args.model in ['mobilenetv3_small', 'efficientnet_b0', 'resnet18']:
            NetWork = ClsModelTrain(args.epochs, args.benchmark, args.model, log_path, args.base_lr)
        else:
            raise print("Please input correct model!")
        NetWork.run(loaders['train'], unlabeled_loader=None, val_loader=loaders['val'], test_loader=loaders['test'], checkpoint=args.checkpoint)
