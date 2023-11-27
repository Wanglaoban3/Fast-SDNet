from models.semi_supervised import *
from models.total_supvised import *
from torchvision.models import resnet18, mobilenet_v3_small, efficientnet_b0
import torch.nn as nn


def get_model(net_type="uaps", in_chns=3, class_num=4):
    # semi-supervised model
    if net_type in ['dct', 'mt', 'uamt', 'baseline']:
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "uaps":
        net = UNet_UAPS(in_chns=in_chns, class_num=class_num)
    elif net_type == "cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num)
    elif net_type == 'nlc':
        net = UNet_NLC(in_chns=in_chns, class_num=class_num)

    # total supervised model
    elif net_type == 'fastsurfacenet':
        net = FastSurfaceNet(in_channels=in_chns, num_classes=class_num)
    elif net_type == 'u_net':
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == 'edrnet':
        net = EDRNet(in_chns, class_num)
    elif net_type == 'enet':
        net = ENet(class_num)
    elif net_type == 'fdsnet':
        net = FDSNet(class_num)
    elif net_type == 'fastcnn':
        net = FastSCNN(in_chns, class_num)
    elif net_type == 'bisenet':
        net = BiSeNet(class_num)

    # classification model
    elif net_type == 'resnet18':
        net = resnet18()
        net.fc = nn.Linear(512, class_num)
    elif net_type == 'efficientnet_b0':
        net = efficientnet_b0()
        net.classifier[-1] = nn.Linear(1280, class_num)
    elif net_type == 'mobilenetv3_small':
        net = mobilenet_v3_small()
        net.classifier[-1] = nn.Linear(1024, class_num)
    else:
        raise print('Please input correct net_type!')

    return net
