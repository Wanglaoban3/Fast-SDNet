from .u_net import *


class UNet_UCC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_UCC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16,32, 64, 128, 256], #'feature_chns': [32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        return main_seg, aux_seg1
