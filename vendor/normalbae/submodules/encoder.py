import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5_ap'
        print('Loading base model ()...'.format(basemodel_name), end='')
        # LC change: torch.hub.load would download and run code from GitHub at load time. The same network comes from
        # the timm package instead (timm's name for these weights is tf_efficientnet_b5.ap_in1k). No pretrained download:
        # scannet.pt holds every weight. timm folds act1 into bn1, so an empty act1 is put back after bn1; the module
        # list (and so the feature indices the decoder reads) is then the same as gen-efficientnet's.
        import timm
        basemodel = timm.create_model(basemodel_name.replace('tf_efficientnet_b5_ap', 'tf_efficientnet_b5.ap_in1k'), pretrained=False)
        modules = list(basemodel._modules.items())
        basemodel._modules.clear()
        for k, v in modules:
            basemodel._modules[k] = v
            if k == 'bn1':
                basemodel._modules['act1'] = nn.Identity()
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


