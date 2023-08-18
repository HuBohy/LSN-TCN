import os
import math
import torch
import torch.nn as nn
import numpy as np

from lipreading.models.resnet1D import ResNet1D, BasicBlock1D
from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet

from utils import load_json

def threeD_to_2D_tensor(x):
    n_batch, n_channels, sx, sy, s_time = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

def get_model_from_json(args):
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        "'.json' config path does not exist. Path input: {}".format(args.config_path)
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                  }

    model = LSNTCN(num_classes=args.num_classes,
                    tcn_options=tcn_options,
                    relu_type=args.relu_type,
                    mode=args.mode).cuda()
    
    return model

class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)

        self.consensus_func = torch.mean
        self.has_aux_losses = False

    def forward(self, x, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, dim=-1)
        return x

class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)
        self.consensus_func = torch.mean

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func(out, dim=-1)
        return out

class LSNTCN(nn.Module):
    def __init__( self, hidden_dim=256, num_classes=500,
                  relu_type='prelu', tcn_options={}, mode='video'):
        super(LSNTCN, self).__init__()
        self.mode = mode

        self.audio_frontend_nout = 1
        self.audio_trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)
        self.video_frontend_nout = 64
        self.video_trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        frontend_relu = nn.PReLU(num_parameters=self.video_frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
                    nn.Conv3d(3, self.video_frontend_nout, kernel_size=(7, 7, 5), stride=(2, 2, 1), padding=(3, 3, 2), bias=False),
                    nn.BatchNorm3d(self.video_frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d( kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)))

        self.backend_out = 512
        tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
        num_channels = [hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers']
        self.tcn = tcn_class( input_size=self.backend_out,
                              num_channels=num_channels,
                              num_classes=num_classes,
                              tcn_options=tcn_options,
                              dropout=tcn_options['dropout'],
                              relu_type=relu_type,
                              dwpw=tcn_options['dwpw'],
                            )
        if self.mode == 'fusion':
            self.tcn_output = nn.Linear(num_channels[-1]*2, num_classes)
        else:
            self.tcn_output = nn.Linear(num_channels[-1], num_classes)
        # -- initialize
        self._initialize_weights_randomly()


    def forward(self, v, a):
        if self.mode == 'video':
            B = v.shape[0]
            v = self.frontend3D(v)
            v = threeD_to_2D_tensor(v)
            v = self.video_trunk(v)
            v = v.view(B, -1, v.size(1))

            x = self.tcn(v)
        elif self.mode == 'audio':
            B = a.shape[0]
            a = self.audio_trunk(a)
            a = a.transpose(1, 2)

            x = self.tcn(a)
        elif self.mode == 'fusion':
            B = v.shape[0]
            v = self.frontend3D(v)
            v = threeD_to_2D_tensor(v)
            v = self.video_trunk(v)
            v = v.view(B, -1, v.size(1))
            v = self.tcn(v)

            a = self.audio_trunk(a)
            a = a.transpose(1, 2)
            a = self.tcn(a)

            x = torch.cat((v, a), dim=-1)            

        return self.tcn_output(x)


    def _initialize_weights_randomly(self):
        use_sqrt = True
        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))