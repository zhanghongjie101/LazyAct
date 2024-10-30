import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.nn import init
from torch.autograd import Variable

def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module):
    """ Parameter initializer for Atari models

    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()

class SkipCNN(nn.Module):
    def __init__(self, num_actions, num_skips=10):
        """ Basic convolutional actor-critic network for Atari 2600 games

        Equivalent to the network in the original DQN paper.

        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()
        self.size = 2.0
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        cnn_seq = [(0,1,8,4),(0,1,4,2),(0,1,3,1)]
        self.h_out, self.w_out = self.cnn_to_fc(84,84,cnn_seq)
        self.fc = nn.Linear(64 * self.h_out * self.w_out, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pi = nn.Linear(512+num_skips, num_actions)
        self.skip = nn.Linear(512, num_skips)
        self.v_a = nn.Linear(512, 1)
        self.v_c = nn.Linear(512, 1)

        self.num_actions = num_actions
        self.num_skips = num_skips

        # parameter initialization
        self.apply(atari_initializer)
        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v_a.weight.data = ortho_weights(self.v_a.weight.size())
        self.v_c.weight.data = ortho_weights(self.v_c.weight.size())

    def forward(self, x, skip=None):
        """ Module forward pass

        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        if skip is None:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.relu4(x)
            skip = self.skip(x.detach())
            return skip, x
        else:
            v_a = self.v_a(x)
            v_c = self.v_c(x)
            pi_x = torch.cat((x, skip),-1)
            pi = self.pi(pi_x)
            return pi, v_a, v_c

    def cnn_to_fc(self, h_in, w_in, seq):
        h_out = h_in
        w_out = w_in
        for c in seq:
            h_out = int((h_out+2.0*c[0]-c[1]*(c[2]-1)-1)/c[3]+1)
            w_out = int((w_out+2.0*c[0]-c[1]*(c[2]-1)-1)/c[3]+1)
        return h_out, w_out