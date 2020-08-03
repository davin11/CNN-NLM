'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import math
import torch.nn as nn
from n3net import non_local

def cnn_from_def(cnn_opt):
    kernel = cnn_opt.get("kernel", 3)
    bn = cnn_opt.get("bn", True)
    depth = cnn_opt.get("depth", 1)
    channels = cnn_opt.get("features")
    inchannels = cnn_opt.get("nplanes_in")
    outchannels = cnn_opt.get("nplanes_out")
    firstbn = cnn_opt.get("firstbn", bn)
    lastact = cnn_opt.get("lastact", 'linear')
    activation = cnn_opt.get("activation", 'relu')
    padding = cnn_opt.get("padding", True)

    kernels = [kernel, ] * depth
    features = [channels, ] * (depth - 1) + [outchannels, ]
    bns = [firstbn, ] + [bn, ] * (depth - 2) + [False, ]
    dilats = [1, ] * depth
    acts = [activation, ] * (depth - 1) + [lastact, ]

    from models.DnCNN import make_net
    net = make_net(inchannels, kernels, features, bns, acts, dilats=dilats, padding=None if padding else 0)
    net.nplanes_out = outchannels
    net.nplanes_in = inchannels

    return net

class N3Block(nn.Module):
    r"""
    N3Block operating on a 2D images
    """
    def __init__(self, nplanes_in_data, nplanes_in_feat, k, patchsize=10, stride=5,
                 nl_match_window=15, residue=True,
                 nl_temp=dict(), embedcnn=dict()):
        r"""
        :param nplanes_in_data: number of maps for data input
        :param nplanes_in_feat: number of maps for feature input
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param nl_match_window: size of matching window around each patch,
            i.e. the nl_match_window x nl_match_window patches around a query patch
            are used for matching
        :param nl_temp: options for handling the the temperature parameter
        :param embedcnn: options for the embedding cnn, also shared by temperature cnn
        """
        super(N3Block, self).__init__()
        self.patchsize = patchsize
        self.stride    = stride
        self.residue   = residue

        # patch embedding
        embedcnn["nplanes_in"] = nplanes_in_feat
        self.embedcnn = cnn_from_def(embedcnn)

        # temperature cnn
        with_temp = nl_temp.get("external_temp")
        if with_temp:
            tempcnn_opt = dict(**embedcnn)
            tempcnn_opt["nplanes_out"] = 1
            self.tempcnn = cnn_from_def(tempcnn_opt)
        else:
            self.tempcnn = None

        self.nplanes_in = nplanes_in_data
        self.nplanes_out = (k+1) * nplanes_in_data

        indexer = lambda xe_patch,ye_patch: non_local.index_neighbours(xe_patch, ye_patch, nl_match_window, exclude_self=True)
        self.n3aggregation = non_local.N3Aggregation2D(indexing=indexer, k=k,
                                                       patchsize=patchsize, stride=stride, temp_opt=nl_temp,
                                                       residue=self.residue)
        self.k = k


    def forward(self, x_data, x_faet):
        if self.k <= 0:
            return x_data

        xe = self.embedcnn(x_faet)

        if self.tempcnn is not None:
            log_temp = self.tempcnn(x_faet)
        else:
            log_temp = None

        y = self.n3aggregation(x_data, xe, log_temp=log_temp)

        return y


def add_commandline_n3params(parser, name, k=7, external_temp=True):
    # Nonlocal block Parameters
    from models.DnCNN import add_commandline_networkparams
    from utils.utils import add_commandline_flag

    parser.add_argument("--%s.k"%name, type=int, default=k) # number of neighborhood volumes
    parser.add_argument("--%s.patchsize"%name, type=int, default=10)
    parser.add_argument("--%s.stride"%name, type=int, default=5)

    add_commandline_networkparams(parser, "%s.embedcnn" % name, 64, 3, 3, "relu", True) # Specification of embedding CNNs: features, depth, kernelsize, activation, batchnorm
    parser.add_argument("--%s.embedcnn.nplanes_out" % name, type=int, default=8)  # output channels of embedding CNNs
    add_commandline_flag(parser, "--%s.nl_temp.external_temp"% name, "--%s.nl_temp.no_external_temp"% name, default=external_temp) # whether to have separate temperature CNN
    parser.add_argument("--%s.nl_temp.temp_bias"%name, type=float, default=0.1) # constant bias of temperature
    add_commandline_flag(parser, "--%s.nl_temp.distance_bn"%name, "--%s.nl_temp.no_distance_bn"%name, default=True) # whether to have batch norm layer after calculat of pairwise distances
    add_commandline_flag(parser, "--%s.nl_temp.avgpool"%name, "--%s.nl_temp.no_avgpool"%name, default=True) # in case of separate temperature CNN: whether to average pool temperature of each patch or to take temperature of center pixel


if __name__ == '__main__':
    import argparse
    from utils.utils import args2obj
    parser = argparse.ArgumentParser(description='N3Block')
    add_commandline_n3params(parser, 'n3block')
    args = parser.parse_args()
    args = args2obj(args)
    print(args.n3block)
    N3Block(8, 8, **args.n3block)