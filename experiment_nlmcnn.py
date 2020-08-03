"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
 
"""

import os
import tensorboardX as tbx
import torch
import models.NlmCNN as NlmCNN

class Experiment:
    def __init__(self, basedir, expname=None):
        os.makedirs(basedir, exist_ok=True)

        if expname is None:
            self.expname = utils.get_exp_dir(basedir)
        else:
            self.expname = expname
        self.expdir = os.path.join(basedir, self.expname)

    def create_network(self):
        n3block_opt = dict(**self.args.n3block)
        type = self.args.backnet
        sizearea = self.args.sizearea
        print(type, sizearea, n3block_opt)
        network_weights = NlmCNN.make_backnet(1, type=type, sizearea=sizearea, bn_momentum=0.1, n3block_opt=n3block_opt, padding=False)
        net = NlmCNN.NlmCNN(network_weights, sizearea=sizearea, sar_data = True, padding=False)
        return net

    def preprocessing_int2net(self, img):
        return img

    def postprocessing_net2int(self, img):
        return img

    def preprocessing_amp2net(self, img):
        return img.square()

    def postprocessing_net2amp(self, img):
        return img.abs().sqrt()

    def create_loss(self):
        def criterion(pred, targets, mask):
            loss = ((pred + targets)/2.0).abs().log() - (pred.abs().log() + targets.abs().log()) / 2 # glrt
            loss = loss.view(pred.shape[0], -1)

            mask = mask.view(pred.shape[0], -1)
            loss = (mask * loss).sum(dim=1)

            return loss
        return criterion

    def create_optimizer(self):
        args = self.args
        assert(args.optimizer == "adam")
        parameters = utils.parameters_by_module(self.net)
        self.base_lr = args.adam["lr"]
        optimizer = torch.optim.Adam(parameters, lr=self.base_lr, weight_decay=args.adam["weightdecay"],
                                     betas=(args.adam["beta1"], args.adam["beta2"]), eps=args.adam["eps"])

        # bias parameters do not get weight decay
        for pg in optimizer.param_groups:
            if pg["name"] == "bias":
                pg["weight_decay"] = 0

        return optimizer

    def learning_rate_decay(self, epoch):
        if epoch < 20:
            return 1
        elif epoch < 40:
            return 0.1
        elif epoch < 50:
            return 0.01
        else:
            return 0

    def setup(self, args=None, use_gpu=True):
        print(self.expname, self.expdir)
        os.makedirs(self.expdir, exist_ok=True)

        if args == None:
            self.args = utils.load_args(self.expdir)
        else:
            self.args = utils.args2obj(args)
            utils.save_args(self.expdir, self.args)

        writer_dir = os.path.join(self.expdir, 'train')
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = tbx.SummaryWriter(log_dir=writer_dir)
        self.use_cuda = torch.cuda.is_available() and use_gpu
        self.net = self.create_network()
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_loss()

        print(self.net)
        print("#Parameter %d" % utils.parameter_count(self.net))

        self.epoch = 0

        if self.use_cuda:
            self.net.cuda()

    def add_summary(self, name, value, epoch=None):
        if epoch is None:
            epoch = self.epoch
        try:
            self.writer.add_scalar(name, value, epoch)
        except:
            pass

def get_patchsize(patchsize, backnet):
    if patchsize>0:
        return patchsize
    elif backnet==2:
        return 104
    else:
        return 48

def main_sync_sar(args):
    exp_basedir = args.exp_basedir % args.backnet if '%d' in args.exp_basedir else args.exp_basedir
    patchsize = get_patchsize(args.patchsize, args.backnet)

    if args.weights:
        from experiment_utility import load_checkpoint, test_list_weights
        from dataset.folders_data import list_test_10synt as listfile_test
        listfile_test = [x for x in listfile_test if x[0][-3:] == '_04']

        assert (args.exp_name is not None)
        experiment = Experiment(exp_basedir, args.exp_name)
        experiment.setup(use_gpu=args.use_gpu)
        load_checkpoint(experiment, args.eval_epoch)
        outdir = os.path.join(experiment.expdir, "weights%03d" % args.eval_epoch)
        test_list_weights(experiment, outdir, listfile_test, pad=18)
    elif args.eval:
        from experiment_utility import load_checkpoint, test_list
        from dataset.folders_data import list_test_10synt as listfile_test

        assert(args.exp_name is not None)
        experiment = Experiment(exp_basedir, args.exp_name)
        experiment.setup(use_gpu=args.use_gpu)
        load_checkpoint(experiment, args.eval_epoch)
        outdir = os.path.join(experiment.expdir, "results%03d" % args.eval_epoch)
        test_list(experiment, outdir, listfile_test, pad=18)
    else:
        from experiment_utility import trainloop
        from dataloaders import create_train_syncsar_dataloaders as create_train_dataloaders
        from dataloaders import create_valid_syncsar_dataloaders as create_valid_dataloaders
        from dataloaders import PreprocessingIntNoisyFromAmp as Preprocessing

        experiment = Experiment(exp_basedir, args.exp_name)
        experiment.setup(args, use_gpu=args.use_gpu)
        trainloader = create_train_dataloaders(patchsize, args.batchsize, args.trainsetiters)
        validloader = create_valid_dataloaders(args.patchsizevalid, args.batchsizevalid)
        trainloop(experiment, trainloader, Preprocessing(), log_data=False, validloader=validloader)

if __name__ == '__main__':
    import argparse
    import os
    from utils import utils
    import torch
    from n3net.n3block import add_commandline_n3params

    parser = argparse.ArgumentParser(description='NLMCNN for SAR image denoising')
    parser.add_argument("--backnet", type=int, default=2)
    parser.add_argument("--sizearea", type=int, default=25)
    add_commandline_n3params(parser, "n3block", k=7, external_temp=True)

    # Optimizer
    parser.add_argument('--optimizer', default="adam", choices=["adam", "sgd"]) # which optimizer to use
    # parameters for Adam
    parser.add_argument("--adam.beta1", type=float, default=0.9)
    parser.add_argument("--adam.beta2", type=float, default=0.999)
    parser.add_argument("--adam.eps", type=float, default=1e-8)
    parser.add_argument("--adam.weightdecay", type=float, default=1e-4)
    parser.add_argument('--adam.lr', type=float, default=0.001)
    # parameters for SGD
    parser.add_argument("--sgd.momentum", type=float, default=0.9)
    parser.add_argument("--sgd.weightdecay", type=float, default=1e-4)
    parser.add_argument('--sgd.lr', type=float, default=0.001)

    # Eval mode
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--weights', action='store_true')
    parser.add_argument('--eval_epoch', type=int)

    # Training options
    parser.add_argument("--batchsize"     , type=int, default= 32)
    parser.add_argument("--patchsize"     , type=int, default=-1 )
    parser.add_argument("--batchsizevalid", type=int, default=8  )
    parser.add_argument("--patchsizevalid", type=int, default=256)

    # Misc
    utils.add_commandline_flag(parser, "--use_gpu", "--use_cpu", True)
    parser.add_argument("--exp_name"   , default=None)

    base_expdir = "./results/sar_sync/nlmcnn_%d/"
    parser.add_argument("--exp_basedir", default=base_expdir)
    parser.add_argument("--trainsetiters", type=int, default=640)
    args = parser.parse_args()
    main_sync_sar(args)
