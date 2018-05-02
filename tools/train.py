import os
import math
import json
import argparse
import multiprocessing

import numpy as np
import cupy as cp

import chainer
from chainer import training, serializers
from chainer.training import extensions
from chainer import serializers

from pix2pixHD.networks import GlobalGenerator, LocalEnhancer, MultiScaleGenerator, Discriminator, MultiScaleDiscriminator
import pix2pixHD.networks as networks
from pix2pixHD.updater import Pix2pixHDUpdater


def setup_models(args, size, device=0):
    normalization = chainer.links.BatchNormalization

    disc = MultiScaleDiscriminator(return_features=args.no_feature_loss, norm=normalization)
            
    if args.generator == "MultiScaleGenerator":
        enhancer = LocalEnhancer(norm=normalization, input_size=size)
        global_gen = GlobalGenerator(norm=normalization, input_size=[int(x/2) for x in size])

        if args.global_generator_model:
            print("Loading pretrained global generator from {}".format(args.global_generator_model))
            serializers.load_npz(args.global_generator_model, global_gen)
        if args.local_enhancer_model:
            print("Loading pretrained local enhancer from {}".format(args.local_enhancer_model))
            serializers.load_npz(args.local_enhancer_model, enhancer)

        if args.fix_global_num_epochs > 0:
            print("disabling updates for global generator for {} epochs".format(args.fix_global_num_epochs))

        gen = MultiScaleGenerator(global_gen, enhancer)
        if args.multiscale_model:
            serializers.load_npz(args.multiscale_model, gen)
    else:
        gen = getattr(networks, args.generator)(norm=normalization, input_size=size)

    # copy models to GPU(s)
    if device > -1:
        chainer.cuda.get_device(device).use()
        disc.to_gpu()
        gen.to_gpu()         
        
    return disc, gen


def train(args):
    with open(args.config, "r") as cfg:
        data_config = json.load(cfg)
    
    size = data_config["train"]["kwargs"]["size"]

    # set up GPUs if necessary
    device = -1
    comm = None

    if args.device[0] < 0:
        print("training on CPU (not recommended!)")
    else:
        if len(args.device) > 1:
            import chainermn
            comm = chainermn.create_communicator()
            device = args.device[comm.intra_rank]
            
            print("using multi-gpu training with GPU {}".format(device))
        elif args.device[0] >= 0:
            device = args.device[0]
            print("using single gpu training with GPU {}".format(device))

    disc, gen = setup_models(args, size, device)
        
    def setup_optimizer(opt, model, comm=None):
        if comm is not None:
            opt = chainermn.create_multi_node_optimizer(opt, comm)
        opt.setup(model)
        return opt
    
    opt_disc = setup_optimizer(chainer.optimizers.Adam(alpha=args.learning_rate), disc, comm)
    opt_gen = setup_optimizer(chainer.optimizers.Adam(alpha=args.learning_rate), gen, comm)
    
    # pretraining of the global generator needs half-size images
    if comm is None or comm.rank == 0:
        train_d = getattr(pix2pixHD, data_config["class_name"])(*data_config["train"]["args"], **data_config["train"]["kwargs"], one_hot=args.no_one_hot)
        test_d = getattr(pix2pixHD, data_config["class_name"])(*data_config["test"]["args"], **data_config["test"]["kwargs"], one_hot=args.no_one_hot, random_flip=False)
    else:
        train_d, test_d = None, None

    if comm is not None:
        train_d = chainermn.scatter_dataset(train_d, comm)
        test_d = chainermn.scatter_dataset(test_d, comm)
        multiprocessing.set_start_method('forkserver')
        
    train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=2)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize, shuffle=False)
    
    iterators = {'main': train_iter, 'test': test_iter}
    optimizers = {'discriminator': opt_disc, 'generator': opt_gen}
    
    updater = Pix2pixHDUpdater(iterators, optimizers, device=device)

    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.output)

    if comm is None or comm.rank == 0:
        trigger = (100, "iteration")
        if comm is None:
            # this is a hack... sorry
            trainer.extend(train_d.visualizer(n=args.num_vis, one_hot=args.no_one_hot), trigger=trigger) #(1, "epoch"))
        else:
            trainer.extend(train_d._dataset.visualizer(n=args.num_vis, one_hot=args.no_one_hot), trigger=trigger)
        trainer.extend(extensions.LogReport(trigger=(10, "iteration")))
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'Dloss_real', 'Dloss_fake', 'Gloss', "feat_loss", "lr"]), trigger=(10, "iteration"))
        trainer.extend(extensions.ProgressBar(update_interval=10))

        trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(args.epochs // 10, "epoch"))
        trainer.extend(extensions.snapshot_object(gen, 'generator_model_epoch_{.updater.epoch}'), trigger=(args.epochs // 10, "epoch"))
            
    # decay the learning rate from halfway through training
    trainer.extend(extensions.LinearShift("alpha", value_range=(args.learning_rate, 0.0), time_range=(args.epochs // 2, args.epochs), optimizer=opt_disc), trigger=(1, "epoch"))
    trainer.extend(extensions.LinearShift("alpha", value_range=(args.learning_rate, 0.0), time_range=(args.epochs // 2, args.epochs), optimizer=opt_gen), trigger=(1, "epoch"))

    trainer.extend(extensions.observe_value("lr", lambda trainer: trainer.updater.get_optimizer("discriminator").lr), trigger=(10, "iteration"))
        
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', '-g', nargs="+", type=int, default=[-1])
    parser.add_argument('--epochs', '-e', type=int, default=200)
    parser.add_argument('--batchsize', '-b', type=int, default=1)
    parser.add_argument('--learning_rate', '-l', type=float, default=0.0002)
    parser.add_argument('--output', '-o', type=str, default="output")
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--num_vis', '-V', type=int, default=4)
    parser.add_argument('--generator', '-G', default="MultiScaleGenerator",
                        const="MultiScaleGenerator", nargs="?",
                        choices=["LocalEnhancer", "GlobalGenerator", "MultiScaleGenerator"])
    parser.add_argument("--global_generator_model", "-x", type=str, default=None)
    parser.add_argument("--local_enhancer_model", "-y", type=str, default=None)
    parser.add_argument("--multiscale_model", "-z", type=str, default=None)
    parser.add_argument("--no_feature_loss", "-f", action="store_false")
    parser.add_argument("--no_one_hot", "-v", action="store_false")
    parser.add_argument("--fix_global_num_epochs", "-F", type=int, default=0, help="Dont update the global generator for this many epochs.")
    parser.add_argument("--config", "-c", type=str, required=True, help="json file containing dataset information.")
    
    args = parser.parse_args()
    print(args)
    
    train(args)
