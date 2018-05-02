#!/usr/bin/env python

from __future__ import print_function

import functools

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
import cupy as cp

from chainer import cuda, function, reporter


class Pix2pixHDUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.feat_lambda = kwargs.pop("feature_lambda", 10.)
        self.fix_global_num_epochs = kwargs.pop("fix_global_num_epochs", 0)
        super().__init__(*args, **kwargs)

        self.label_true = []
        self.label_false = []
                    
    def update_model(self, opt, loss):
        opt.target.cleargrads()
        loss.backward()
        opt.update()

    def get_batch(self, iterator):
        a, b = self.converter(iterator.next(), self.device)
        return chainer.Variable(a), chainer.Variable(b)

    def make_labels(self, D):
        if self.device >= 0:
            for d in D:
                self.label_true.append(cp.full(d.shape, 1, dtype=cp.float32))
                self.label_false.append(cp.full(d.shape, 0, dtype=cp.float32))
        else:
            for d in D:
                self.label_true = np.full(d.shape, 1, dtype=np.float32)
                self.label_false = np.full(d.shape, 0, dtype=np.float32)     

    def discriminator_loss(self, D, label_type):
        if not self.label_true:
            self.make_labels(D)
            
        for i, d in enumerate(D):
            loss += F.mean_squared_error(d, self.label_true[i] if label_type else self.label_false[i])

        return loss

    def update_core(self):
        generator_optimizer = self.get_optimizer('generator')
        discriminator_optimizer = self.get_optimizer('discriminator')
        
        gen = generator_optimizer.target
        disc = discriminator_optimizer.target    
        
        label, image = self.get_batch(self.get_iterator('main'))
        real_features, y_real = disc(image, label)

        fake = gen(label)
        fake_features, y_fake = disc(fake, label)

        Dloss_real = self.discriminator_loss(y_real, True)
        Dloss_fake = self.discriminator_loss(y_fake, False)

        Dloss = 0.5 * (Dloss_real + Dloss_fake)
        Gloss = 0.5 * self.discriminator_loss(y_fake, True)
        
        if disc.return_features:
            feature_loss = self.feature_loss(real_features, fake_features)
        else:
            feature_loss = 0.
            
        # fix global generator if we selected that option
        if gen.__class__.__name__ == "MultiScaleGenerator":
            if self.fix_global_num_epochs >= self.epoch:
                gen.global_generator.disable_update()
            else:
                gen.global_generator.enable_update()

        self.update_model(generator_optimizer, Gloss + feature_loss)
        fake.unchain_backward()
        label.unchain_backward()
        self.update_model(discriminator_optimizer, Dloss)
        reporter.report({"Dloss_real": Dloss_real, "Dloss_fake": Dloss_fake, "Gloss": Gloss, "feat_loss": feature_loss})

    def feature_loss(self, features_real, features_fake):
        feature_loss = 0.
        for x, y in zip(features_real, features_fake):
            for m, n in zip(x, y):
                feature_loss += self.feat_lambda * F.mean_absolute_error(chainer.Variable(m.data), n) / (len(features_fake))

        return feature_loss
