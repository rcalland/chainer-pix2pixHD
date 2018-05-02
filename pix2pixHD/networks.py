import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer.links.model.vision import resnet


# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):

    def __init__(self, ch0, ch1, ksize=3, stride=1, padding=0, bn=True, sample='down', activation=F.relu, normalization=L.BatchNormalization, dropout=False, outsize=None):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.nobias = self.bn
        self.outsize = outsize
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
                layers['c'] = L.Convolution2D(ch0, ch1, self.ksize, self.stride, self.padding, initialW=w, nobias=self.nobias)
        else:
                layers['c'] = L.Deconvolution2D(ch0, ch1, self.ksize, self.stride, self.padding, initialW=w, nobias=self.nobias, outsize=self.outsize)
        if bn:
            layers['norm'] = normalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.norm(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class ResBlock(chainer.Chain):

    def __init__(self, ch0, ch1, bn=True, normalization=L.BatchNormalization):
        layers = {}
        self.bn = bn
        w = chainer.initializers.Normal(0.02)
        layers['conv1'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w, nobias=True)
        layers['conv2'] = L.Convolution2D(ch1, ch1, 3, 1, 1, initialW=w, nobias=True)
        if bn:
            layers['bn1'] = normalization(ch1)
            layers['bn2'] = normalization(ch1)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x):
        if self.bn:
            h = F.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return h + x
        else:
            h = F.relu(self.conv1(x))
            h = self.conv2(h)
            return h + x


class GlobalGenerator(chainer.Chain):

    def __init__(self, input_size=(512, 1024), num_resblocks=9, norm=L.BatchNormalization):
        super().__init__()
        self.num_resblocks = num_resblocks
        with self.init_scope():
            self.c7s1_64 = CBR(None, 64, ksize=7, stride=1, padding=3, normalization=norm)
            self.d128 = CBR(64, 128, ksize=3, stride=2, padding=1, normalization=norm)
            self.d256 = CBR(128, 256, ksize=3, stride=2, padding=1, normalization=norm)
            self.d512 = CBR(256, 512, ksize=3, stride=2, padding=1, normalization=norm)
            self.d1024 = CBR(512, 1024, ksize=3, stride=2, padding=1, normalization=norm)

            for i in range(self.num_resblocks):
                self.add_link("R1024_{}".format(i), ResBlock(1024, 1024, normalization=norm))

            self.u512 = CBR(1024, 512, ksize=3, stride=2, padding=1, sample="up", normalization=norm, outsize=[int(x / 8) for x in input_size])
            self.u256 = CBR(512, 256, ksize=3, stride=2, padding=1, sample="up", normalization=norm, outsize=[int(x / 4) for x in input_size])
            self.u128 = CBR(256, 128, ksize=3, stride=2, padding=1, sample="up", normalization=norm, outsize=[int(x / 2) for x in input_size])
            self.u64 = CBR(128, 64, ksize=3, stride=2, padding=1, sample="up", normalization=norm, outsize=input_size)
            self.c7s1_3 = CBR(64, 3, ksize=7, stride=1, padding=3, bn=False, normalization=None, activation=F.tanh)

    def __call__(self, x, return_image=True):
        h = self.c7s1_64(x)
        h = self.d128(h)
        h = self.d256(h)
        h = self.d512(h)
        h = self.d1024(h)
        for i in range(self.num_resblocks):
            h = self["R1024_{}".format(i)](h)
        h = self.u512(h)
        h = self.u256(h)
        h = self.u128(h)
        h = self.u64(h)
        if return_image:
            h = self.c7s1_3(h)
        return h


class LocalEnhancer(chainer.Chain):

    def __init__(self, input_size=(512, 1024), num_resblocks=3, norm=L.BatchNormalization):
        super().__init__()
        self.num_resblocks = num_resblocks
        with self.init_scope():
            self.c7s1_32 = CBR(None, 32, ksize=7, stride=1, padding=3, normalization=norm)
            self.d64 = CBR(32, 64, ksize=3, stride=2, padding=1, normalization=norm)
            for i in range(self.num_resblocks):
                self.add_link("R64_{}".format(i), ResBlock(64, 64, normalization=norm))
            self.u32 = CBR(64, 32, ksize=3, stride=2, padding=1, sample="up", normalization=norm, outsize=input_size)
            self.c7s1_3 = CBR(32, 3, ksize=7, stride=1, padding=3, bn=False, normalization=None, activation=F.tanh)

    def __call__(self, x, global_features=None):
        h = self.c7s1_32(x)
        h = self.d64(h)
        if global_features is not None:
            h = h + global_features
        for i in range(self.num_resblocks):
            h = self["R64_{}".format(i)](h)
        h = self.u32(h)
        h = self.c7s1_3(h)
        return h


class MultiScaleGenerator(chainer.Chain):

    def __init__(self, global_generator=GlobalGenerator(), local_enhancer=LocalEnhancer()):
        super().__init__()

        with self.init_scope():
            self.global_generator = global_generator
            self.local_enhancer = local_enhancer
            
    def __call__(self, x):
        x_half_size = F.average_pooling_2d(x, 3, stride=2, pad=1)
        global_features = self.global_generator(x_half_size, return_image=False)
        h = self.local_enhancer(x, global_features)
        return h


class Discriminator(chainer.Chain):

    def __init__(self, return_features=False, norm=L.BatchNormalization):
        super().__init__()
        self.return_features = return_features
        kw = 4
        pad = int(np.ceil((kw-1.0)/2))
        with self.init_scope():
            self.C64 = CBR(None, 64, ksize=kw, stride=2, padding=pad, bn=False, normalization=None, activation=F.leaky_relu)
            self.C128 = CBR(64, 128, ksize=kw, stride=2, padding=pad, bn=not spectral_norm, normalization=norm, activation=F.leaky_relu)
            self.C256 = CBR(128, 256, ksize=kw, stride=2, padding=pad, bn=not spectral_norm, normalization=norm, activation=F.leaky_relu)
            self.C512 = CBR(256, 512, ksize=kw, stride=1, padding=pad, bn=not spectral_norm, normalization=norm, activation=F.leaky_relu)
            self.out = L.Convolution2D(512, 1, ksize=kw, stride=1, pad=pad)
                
    def __call__(self, x):
        h0 = self.C64(x)
        h1 = self.C128(h0)
        h2 = self.C256(h1)
        h3 = self.C512(h2)
        h4 = self.out(h3)

        if self.return_features:
            return [h0, h1, h2, h3, h4]
        else:
            return h4


class MultiScaleDiscriminator(chainer.Chain):

    def __init__(self, num_scales=3, return_features=False, norm=L.BatchNormalization):
        super().__init__()
        self.return_features = return_features
        self.num_scales = num_scales
        for i in range(self.num_scales):
            self.add_link("D_{}".format(i),
                          Discriminator(return_features=self.return_features, norm=norm))

    def __call__(self, x, y):
        h = F.concat((x, y), axis=1)
        features = []
        result = []
        for i in range(self.num_scales):
            if i > 0:
                h = F.average_pooling_2d(h, 3, stride=2, pad=1)
            ret = self["D_{}".format(i)](h)

            if self.return_features:
                features.append(ret[:-1])
                result.append(ret[-1])
            else:
                result.append(ret)

        return features, result
