import random
import matplotlib
matplotlib.use("Agg")
import json
import os
import numpy as np
import cv2

import chainer
from chainer import Variable
from chainer.dataset import dataset_mixin

from chainercv.transforms import flip

try:
    from cityscapesscripts.helpers.labels import labels, id2label
except ImportError:
    raise ImportError("citysscapescripts not in path. See https://github.com/mcordts/cityscapesScripts")


def transform_image(img):
    img = img.transpose(1, 2, 0)
    img += 1
    img /= 2
    img *= 255.0
    return img


class CityscapesDataset(dataset_mixin.DatasetMixin):
    
    def __init__(self, image_root, gt_root, size=(256, 256), one_hot=False, random_flip=True):
        self.one_hot = one_hot
        self.random_flip = random_flip
        self.size = size

        self.pairs = self.build_list(image_root, gt_root)

        if self.one_hot:
            self.input_channels = len(labels)
        else:
            self.input_channels = 3

    def build_list(self, _root, _gtroot):
        # allow for either string or list of strings as input
        if not isinstance(_root, (list, tuple)):
            _root = [_root]
        if not isinstance(_gtroot, (list, tuple)):
            _gtroot = [_gtroot]

        pairs = []

        for rootpath, gtrootpath in zip(_root, _gtroot):
            for seq in os.listdir(rootpath):
                path = os.path.join(rootpath, seq)
                gtpath = os.path.join(gtrootpath, seq)
                for img in os.listdir(path):
                    fullpath = os.path.join(path, img)
                    truth_file = "gtFine_labelIds" if self.one_hot else "gtFine_color"
                    gtfullpath = os.path.join(gtpath, img.replace("leftImg8bit", truth_file))
                    pairs.append((fullpath, gtfullpath, None))

        print("found {} pairs".format(len(pairs)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def get_image(self, imagepath):
        h, w = self.size
        img = cv2.imread(imagepath, cv2.IMREAD_COLOR).astype(np.float32)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img /= 255.0
        img *= 2.0
        img -= 1.0
        return img.transpose(2, 0, 1)

    def get_label(self, imagepath):
        h, w = self.size
        img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED if self.one_hot else cv2.IMREAD_COLOR).astype(np.float32)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.one_hot:
            img = self.onehot_labels(img)
        else:
            img /= 255.0
            img *= 2.0
            img -= 1.0
        return img.transpose(2, 0, 1)

    def onehot_labels(self, lbl):
        _lbl = np.zeros((lbl.shape + (self.input_channels,))).astype(np.uint8)
        for i in range(self.input_channels):
            _lbl[:,:, i] = lbl==i
        return _lbl.astype(np.float32)

    def get_example(self, i):
        imgpath, lblpath, polypath = self.pairs[i]

        img = self.get_image(imgpath)
        lbl = self.get_label(lblpath)

        if random.random() < 0.5:
            lbl = flip(lbl, x_flip=True)
            img = flip(img, x_flip=True)

        if self.labels_only:
            return lbl
        else:
            return lbl, img

    def visualizer(self, output_path="preview", n=1, one_hot=False):
        @chainer.training.make_extension()
        def make_image(trainer):
            updater = trainer.updater
            output = os.path.join(trainer.out, output_path)
            os.makedirs(output, exist_ok=True)

            rows = []
            for i in range(n):
                label, image = updater.converter(updater.get_iterator("test").next(), updater.device)

                # turn off train mode
                with chainer.using_config('train', False):
                    generated = updater.get_optimizer("generator").target(label).data

                # convert to cv2 image
                img = transform_image(generated[0])
                label = label[0].transpose(1, 2, 0) #transform_image(label[0])
                image = transform_image(image[0])

                # return image from device if necessary
                if updater.device >= 0:
                    img = img.get()
                    label = label.get()
                    image = image.get()

                # convert the onehot label to RGB
                if one_hot:
                    _label = np.zeros_like(img).astype(np.float32)
                    for i, lbl in enumerate(labels):
                        if lbl.ignoreInEval is False:
                            mask = label[: ,:, lbl.id] == 1
                            _label[mask] = np.array(lbl.color[::-1])
                    else:
                        _label = label
                        _label += 1
                        _label /= 2
                        _label *= 255.0

                rows.append(np.hstack((_label, img, image)).astype(np.uint8))

                cv2.imwrite(os.path.join(output, "iter_{}.png".format(updater.iteration)), np.vstack(rows))

        return make_image
