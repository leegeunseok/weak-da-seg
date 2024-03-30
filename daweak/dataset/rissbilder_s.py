###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2019
###########################################################################

import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
import cv2  ####################
import random  #################

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RissbilderSSegmentation(data.Dataset):
    def __init__(
            self,
            dataset=None,
            path=None,
            split=None,
            mode=None,
            data_root=None,
            max_iters=None,
            size=(256, 256),
            use_pixeladapt=False
    ):
        self.dataset = dataset
        self.path = path
        self.split = split
        self.mode = mode
        self.data_root = data_root
        self.size = size
        self.ignore_label = 255
        self.mean = np.array((127.06984280, 118.93133290, 104.85159191), dtype=np.float32)
        self.use_pixeladapt = use_pixeladapt

        # load image list
        list_path = osp.join(self.data_root, '%s_list/%s.txt' % (self.dataset, self.split))
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        # map label IDs to the format of Cityscapes
        self.id_to_trainid = {0: 0, 1: 1}

        # load dataset
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.path, "leftImg8bit/%s/%s" % (self.split, name))
            label_file = osp.join(self.path, "gtFine/%s/%s_gtFine_labelIds.png" % (self.split, name[:-16]))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            if use_pixeladapt:
                p_img_file = osp.join(self.path, "leftImg8bit/%s/%s" % (self.split, name))
                self.files.append({
                    "img": p_img_file,
                    "label": label_file,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.size, Image.BICUBIC)
        label = label.resize(self.size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        ####################################################################
        flip_type = 'none'
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1)
            label_copy = np.flip(label_copy, axis=1)
            flip_type = 'horizontal'

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=0)
            label_copy = np.flip(label_copy, axis=0)
            if flip_type == 'none':
                flip_type = 'vertical'
            elif flip_type == 'horizontal':
                flip_type = 'ho_ver'

        # convert label for dilation
        label_for_dilation = label_copy.astype(np.uint8)
        # define the dilation kernel
        kernel = np.ones((6,6), np.uint8)
        # apply dilation
        dilated_label = cv2.dilate(label_for_dilation, kernel, iterations=1)
        # convert dilated_label back to float32
        dilated_label = dilated_label.astype(np.float32)
        ####################################################################

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), dilated_label.copy(), np.array(size), name  #####################