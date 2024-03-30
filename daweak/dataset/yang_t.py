###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2019
###########################################################################

import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
import json
import cv2  #################
import random  #################

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YangTSegmentation(data.Dataset):
    def __init__(
            self,
            dataset=None,
            path=None,
            split=None,
            mode=None,
            data_root=None,
            max_iters=None,
            size=(256, 256),
            use_points=False
    ):
        self.dataset = dataset
        self.path = path
        self.split = split
        self.mode = mode
        self.data_root = data_root
        self.size = size
        self.ignore_label = 255
        self.mean = np.array((135.29352, 139.95343, 142.78117), dtype=np.float32)
        self.use_points = use_points

        # label mapping
        with open(osp.join(self.data_root, '%s_list/info.json' % self.dataset), 'r') as fp:
            info = json.load(fp)
        self.mapping = np.array(info['label2train'], dtype=np.int)

        # load image list
        list_path = osp.join(self.data_root, '%s_list/%s.txt' % (self.dataset, self.split))
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.point_labels = []
        if self.use_points:
            for label_line in self.img_ids:

                name = osp.join(
                    self.path, "gtFine/%s/%s_gtFine_labelIds.png" % (self.split, label_line[:-16])
                )
                label = Image.open(name).resize(self.size, Image.NEAREST)
                label = np.array(label).astype(np.uint8)
                label = self.label_mapping(label, self.mapping)
                choose_idx = []
                set_label = set(label.reshape(-1)) - {255}
                for lbl in set_label:
                    idx = np.where(label == lbl)
                    choose = np.random.choice(len(idx[0]), size=min(1, len(idx[0])), replace=False)
                    for c in choose:
                        choose_idx.append([idx[0][c], idx[1][c]])
                self.point_labels.append(np.array(choose_idx))
            # self.point_labels = list(self.point_labels)

        if max_iters is not None:
            self.point_labels = \
                self.point_labels * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        # load dataset
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.path, "leftImg8bit/%s/%s" % (self.split, name))
            label_file = osp.join(
                self.path, "gtFine/%s/%s_gtFine_labelIds.png" % (self.split, name[:-16])
            )
            self.files.append({
                "img": img_file,
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
        label = np.asarray(label, np.uint8)

        # label mapping
        if self.mode == 'val':
            label = self.label_mapping(label, self.mapping)

        ###################################################################################################################################
        flip_type = 'none'
        if self.split == 'train':
            # Random horizontal flip
            if random.random() > 0.5:
                image = np.flip(image, axis=1)
                label = np.flip(label, axis=1)
                flip_type = 'horizontal'

            # Random vertical flip
            if random.random() > 0.5:
                image = np.flip(image, axis=0)
                label = np.flip(label, axis=0)
                if flip_type == 'none':
                    flip_type = 'vertical'
                elif flip_type == 'horizontal':
                    flip_type = 'ho_ver'           

        point_label_list = []
        if self.use_points:
            point_label_list = self.point_labels[index]
            # print('point_label_list_1: ', point_label_list)
            if flip_type != 'none':
                for p in point_label_list:
                    if flip_type == 'horizontal' or flip_type == 'ho_ver':
                        # Flip x coordinate
                        p[1] = self.size[1] - 1 - p[1]
                    if flip_type == 'vertical' or flip_type == 'ho_ver':
                        # Flip y coordinate
                        p[0] = self.size[0] - 1 - p[0]

            tmp_label = Image.fromarray(label.astype('uint8')).resize(self.size, Image.NEAREST)
            tmp_label = np.asarray(tmp_label, np.uint8)
            categories = []
            # print('point_label_list_2: ', point_label_list)
            for i, p in enumerate(point_label_list):
                categories.append(tmp_label[tuple(p)])
            point_label_list = np.concatenate([np.array(point_label_list),
                                               np.array(categories).reshape(-1, 1)], axis=1)
            # print(name, point_label_list)

        # convert label for dilation
        label_for_dilation = label.astype(np.uint8)
        # define the dilation kernel
        kernel = np.ones((6,6), np.uint8)
        # apply dilation
        dilated_label = cv2.dilate(label_for_dilation, kernel, iterations=1)
        # convert dilated_label back to float32
        dilated_label = dilated_label.astype(np.float32)
        ##############################################################################################################################

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        if self.mode == 'val':
            return image.copy(), dilated_label.copy(), np.array(size), name, point_label_list.copy()  ##################
        else:
            return image.copy(), np.array(size), name

    def label_mapping(self, input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int64)
