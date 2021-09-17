#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np

from collections import defaultdict

from pycocotools.coco import COCO

import sys
sys.path.insert(0, "/home/yexuan/tensorbay-python-sdk")

from tensorbay import GAS
from tensorbay.dataset import Dataset as TensorBayDataset
from tensorbay.dataset import Segment

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class COCODataset(Dataset):
    """COCO dataset class."""

    def __init__(
        self,
        data_dir=None,
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        *,
        gas: GAS,
    ):
        """COCO dataset initialization. Annotation data are read into memory by COCO API.

        Args:
            data_dir (str): dataset root directory
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.imgs = None
        self.name = name

        coco_dataset = gas.get_dataset("COCO2017", use_cache=True)
        # if name == "train":
        #     self.coco_segment = Segment(name, coco_dataset)[:10000]
        # else:
        self.coco_segment = Segment(name, coco_dataset)
        catalog = coco_dataset.get_catalog()
        subcatalog = catalog.box2d
        self.category_to_index = subcatalog.get_category_to_index()
        self.categories = catalog.panoptic_mask.categories[1:81]
        self.class_ids = sorted(category.category_id for category in self.categories)
        self.img_size = img_size
        self.preproc = preproc
        self.all_data = []
        # self._load_all_data()
        if cache:
            self._cache_images()

    def __len__(self) -> int:
        return len(self.coco_segment)

    def __del__(self) -> None:
        # del self.all_data
        del self.coco_segment

    def _get_anno_num(self) -> int:
        return len(self.all_data)

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.coco_segment), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            anno_num = self._get_anno_num()
            loaded_images = ThreadPool(NUM_THREADs).imap( lambda x: self.load_resized_img(x),
                range(anno_num),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(anno_num))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.coco_segment), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def _load_all_data(self):
        for data in self.coco_segment:
            img, height, width = self._load_resized_data_img(data)
            res, img_info, resized_info = self._load_anno(data, height, width)
            self.all_data.append((res, img_info, resized_info, img))

    def load_anno(self, index):
        # return self.all_data[index][0]
        return [box2d for box2d in self.coco_segment[index].label.box2d if self.category_to_index[box2d.category] <80]

    def _load_anno(self, data, height, width):
        objs = []
        for bbox in data.label.box2d:
            index = self.category_to_index[bbox.category]
            if index >= 80:
                continue
            x1 = np.max((0, bbox[0]))
            y1 = np.max((0, bbox[1]))
            x2 = np.min((width, bbox[2]))
            y2 = np.min((height, bbox[3]))
            if bbox.area() > 0 and x2 >= x1 and y2 >= y1:
                objs.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "index": index
                    }
                )

        res = np.zeros((len(objs), 5))
        for idx, obj in enumerate(objs):
            res[idx, :4] = obj["bbox"]
            res[idx, 4] = obj["index"]

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        return res, img_info, resized_info

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        data = self.coco_segment[index]
        buf = np.asarray(bytearray(data.open().read()), dtype="uint8")
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert image is not None
        return image

    def _load_resized_data_img(self, data):
        img = self.load_data_image(data)
        height, width = img.shape[:2]
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img, height, width

    def load_data_image(self, data):
        buf = np.asarray(bytearray(data.open().read()), dtype="uint8")
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert image is not None
        return image

    def pull_item(self, index):
        data = self.coco_segment[index]
        img, height, width = self._load_resized_data_img(data)
        res, img_info, resized_info = self._load_anno(data, height, width)
        # res, img_info, resized_info, img = self.all_data[index]
        return img, res.copy(), img_info, np.array([self._get_image_id(index)])

    def _get_image_id(self, index):
        return int(os.path.splitext(os.path.basename(self.coco_segment[index].path))[0])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

    def to_coco(self) -> COCO:
        coco = COCO()
        self.create_coco_index(coco)
        return coco

    def create_coco_index(self, coco: COCO) -> None:
        print('creating coco index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        images, annotations, categories = [], [], []

        ann_count = 0
        for data in self.coco_segment:
            image_path = os.path.basename(data.path)
            image_id = int(os.path.splitext(image_path)[0])
            image_info = {
                "file_name": image_path,
                "id": image_id
            }
            images.append(image_info)
            imgs[image_id] = image_info
            for bbox in data.label.box2d:
                category_id = self.category_to_index[bbox.category]
                if category_id >= 80:
                    continue
                ann_count += 1
                coco_category_id = self.class_ids[category_id]
                ann = {
                    "area": bbox.area(),
                    "iscrowd": int(bool(data.label.rle)),
                    "image_id": image_id,
                    "bbox": [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
                    "category_id": coco_category_id,
                    "id": ann_count
                }
                imgToAnns[image_id].append(ann)
                catToImgs[coco_category_id].append(image_id)
                anns[ann_count] = ann
                annotations.append(ann)

        for category in self.categories:
            supercategory, name = category.name.rsplit(".", 1)
            coco_category = {
                "supercategory": supercategory,
                "id": category.category_id,
                "name": name,
            }
            categories.append(coco_category)
            cats[category.category_id] = coco_category

        coco.anns = anns
        coco.imgToAnns = imgToAnns
        coco.catToImgs = catToImgs
        coco.imgs = imgs
        coco.cats = cats
        coco.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with open("/home/yexuan/workspace/training/coco_val.json", "w") as fp:
            import json
            json.dump(coco.dataset, fp, indent=4)

        print('coco index created!')
