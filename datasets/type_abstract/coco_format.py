import os

import numpy as np

from datasets.type_abstract.basic import BasicDataset
from pycocotools.coco import COCO


class DatasetCOCOFormat(BasicDataset):
    def __init__(self, data_cfg):
        """
        basic dataset implementation
        rewrite getitem
        """
        super(DatasetCOCOFormat, self).__init__(data_cfg)

        ann_file = os.path.join(self.data_cfg.root, self.data_cfg.data_type, 'annotations.json')
        self.coco = COCO(ann_file)
        self.fetch_id_type = self.data_cfg.get('fetch_id_type', 'images')
        self.anns = sorted(self.coco.getImgIds())
        self.anns = np.array([ann for ann in self.anns if len(self.coco.getAnnIds(imgIds=ann, iscrowd=None))])
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.img_path = os.path.join(self.data_cfg.root, self.data_cfg.data_type, 'images')
        self.effective_percent = self.data_cfg.get('effective_percent', 1)

    def load_coco(self, ann_file_path):
        return COCO(ann_file_path)

    def __len__(self):
        # Use a small part to explore a new dataset/method
        if self.effective_percent > 1:
            return self.effective_percent
        return int(len(self.coco.dataset[self.fetch_id_type]) * self.effective_percent)

    def fetch_by_images(self, fetch_id):
        img_id = fetch_id
        annotations_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        annotations = self.coco.loadAnns(annotations_ids)
        return img_id, annotations

    def fetch_by_annotations(self, fetch_id):
        annos = [self.coco.dataset['annotations'][fetch_id]]
        img_id = annos[0]['image_id']
        return img_id, annos

    def process_info(self, fetch_id):
        return self.__getattribute__(f'fetch_by_{self.fetch_id_type}')(fetch_id)
