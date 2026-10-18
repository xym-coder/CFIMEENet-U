import re
import os
import numpy as np
from tool_functions import format_transform
from dataloader.imageloader import pil_load_img
from dataloader.dataloader_base import TextDataset, TextInstance
from tool_functions.file_tool_functions import read_lines

class Uyghur(TextDataset):
    def __init__(
        self,
        data_root,
        is_training=True,
        load_memory=False,
        transform=None,
        ignore_list=None
    ):

        super().__init__(transform, is_training)

        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(
            data_root,
            'Train' if is_training else 'Test'
        )

        self.annotation_root = os.path.join(
            data_root,
            'Train' if is_training else 'Test'
        )

        self.image_list = os.listdir(self.image_root)

        p = re.compile('.rar|.txt')
        self.image_list = [
            x for x in self.image_list if not p.findall(x)
        ]

        p = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        self.annotation_list = [
            '{}'.format(p.sub("", img_name))
            for img_name in self.image_list
        ]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @staticmethod
    def parse_txt(gt_path):

        lines = read_lines(gt_path + ".txt")

        polygons = []

        for line in lines:

            line = format_transform.remove_all(
                line.strip('\ufeff'),
                '\xef\xbb\xbf'
            )

            gt = line.split(',')

            x1, y1, x2, y2, x3, y3, x4, y4 = list(
                map(int, gt[:8])
            )

            xx = [x1, x2, x3, x4]

            yy = [y1, y2, y3, y4]

            label = gt[-1].strip().replace("###", "#")

            pts = np.stack([xx, yy]).T.astype(np.int32)

            polygons.append(TextInstance(pts, 'c', label))

        return polygons

    def load_img_gt(self, item):

        image_id = self.image_list[item]

        image_path = os.path.join(self.image_root, image_id)

        image = pil_load_img(image_path)

        try:

            annotation_id = self.annotation_list[item]

            annotation_path = os.path.join(
                self.annotation_root,
                annotation_id
            )

            polygons = self.parse_txt(annotation_path)

        except Exception as e:

            print(e)
            polygons = None

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id.split("/")[-1]
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):

        if self.load_memory:
            data = self.datas[item]

        else:
            data = self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(
                data["image"],
                data["polygons"],
                image_id=data["image_id"],
                image_path=data["image_path"]
            )

        else:
            return self.get_test_data(
                data["image"],
                data["polygons"],
                image_id=data["image_id"],
                image_path=data["image_path"]
            )

    def __len__(self):

        return len(self.image_list)


if __name__ == '__main__':
    import cv2
    from tool_functions.augmentation import Augmentation
    from tool_functions.visualize import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)

    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640,
        mean=means,
        std=stds
    )

    trainset = Uyghur(
        data_root='../data/Uyghur',
        is_training=True,
        transform=transform
    )

    for idx in range(0, len(trainset)):
        t0 = time.time()

        img, train_mask, tr_mask = trainset[idx]

        img, train_mask, tr_mask = map(
            lambda x: x.cpu().numpy(),
            (img, train_mask, tr_mask)
        )

        img = img.transpose(1, 2, 0)

        img = ((img * stds + means) * 255).astype(np.uint8)

        print(idx, img.shape)

        for i in range(tr_mask.shape[2]):
            cv2.imshow(
                "tr_mask_{}".format(i),
                cav.heatmap(
                    np.array(
                        tr_mask[:, :, i] * 255 / np.max(tr_mask[:, :, i]),
                        dtype=np.uint8
                    )
                )
            )

        cv2.imshow(
            "train_mask",
            cav.heatmap(
                np.array(
                    train_mask * 255 / np.max(train_mask),
                    dtype=np.uint8
                )
            )
        )

        cv2.imshow('imgs', img)

        cv2.waitKey(0)