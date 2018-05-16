from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize
from PIL import Image

from dataloaders.helpers import *
from torch.utils.data import Dataset


class DAVIS2017(Dataset):
    """DAVIS 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='data/davis',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None,
                 inst=1,
                 count=5):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        self.inst = inst
        self.count = count

        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'

        if self.seq_name is None:

            # Initialize the original DAVIS splits for training the parent network
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else:

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
            labels.extend([None]*(len(names_img)-1))
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        # for fine-tuning from sparse annotations (selected on load)
        self.sparse_label = None

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = Image.open(os.path.join(self.db_root_dir, self.labels[idx]))

        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.uint8)
            # mask out given instance in annotation
            gt[np.logical_and(gt != self.inst, gt != 255) ] = 0
            gt[gt == self.inst] = 1

            # make fixed, sparse label
            # label will be mostly ignore (255),
            # with sparse positive (1) and negative (0) annotations
            # TODO(shelhamer) set sparsity with flag
            if self.sparse_label is None:
                self.sparse_label = np.full_like(gt, 255)
                for lbl in (0, 1):
                    mask_idx = np.where(gt.flat == lbl)[0]
                    sparse_idx = np.random.choice(mask_idx, size=self.count, replace=False)
                    self.sparse_label.flat[sparse_idx] = lbl
            gt = self.sparse_label

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    dataset = DAVIS2017(db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2017',
                        train=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        if i == 10:
            break

    plt.show(block=True)
