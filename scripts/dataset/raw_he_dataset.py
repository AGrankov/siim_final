import sys
sys.path.append('..')

import cv2
from torch.utils import data
import os
import numpy as np
import pandas as pd
import pydicom
from scripts.utils.mask_functions import read_flat_mask


class RawHeSegmentationDataset(data.Dataset):
    def __init__(self, dcm_files, masks_file=None, transform=None):
        self.all_files = dcm_files

        if masks_file is not None:
            df = pd.read_csv(masks_file)
            df.columns = [x.strip() for x in df.columns]
            df.drop(df[df['EncodedPixels'].str.strip() == '-1'].index,
                    axis=0,
                    inplace=True)
            df = df.reset_index(drop=True)
            self.masks_df = df
        else:
            self.masks_df = None

        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        img_id = os.path.split(self.all_files[index])[1][:-4]

        with pydicom.dcmread(self.all_files[index]) as img_data:
            img = img_data.pixel_array.copy()

        equ = cv2.equalizeHist(img)

        img = equ
        img = np.expand_dims(img, 2)

        mask = None
        if self.masks_df is not None:
            mask = read_flat_mask(img_id, self.masks_df, shape=img.shape[:2])
            mask = mask * 255

        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=img)
                img = augmented['image']

        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255

        if mask is not None:
            mask = mask // 255
            return img, mask

        return img
