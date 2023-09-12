import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gc
import functools
from scipy import ndimage

from scripts.utils.utils import getListOfFiles
from scripts.utils.mask_functions import mask2rle

from scripts.models.unet_resnet34 import UnetResnet34
import segmentation_models_pytorch as smp
from scripts.dataset.he_dataset import HeSegmentationDataset
from scripts.dataset.raw_dataset import RawSegmentationDataset

from scripts.predicting.tta_predictor import TTAPredictor

from albumentations import (
    Resize,
    Normalize,
    HorizontalFlip,
    Crop,
    Compose,
    Rotate,
    PadIfNeeded,
    CenterCrop
)

tqdm.monitor_interval = 0


def filter_small_masks(mask, threshold_size=250):
    labled, n_objs = ndimage.label(mask)
    result = np.zeros_like(mask, dtype=np.int8)
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if obj.sum() > threshold_size:
            result[obj > 0] = 1
    return result


def predict_seg(args, model_checkpoint, base_model, dataset_base, base_transforms):
    checkpoint = torch.load(model_checkpoint)

    model = torch.nn.DataParallel(base_model).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()

    tta_predictor = TTAPredictor(
        model=model,
        ds_base=dataset_base,
        batch_size=args.batch_size,
        workers=args.workers,
        base_transforms=base_transforms,
        put_deafult=False
    )

    current_size = int(args.image_size)

    tta_predictor.put(
        [Resize(current_size, current_size)],
        None
    )
    tta_predictor.put(
        [Resize(current_size, current_size), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True)])
    )

    tta_predictor.put(
        [Resize(int(current_size * 1.25), int(current_size * 1.25))],
        Resize(current_size, current_size)
    )
    tta_predictor.put(
        [Resize(int(current_size * 0.75), int(current_size * 0.75))],
        Resize(current_size, current_size)
    )

    tta_predictor.put(
        [HorizontalFlip(always_apply=True), Resize(int(current_size * 1.25), int(current_size * 1.25))],
        Compose([HorizontalFlip(always_apply=True), Resize(current_size, current_size)])
    )
    tta_predictor.put(
        [HorizontalFlip(always_apply=True), Resize(int(current_size * 0.75), int(current_size * 0.75))],
        Compose([HorizontalFlip(always_apply=True), Resize(current_size, current_size)])
    )

    tta_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(-15,-15), always_apply=True)],
        Rotate(limit=(15,15), always_apply=True)
    )
    tta_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(15,15), always_apply=True)],
        Rotate(limit=(-15,-15), always_apply=True)
    )

    tta_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(-15,-15), always_apply=True), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True), Rotate(limit=(15,15), always_apply=True)])
    )
    tta_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(15,15), always_apply=True), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True), Rotate(limit=(-15,-15), always_apply=True)])
    )

    res_preds = []
    res_targets = []
    for pred, targets in tqdm(tta_predictor):
        res_preds.extend(pred)
        if targets is not None:
            res_targets.extend(targets)
    res_preds = np.array(res_preds)
    if len(res_targets) > 0:
        res_targets = np.array(res_targets)

    del tta_predictor
    gc.collect()

    return res_preds, res_targets


def main():
    parser = argparse.ArgumentParser(description='Predict test segmentation')

    parser.add_argument('-i', '--input', default='../input/dicom-images-test', help='input data directory')
    parser.add_argument('-ss', '--sample_submission', default='../input/sample_submission.csv', help='sample submission file')
    parser.add_argument('-o', '--output', default='../submissions/f1.csv.gz', help='output file')

    parser.add_argument('-bs', '--batch_size', default=4, help='size of batches', type=int)
    parser.add_argument('-w', '--workers', default=4, help='data loader wokers count', type=int)
    parser.add_argument('-is', '--image_size', default=1024, help='image size', type=int)

    parser.add_argument('-st', '--segmentation_threshold', default=0.65, help='threshold for segmentation', type=float)
    parser.add_argument('-mft', '--mask_filter_threshold', default=1500, help='threshold for mask size filtering', type=int)

    parser.add_argument('-uk', '--upload_kaggle', default=False, help='Is upload result to kaggle automatically', type=bool)

    args = parser.parse_args()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    clf_preds_1 = pd.read_csv('../p_input/binary/efn0_1024_d75_he_v1.csv')
    clf_preds_2 = pd.read_csv('../p_input/binary/efn0_1024_d75_add_v1.csv')
    clf_preds_3 = pd.read_csv('../p_input/binary/efn5_512_d90_he_v1.csv')
    clf_preds_4 = pd.read_csv('../p_input/binary/efn0_256_d80_he_cropped_v1.csv')
    clf_preds_5 = pd.read_csv('../p_input/binary/efn0_512_aug_dynamic_v1.csv')

    clf_tmp_1 = np.stack((
        clf_preds_1['raw_target'].values,
        clf_preds_2['raw_target'].values,
        clf_preds_3['raw_target'].values
    )).mean(axis=0)

    clf_tmp_2 = np.stack((
        clf_preds_4['raw_target'].values,
        clf_preds_5['raw_target'].values,
    )).mean(axis=0)

    clf_mean = np.stack((
        clf_tmp_1,
        clf_tmp_2,
    )).mean(axis=0)

    test_all_files_set = set(clf_preds_1['ImageId'].values[clf_mean > 0.356])

    test_all_files = getListOfFiles(args.input)
    test_all_files = [x for x in test_all_files if os.path.split(x)[1][:-4] in test_all_files_set]
    test_all_files = np.array(sorted(test_all_files))
    print('len of test_all_files {}'.format(len(test_all_files)))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ur34_1024_v1_checkpoint_name = '../models/unet_resnet34_1024_v1/best_dice_model.t7'
    ur34_1024_v1_base_model = UnetResnet34(in_channels=3, num_classes=1, num_filters=32, pretrained=False, is_deconv=True)
    ur34_1024_v1_test_dataset_base = functools.partial(
        HeSegmentationDataset,
        dcm_files=test_all_files,
    )
    ur34_1024_v1_base_transformations = []

    ur34_960_v1_checkpoint_name = '../models/unet_resnet34_960_v1/best_dice_model.t7'
    ur34_960_v1_base_model = ur34_1024_v1_base_model
    ur34_960_v1_test_dataset_base = ur34_1024_v1_test_dataset_base
    ur34_960_v1_base_transformations = ur34_1024_v1_base_transformations

    smp_ur34_1024_v1_checkpoint_name = '../models/smp_ur34_1024_v1/best_dice_model.t7'
    smp_ur34_1024_v1_base_model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    smp_ur34_1024_v1_test_dataset_base = functools.partial(
        RawSegmentationDataset,
        dcm_files=test_all_files,
    )
    smp_ur34_1024_v1_base_transformations = [norm]
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print("Predicting unet resnet 34 image size 1024 v1")
    ur34_1024_v1_test_res, _ = predict_seg(
        args,
        ur34_1024_v1_checkpoint_name,
        ur34_1024_v1_base_model,
        ur34_1024_v1_test_dataset_base,
        ur34_1024_v1_base_transformations
    )

    print("Predicting unet resnet 34 image size 960 v1")
    ur34_960_v1_test_res, _ = predict_seg(
        args,
        ur34_960_v1_checkpoint_name,
        ur34_960_v1_base_model,
        ur34_960_v1_test_dataset_base,
        ur34_960_v1_base_transformations
    )

    print("Predicting smp unet resnet 34 image size 1024 v1")
    smp_ur34_1024_v1_test_res, _ = predict_seg(
        args,
        smp_ur34_1024_v1_checkpoint_name,
        smp_ur34_1024_v1_base_model,
        smp_ur34_1024_v1_test_dataset_base,
        smp_ur34_1024_v1_base_transformations
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    tmp_preds = np.stack((ur34_1024_v1_test_res, ur34_960_v1_test_res)).mean(axis=0)
    test_preds = np.stack((tmp_preds, smp_ur34_1024_v1_test_res)).mean(axis=0)

    res_preds = (test_preds > args.segmentation_threshold).astype(np.uint8)
    res_preds = [filter_small_masks(x, args.mask_filter_threshold) for x in res_preds]

    for idx in range(len(res_preds)):
        if res_preds[idx].sum() > 0:
            res_preds[idx] = (test_preds[idx] > 0.32).astype(np.uint8)

    print('Converting results to rle encoding')
    res_rle = [mask2rle(x.T * 255, x.shape[0], x.shape[1]) if x.sum() > 0 else '-1' for x in tqdm(res_preds)]

    res_df = pd.read_csv(args.sample_submission)
    pred_df = pd.DataFrame(data=
    {'ImageId': [os.path.split(x)[1][:-4] for x in test_all_files],
     'preds': res_rle}
    )
    res_df = res_df.merge(pred_df, how='left', on='ImageId')
    res_df.loc[pd.notna(res_df['preds']), 'EncodedPixels'] = res_df[pd.notna(res_df['preds'])]['preds']
    res_df.drop('preds', 1, inplace=True)
    res_df.to_csv(args.output, index=False, compression='gzip')

    if args.upload_kaggle:
        os.system('kaggle competitions submit siim-acr-pneumothorax-segmentation -f {} -m "{}"'.format(args.output, "message"))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    main()
