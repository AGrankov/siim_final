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
import cv2

from sklearn.model_selection import train_test_split

from scripts.utils.utils import getListOfFiles
from scripts.utils.metrics import dice, correct_dice
from scripts.utils.mask_functions import mask2rle

from scripts.models.unet_resnet34 import UnetResnet34
from scripts.dataset.he_dataset import HeSegmentationDataset
from scripts.dataset.raw_dataset import RawSegmentationDataset
import segmentation_models_pytorch as smp

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
torch.backends.cudnn.benchmark = True


def resize_preds(preds):
    preds_resized = np.zeros((len(preds), 1024, 1024), dtype=np.float32)
    for i in tqdm(range(len(preds))):
        resized_image = cv2.resize(preds[i], (1024, 1024))
        preds_resized[i] = resized_image
    return preds_resized


def filter_small_masks(mask, threshold_size=250):
    labled, n_objs = ndimage.label(mask)
    result = np.zeros_like(mask, dtype=np.int8)
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if obj.sum() > threshold_size:
            result[obj > 0] = 1
    return result


def filter_all_masks(mask):
    labled, n_objs = ndimage.label(mask)
    sizes = ndimage.sum(mask, labled, range(n_objs + 1))
    res_idx = np.argmax(sizes)
    return (labled == res_idx).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Evaluate unet resnet34 clean')

    parser.add_argument('-i', '--input', default='../input/dicom-images-train', help='input eval data directory')
    parser.add_argument('-id', '--input_df', default='../input/train-rle.csv', help='input train df file')

    parser.add_argument('-s', '--seed', default=42, help='seed')
    parser.add_argument('-tfr', '--test_fold_ratio', default=0.2, help='test fold ratio')

    parser.add_argument('-mp', '--model_path', default='../models/unet_resnet34_1024_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/unet_resnet34_960_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/smp_ur34_1024_v1/', help='path to models directory')
    parser.add_argument('-bs', '--batch_size', default=2, help='size of batches', type=int)
    parser.add_argument('-w', '--workers', default=6, help='data loader wokers count', type=int)
    parser.add_argument('-is', '--image_size', default=1024, help='image size', type=int)

    args = parser.parse_args()

    df = pd.read_csv(args.input_df)
    ids_with_mask = set(df[df[' EncodedPixels'].str.strip() != '-1']['ImageId'].values)

    # # # # # # # # # Model loading # # # # # # # # #
    checkpoint = torch.load(os.path.join(args.model_path, 'best_dice_model.t7'))

    model = UnetResnet34(
        in_channels=3,
        num_classes=1,
        num_filters=32,
        pretrained=False,
        is_deconv=True
    )
    # model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    current_size = int(args.image_size)

    norm = Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.225, 0.225, 0.225],
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    clf_preds_1 = pd.read_csv('../p_input/binary/efficientnet0_1024_d75_he_v1_raw_10xTTA.csv')
    clf_preds_2 = pd.read_csv('../p_input/binary/efn5_512_d90_he_v1_raw_10xTTA.csv')
    clf_preds_3 = pd.read_csv('../p_input/binary/efn0_1024_d75_add_v1_10xTTA.csv')
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

    # test_all_files_set = set(clf_preds_2['ImageId'].values[clf_mean > 0.5])
    test_all_files_set = set(clf_preds_2['ImageId'].values[clf_mean > 0.356])
    # test_all_files_set = set(clf_preds_2['ImageId'].values[clf_mean > 0.289])

    leak_df = pd.read_csv('../input/leaky_raddar/leak_probabilities.csv')
    predict_for = set(leak_df[leak_df['pred'] > 0.7]['ImageId'].values)
    test_all_files_set.update(predict_for)

    test_all_files = getListOfFiles('../input/dicom-images-test')
    test_all_files = [x for x in test_all_files if os.path.split(x)[1][:-4] in test_all_files_set]
    test_all_files = np.array(sorted(test_all_files))
    print('len of test_all_files {}'.format(len(test_all_files)))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    test_dataset_base = functools.partial(
        HeSegmentationDataset,
        dcm_files=test_all_files,
    )
    # test_dataset_base = functools.partial(
    #     RawSegmentationDataset,
    #     dcm_files=test_all_files,
    # )

    test_predictor = TTAPredictor(
        model=model,
        ds_base=test_dataset_base,
        is_validation=False,
        batch_size=args.batch_size,
        workers=args.workers,
        base_transforms=[],
        # base_transforms=[norm],
        put_deafult=False
    )

    test_predictor.put(
        [Resize(current_size, current_size)],
        None
    )
    test_predictor.put(
        [Resize(current_size, current_size), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True)])
    )

    test_predictor.put(
        [Resize(int(current_size * 1.25), int(current_size * 1.25))],
        Resize(current_size, current_size)
    )
    test_predictor.put(
        [Resize(int(current_size * 0.75), int(current_size * 0.75))],
        Resize(current_size, current_size)
    )

    test_predictor.put(
        [HorizontalFlip(always_apply=True), Resize(int(current_size * 1.25), int(current_size * 1.25))],
        Compose([HorizontalFlip(always_apply=True), Resize(current_size, current_size)])
    )
    test_predictor.put(
        [HorizontalFlip(always_apply=True), Resize(int(current_size * 0.75), int(current_size * 0.75))],
        Compose([HorizontalFlip(always_apply=True), Resize(current_size, current_size)])
    )

    test_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(-15,-15), always_apply=True)],
        Rotate(limit=(15,15), always_apply=True)
    )
    test_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(15,15), always_apply=True)],
        Rotate(limit=(-15,-15), always_apply=True)
    )

    test_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(-15,-15), always_apply=True), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True), Rotate(limit=(15,15), always_apply=True)])
    )
    test_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(15,15), always_apply=True), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True), Rotate(limit=(-15,-15), always_apply=True)])
    )

    test_preds = []
    for pred, _ in tqdm(test_predictor):
        test_preds.extend(pred)
    test_preds = np.array(test_preds)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_preds_resnet34_1024_v1 = test_preds
    # test_preds_resnet34_960_v1 = test_preds
    # test_preds_smp_ur34_1024_v1 = test_preds
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    old_ss = pd.read_csv('../input/sample_submission_old.csv')
    ss_uc = old_ss.groupby(['ImageId'])['EncodedPixels'].count()
    uc_sure = set(ss_uc[ss_uc > 1].index.values)
    idxes_of_sure = np.array([idx for idx, x in enumerate(test_all_files) if os.path.split(x)[1][:-4] in uc_sure])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # test_preds = np.stack((test_preds_resnet34_1024_v1, test_preds_resnet34_960_v1)).mean(axis=0)
    tmp_preds = np.stack((test_preds_resnet34_1024_v1, test_preds_resnet34_960_v1)).mean(axis=0)
    test_preds = np.stack((tmp_preds, test_preds_smp_ur34_1024_v1)).mean(axis=0)

    # args.output = '../submissions/clx5_0.356_ur34_1024_v1_ur34_960_v1_0.8_f250_0.34.csv.gz'
    # args.output = '../submissions/clx5_0.356_ur34_1024_v1_ur34_960_v1_smp_ur34_1024_v1_0.65_f1500_0.32.csv.gz'
    # args.output = '../submissions/clx5_0.356_ur34_1024_v1_ur34_960_v1_smp_ur34_1024_v1_0.8_f250_0.32_si0.10.csv.gz'
    args.output = '../submissions/clx5_0.356_leaky_cl_0.7_ur34_1024_v1_ur34_960_v1_smp_ur34_1024_v1_0.65_f1500_0.32_si0.10.csv.gz'

    res_preds = (test_preds > 0.65).astype(np.uint8)
    res_preds = [filter_small_masks(x, 1500) for x in tqdm(res_preds)]

    print('empty count before {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    for idx in range(len(res_preds)):
        if res_preds[idx].sum() > 0:
            res_preds[idx] = (test_preds[idx] > 0.32).astype(np.uint8)
    print('res empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    print('res non empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) > 0).sum()))

    tmp_idxes = sorted(list(set(idxes_of_sure) & set(np.argwhere(np.sum(res_preds, axis=1).sum(axis=1) == 0).flatten())))
    for idx in tmp_idxes:
        res_preds[idx] = (test_preds[idx] > 0.1).astype(np.uint8)

    print('res empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    print('res non empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) > 0).sum()))

    # leaky_idxes = np.array([idx for idx, x in enumerate(test_all_files) if os.path.split(x)[1][:-4] in predict_for])
    # for idx in leaky_idxes:
    #     res_preds[idx] = (test_preds[idx] > 0.1).astype(np.uint8)
    #
    # print('res empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    # print('res non empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) > 0).sum()))

    # Optimal count for stage 1 test is 290 pneumothorax xrays

    res_rle = [mask2rle(x.T * 255, x.shape[0], x.shape[1]) if x.sum() > 0 else '-1' for x in tqdm(res_preds)]

    res_df = pd.read_csv('../input/sample_submission.csv')
    pred_df = pd.DataFrame(data=
    {'ImageId': [os.path.split(x)[1][:-4] for x in test_all_files],
     'preds': res_rle}
    )
    res_df = res_df.merge(pred_df, how='left', on='ImageId')
    res_df.loc[pd.notna(res_df['preds']), 'EncodedPixels'] = res_df[pd.notna(res_df['preds'])]['preds']
    res_df.drop('preds', 1, inplace=True)
    # args.output = '../submissions/tmp.csv.gz'
    res_df.to_csv(args.output, index=False, compression='gzip')
    # res_df = pd.read_csv(args.output)

    os.system('kaggle competitions submit siim-acr-pneumothorax-segmentation -f {} -m "{}"'.format(args.output, "{}_fold_seed_{}".format(args.test_fold_ratio, args.seed)))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    main()
