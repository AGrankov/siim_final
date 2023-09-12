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

from scripts.models.unet_resnet50 import UnetResnet50
from scripts.models.unet_resnet34 import UnetResnet34
from scripts.models.pspnet import PSPNet, PSPNetModified
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
    CenterCrop,
    CLAHE,
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
    parser = argparse.ArgumentParser(description='Evaluate unet resnet50')

    parser.add_argument('-i', '--input', default='../input/dicom-images-train', help='input eval data directory')
    parser.add_argument('-id', '--input_df', default='../input/train-rle.csv', help='input train df file')
    # parser.add_argument('-br', '--binary_result', default='../p_input/binary/efficientnet0_1024_d75_he_v1_0.8.csv', help='binary result file')
    # parser.add_argument('-o', '--output', default='../submissions/efn0_1024_d75_he_v1_10xTTA_efn5_512_d90_v1_10xTTA_efn0_1024_d75_add_v1_10xTTA_0.5_leaky_cl_0.7_unet_resnet50_1024_v3_th0.6_rs42_seed42_f250_10xTTA.csv.gz', help='output file')
    # parser.add_argument('-rcp', '--raw_classifier_preds', default='../p_input/lgb_efn0_1024_d75_he_v1_efn5_512_d90_he_v1_leaky_preds.csv', help='raw classifier predictions')

    parser.add_argument('-s', '--seed', default=42, help='seed')
    parser.add_argument('-tfr', '--test_fold_ratio', default=0.2, help='test fold ratio')

    # parser.add_argument('-mp', '--model_path', default='../models/unet_resnet50_1024_v2/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/unet_resnet50_1024_v3/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/pspnet_512_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/pspnet_512_v4/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/unet_resnet34_1024_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/unet_resnet34_960_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/pspnet_1024_v1/', help='path to models directory')
    parser.add_argument('-mp', '--model_path', default='../models/smp_ur34_1024_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/smp_ur34_1024_v2/', help='path to models directory')
    # parser.add_argument('-bs', '--batch_size', default=2, help='size of batches', type=int)
    # parser.add_argument('-bs', '--batch_size', default=4, help='size of batches', type=int)
    parser.add_argument('-bs', '--batch_size', default=8, help='size of batches', type=int)
    parser.add_argument('-w', '--workers', default=6, help='data loader wokers count', type=int)
    # parser.add_argument('-bs', '--batch_size', default=2, help='size of batches', type=int)
    # parser.add_argument('-w', '--workers', default=2, help='data loader wokers count', type=int)
    parser.add_argument('-is', '--image_size', default=1024, help='image size', type=int)
    # parser.add_argument('-is', '--image_size', default=960, help='image size', type=int)
    # parser.add_argument('-is', '--image_size', default=512, help='image size', type=int)
    # parser.add_argument('-is', '--image_size', default=256, help='image size', type=int)

    args = parser.parse_args()


    df = pd.read_csv(args.input_df)
    ids_with_mask = set(df[df[' EncodedPixels'].str.strip() != '-1']['ImageId'].values)

    # # # # # # # # # Model loading # # # # # # # # #
    checkpoint = torch.load(os.path.join(args.model_path, 'best_dice_model.t7'))
    # checkpoint = torch.load(os.path.join(args.model_path, 'best_loss_model.t7'))

    # model = UnetResnet50(
    #     in_channels=3,
    #     # in_channels=1,
    #     num_classes=1,
    #     num_filters=16,
    #     pretrained=False,
    #     is_deconv=True
    # )
    # model = PSPNet(
    # # model = PSPNetModified(
    #     num_classes=1,
    #     backbone='resnet50',
    #     pretrained=True,
    #     is_distributed=False
    # )
    # model = UnetResnet34(
    #     in_channels=3,
    #     num_classes=1,
    #     num_filters=32,
    #     pretrained=False,
    #     is_deconv=True
    # )
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
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

    all_files = getListOfFiles(args.input)
    all_files = [x for x in all_files if os.path.split(x)[1][:-4] in ids_with_mask]
    all_files = np.array(all_files)

    _, eval_files = train_test_split(
        all_files,
        test_size=args.test_fold_ratio,
        random_state=args.seed
    )

    # eval_dataset_base = functools.partial(
    #     HeSegmentationDataset,
    #     dcm_files=eval_files,
    #     masks_file=args.input_df,
    # )
    eval_dataset_base = functools.partial(
        RawSegmentationDataset,
        dcm_files=eval_files,
        masks_file=args.input_df,
    )

    eval_predictor = TTAPredictor(
        model=model,
        ds_base=eval_dataset_base,
        batch_size=args.batch_size,
        workers=args.workers,
        # base_transforms=[],
        base_transforms=[norm],
        put_deafult=False
    )

    eval_predictor.put(
        [Resize(current_size, current_size)],
        None
    )
    eval_predictor.put(
        [Resize(current_size, current_size), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True)])
    )

    eval_predictor.put(
        [Resize(int(current_size * 1.25), int(current_size * 1.25))],
        Resize(current_size, current_size)
    )
    eval_predictor.put(
        [Resize(int(current_size * 0.75), int(current_size * 0.75))],
        Resize(current_size, current_size)
    )

    eval_predictor.put(
        [HorizontalFlip(always_apply=True), Resize(int(current_size * 1.25), int(current_size * 1.25))],
        Compose([HorizontalFlip(always_apply=True), Resize(current_size, current_size)])
    )
    eval_predictor.put(
        [HorizontalFlip(always_apply=True), Resize(int(current_size * 0.75), int(current_size * 0.75))],
        Compose([HorizontalFlip(always_apply=True), Resize(current_size, current_size)])
    )

    eval_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(-15,-15), always_apply=True)],
        Rotate(limit=(15,15), always_apply=True)
    )
    eval_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(15,15), always_apply=True)],
        Rotate(limit=(-15,-15), always_apply=True)
    )

    eval_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(-15,-15), always_apply=True), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True), Rotate(limit=(15,15), always_apply=True)])
    )
    eval_predictor.put(
        [Resize(current_size, current_size), Rotate(limit=(15,15), always_apply=True), HorizontalFlip(always_apply=True)],
        Compose([HorizontalFlip(always_apply=True), Rotate(limit=(-15,-15), always_apply=True)])
    )

    eval_preds = []
    eval_targets = []
    for pred, targets in tqdm(eval_predictor):
        eval_preds.extend(pred)
        eval_targets.extend(targets)
    eval_preds = np.array(eval_preds)
    eval_targets = np.array(eval_targets)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # pspnet_512_v1_preds = eval_preds
    # pspnet_512_v4_preds = eval_preds
    #
    # resnet50_1024_v2_preds = eval_preds
    # resnet50_1024_v3_preds = eval_preds
    #
    # resnet34_1024_v1_preds = eval_preds
    # resnet34_960_v1_preds = eval_preds
    # smp_ur34_1024_v1_preds = eval_preds
    # smp_ur34_1024_v2_preds = eval_preds
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # pspnet_512_v1_preds_resized = resize_preds(pspnet_512_v1_preds)
    # pspnet_512_v4_preds_resized = resize_preds(pspnet_512_v4_preds)
    # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # #
    # # tmp_preds2 = np.stack((resnet50_1024_v2_preds, resnet50_1024_v3_preds)).mean(axis=0)
    # # eval_preds = np.stack((tmp_preds2, pspnet_512_preds_resized)).mean(axis=0)
    # # eval_preds = np.stack((pspnet_512_v1_preds, pspnet_512_v4_preds)).mean(axis=0)
    # eval_preds = np.stack((resnet50_1024_v2_preds, resnet50_1024_v3_preds, pspnet_512_v1_preds_resized, pspnet_512_v4_preds_resized)).mean(axis=0)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # import pickle as pkl
    # pkl.dump(eval_preds, open('../p_input/tmp_preds.pkl', 'wb'))
    # pkl.dump(eval_targets, open('../p_input/tmp_targets.pkl', 'wb'))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Eval best f1 score 0.8014777367295353 best treshold 0.255
    # Eval best acc score 0.9076346604215456 best treshold 0.365

    clf_preds_1 = pd.read_csv('../p_input/binary/efficientnet0_1024_d75_he_v1_raw_10xTTA.csv')
    clf_preds_2 = pd.read_csv('../p_input/binary/efn5_512_d90_he_v1_raw_10xTTA.csv')
    clf_preds_3 = pd.read_csv('../p_input/binary/efn0_1024_d75_add_v1_10xTTA.csv')
    clf_preds_4 = pd.read_csv('../p_input/binary/efn0_256_d80_he_cropped_v1.csv')

    clf_mean = np.stack((
                         clf_preds_1['raw_target'].values,
                         clf_preds_2['raw_target'].values,
                         clf_preds_3['raw_target'].values,
                         clf_preds_4['raw_target'].values
                         )).mean(axis=0)
    # test_all_files_set = set(clf_preds_2['ImageId'].values[clf_mean > 0.5])
    test_all_files_set = set(clf_preds_2['ImageId'].values[clf_mean > 0.365])
    # test_all_files_set = set(clf_preds_2['ImageId'].values[clf_mean > 0.255])

    # leak_df = pd.read_csv('../input/leaky_raddar/leak_probabilities.csv')
    # predict_for = set(leak_df[leak_df['pred'] > 0.7]['ImageId'].values)
    # test_all_files_set.update(predict_for)

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

    test_predictor = TTAPredictor(
        model=model,
        ds_base=test_dataset_base,
        batch_size=args.batch_size,
        workers=args.workers,
        base_transforms=[],
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
    # test_preds_resnet_v2 = test_preds
    # test_preds_resnet_v3 = test_preds
    # test_preds_pspnet_512_v1 = test_preds
    # test_preds_pspnet_512_v1_resized = resize_preds(test_preds_pspnet_512_v1)
    # test_preds_pspnet_512_v4 = test_preds
    # test_preds_pspnet_512_v4_resized = resize_preds(test_preds_pspnet_512_v4)

    # test_preds_resnet34_1024_v1 = test_preds
    # test_preds_resnet34_960_v1 = test_preds
    # tmp_preds = np.stack((test_preds_resnet_v2, test_preds_resnet_v3)).mean(axis=0)
    # test_preds = np.stack((tmp_preds, test_preds_pspnet_512_v1_resized)).mean(axis=0)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # tmp_preds2 = np.stack((resnet50_1024_v2_preds, resnet50_1024_v3_preds)).mean(axis=0)
    # eval_preds = np.stack((tmp_preds2, pspnet_512_preds_resized)).mean(axis=0)
    # eval_preds = np.stack((pspnet_512_v1_preds, pspnet_512_v4_preds)).mean(axis=0)

    # tmp_preds = np.stack((resnet50_1024_v2_preds, resnet50_1024_v3_preds, pspnet_512_v1_preds_resized, pspnet_512_v4_preds_resized)).mean(axis=0)
    # eval_preds = np.stack((tmp_preds, resnet34_1024_v1_preds, resnet34_960_v1_preds)).mean(axis=0)

    # eval_preds = np.stack((resnet34_960_v1_preds, resnet34_1024_v1_preds)).mean(axis=0)
    # eval_preds = np.stack((efn0_512_v2_preds_resized, resnet34_1024_v1_preds)).mean(axis=0)
    # eval_preds = np.stack((efn0_512_v2_preds_resized, resnet34_1024_v1_preds, resnet34_960_v1_preds)).mean(axis=0)

    tmp_preds = np.stack((resnet34_960_v1_preds,
                          resnet34_1024_v1_preds,
                          # smp_ur34_1024_v2_preds
                        )).mean(axis=0)
    eval_preds = np.stack((smp_ur34_1024_v1_preds, tmp_preds)).mean(axis=0)
    # eval_preds = np.stack((smp_ur34_1024_v1_preds, resnet34_1024_v1_preds, resnet34_960_v1_preds)).mean(axis=0)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dices = []
    thrs = np.arange(0., 1., 0.01)
    for i in tqdm(thrs):
        preds_m = (eval_preds > i).astype(np.int8)
        # dices.append(dice(eval_targets, preds_m))
        dices.append(correct_dice(eval_targets, preds_m))
        print('curr {} dice {}'.format(i, dices[-1]))
    dices = np.array(dices)
    eval_dice_score = dices.max()
    eval_best_thrs = thrs[np.argmax(dices)]

    print('Eval best dice score {} best treshold {}'.format(eval_dice_score, eval_best_thrs))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Resnet34 1024 v1
    # Eval best dice score 0.582241952419281 best treshold 0.32
    # Resnet34 960 v1
    # Eval best dice score 0.5847537517547607 best treshold 0.38
    # EFN0 512 v2 resized
    # Eval best dice score 0.5709494948387146 best treshold 0.43
    # SMP resnet34 1024 v1
    # Eval best dice score 0.5886539220809937 best treshold 0.3
    # SMP resnet34 1024 v2
    # Eval best dice score 0.5743390321731567 best treshold 0.21
    # Resnet34 1024 v1 + EFN0 512 v2 resized
    # Eval best dice score 0.5896256566047668 best treshold 0.36
    # Resnet34 1024 v1 + Resnet34 960 v1
    # Eval best dice score 0.5885118842124939 best treshold 0.34
    # Resnet34 1024 v1 + Resnet34 960 v1 + EFN0 512 v2 resized
    # Eval best dice score 0.5928488373756409 best treshold 0.39
    # Resnet34 1024 v1 + Resnet34 960 v1 + SMP resnet34 1024 v1
    # Eval best dice score 0.5967091917991638 best treshold 0.29
    # (Resnet34 1024 v1 + Resnet34 960 v1) + SMP resnet34 1024 v1
    # Eval best dice score 0.598153293132782 best treshold 0.32
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # dices = []
    # tresholds = []
    # thrs = np.arange(0., 1., 0.01)
    # for i in tqdm(range(len(eval_preds))):
    #     cur_dices = []
    #     for th in thrs:
    #         # preds_m = (eval_preds[i] > th).astype(np.int8)
    #         # cur_dices.append(correct_dice(eval_targets[i], preds_m))
    #         preds_m = (eval_preds[i:i+1] > th).astype(np.int8)
    #         cur_dices.append(correct_dice(eval_targets[i:i+1], preds_m))
    #     best_idx = np.argmax(cur_dices)
    #     dices.append(cur_dices[best_idx])
    #     tresholds.append(thrs[best_idx])
    # dices = np.array(dices)
    #
    # print('Eval best possible dice score {}'.format(np.mean(dices)))

    # Resnet34 1024 v1 + Resnet34 960 v1
    # Eval best possible dice score 0.6760188341140747
    # Resnet34 1024 v1 + Resnet34 960 v1 + EFN0 512 v2 resized
    # Eval best possible dice score 0.6856728196144104
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    old_ss = pd.read_csv('../input/sample_submission_old.csv')
    ss_uc = old_ss.groupby(['ImageId'])['EncodedPixels'].count()
    uc_sure = set(ss_uc[ss_uc > 1].index.values)
    idxes_of_sure = np.array([idx for idx, x in enumerate(test_all_files) if os.path.split(x)[1][:-4] in uc_sure])


    # tmp_preds = np.stack((test_preds_resnet_v2, test_preds_resnet_v3, test_preds_pspnet_512_v1_resized, test_preds_pspnet_512_v4_resized)).mean(axis=0)
    # test_preds = np.stack((tmp_preds, test_preds_resnet34_v1)).mean(axis=0)
    test_preds = np.stack((test_preds_resnet34_1024_v1, test_preds_resnet34_960_v1, test_preds_efn0_512_v2_resized)).mean(axis=0)
    # test_preds = np.stack((test_preds_resnet34_1024_v1, test_preds_resnet34_960_v1)).mean(axis=0)
    # test_preds = np.stack((tmp_preds, test_preds_resnet34_1024_v1, test_preds_resnet34_960_v1)).mean(axis=0)


    # args.output = '../submissions/efn0_1024_d75_he_v1_10xTTA_efn5_512_d90_v1_10xTTA_efn0_1024_d75_add_v1_10xTTA_0.6_leaky_cl_0.7_unet_resnet50_1024_v2_unet_resnet50_1024_v3_th0.6_0.45_rs42_seed42_f250_10xTTA.csv.gz'
    # args.output = '../submissions/prev_cl_unet_resnet50_1024_v2_unet_resnet50_1024_v3_pspnet_512_v1_th0.7_0.3_rs42_seed42_f250_sure_idxes.csv.gz'
    # args.output = '../submissions/efn0_1024_d75_he_v1_10xTTA_efn5_512_d90_v1_10xTTA_efn0_1024_d75_add_v1_10xTTA_0.6_leaky_cl_0.6_unet_resnet50_1024_v2_unet_resnet50_1024_v3_pspnet_512_v1_th0.7_0.3_rs42_seed42_f250_sure_idxes.csv.gz'
    # args.output = '../submissions/prev_cl_unet_resnet50_1024_v2_unet_resnet50_1024_v3_th0.7_0.3_rs42_seed42_f250.csv.gz'
    # args.output = '../submissions/clx4_0.5_leaky_cl_0.7_ur50_1024_v2_ur50_1024_v3_pspnet_512_v1_pspnet_512_v4_ur34_1024_v1_thadaptive_0.36_rs42_seed42_sure_idxes.csv.gz'
    args.output = '../submissions/clx4_0.5_leaky_cl_0.7_ur34_1024_v1_ur34_960_v1_uefn0_512_v2_adaptive_0.29_rs42_seed42_sure_idxes.csv.gz'
    # args.output = '../submissions/clx4_0.5_leaky_cl_0.7_ur34_1024_v1_ur34_960_v1_adaptive_0.34_rs42_seed42_sure_idxes.csv.gz'

    # eval_tresholds = [0.65]
    # th_mask_size = [250]
    #
    # res_preds = (test_preds > eval_tresholds[0]).astype(np.uint8)
    # res_preds = [filter_small_masks(x, th_mask_size[0]) for x in tqdm(res_preds)]
    #
    # print('empty count before {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    # for idx in range(len(res_preds)):
    #     if res_preds[idx].sum() > 0:
    #         res_preds[idx] = filter_small_masks((test_preds[idx] > 0.45).astype(np.uint8), 100)
    # print('empty count after {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))

    # second_cl_preds = (tmp_preds > 0.7).astype(np.uint8)
    # second_cl_preds = [filter_small_masks(x, 100) for x in tqdm(second_cl_preds)]
    # need_zeroing_for = (np.sum(second_cl_preds, axis=1).sum(axis=1) == 0)
    # pkl.dump(need_zeroing_for, open('../p_input/need_zeroing_for.pkl', 'wb'))
    need_zeroing_for = pkl.load(open('../p_input/need_zeroing_for.pkl', 'rb'))

    res_preds = (test_preds > 0.29).astype(np.uint8)
    # res_preds = np.array([filter_small_masks(x, 100) for x in tqdm(res_preds)])
    res_preds[need_zeroing_for] = 0

    print('res empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    print('res non empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) > 0).sum()))

    tmp_idxes = sorted(list(set(idxes_of_sure) & set(np.argwhere(np.sum(res_preds, axis=1).sum(axis=1) == 0).flatten())))

    for idx in tmp_idxes:
        res_preds[idx] = (test_preds[idx] > 0.1).astype(np.uint8)
        # res_preds[idx] = (test_preds[idx] > 0.12).astype(np.uint8)
        # res_preds[idx] = (test_preds[idx] > 0.15).astype(np.uint8)

    print('res empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    print('res non empty {}'.format((np.sum(res_preds, axis=1).sum(axis=1) > 0).sum()))

    # Optimal count for stage 1 test is 290 pneumothorax xrays

    # for th_idx, th in enumerate(eval_tresholds[1:]):
    #     print('empty count before {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))
    #     for idx in range(len(res_preds)):
    #         if res_preds[idx].sum() == 0:
    #             res_preds[idx] = filter_small_masks((test_preds[idx] > th).astype(np.uint8), th_mask_size[th_idx+1])
    #     print('empty count after {}'.format((np.sum(res_preds, axis=1).sum(axis=1) == 0).sum()))

    res_rle = [mask2rle(x.T * 255, x.shape[0], x.shape[1]) if x.sum() > 0 else '-1' for x in tqdm(res_preds)]
    # res_df['EncodedPixels'] = '-1'
    # res_df.loc[res_df['ImageId'].isin(test_all_files_set), 'EncodedPixels'] = res_rle
    # res_df['EncodedPixels'] = res_df['EncodedPixels'].replace('', '-1')
    # res_df['ImageId'] = res_df['ImageId'].apply((lambda x: os.path.split(x)[1][:-4]))
    #
    # res_df = res_df[['ImageId', 'EncodedPixels']]
    # res_df.to_csv(args.output, index=False, compression='gzip')

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

    # tmp_df = pd.read_csv('../submissions/prev_cl_unet_resnet50_1024_v2_unet_resnet50_1024_v3_pspnet_512_v1_th0.7_0.3_rs42_seed42_f250_smart_second_cl_empty_sure_idxes.csv.gz')
    # '../submissions/efn0_1024_d75_he_v1_10xTTA_efn5_512_d90_v1_10xTTA_efn0_1024_d75_add_v1_10xTTA_0.6_leaky_cl_0.7_unet_resnet50_1024_v2_unet_resnet50_1024_v3_th0.6_0.45_rs42_seed42_f250_10xTTA.csv'

    # res_df.loc[res_df['ImageId'].isin(test_all_files_set), 'tt'] = test_all_files
    # is_correct = (res_df['ImageId'] == res_df['tt']).sum() == len(test_all_files)
    # res_df.drop('tt', 1, inplace=True)
    # print('correct {}'.format(is_correct))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # t1 = pd.read_csv('../submissions/clx4_0.5_leaky_cl_0.7_ur50_1024_v2_ur50_1024_v3_pspnet_512_v1_pspnet_512_v4_ur34_1024_v1_thadaptive_0.36_rs42_seed42_sure_idxes.csv.gz')
    # t2 = pd.read_csv('../submissions/clx4_0.5_leaky_cl_0.7_ur50_1024_v2_ur50_1024_v3_pspnet_512_v1_pspnet_512_v4_ur34_1024_v1_thadaptive_0.36_rs42_seed42_f100_sure_idxes.csv.gz')

    # t1 = pd.read_csv('clx4_0.5_leaky_cl_0.7_ur50_1024_v2_ur50_1024_v3_pspnet_512_v1_pspnet_512_v4_ur34_1024_v1_thadaptive_0.36_rs42_seed42_f100_sure_idxes.csv.gz')
    # t2 = pd.read_csv('clx4_0.5_leaky_cl_0.7_ur50_1024_v2_ur50_1024_v3_pspnet_512_v1_pspnet_512_v4_ur34_1024_v1_thadaptive_0.36_rs42_seed42_sure_idxes.csv.gz')

    # t1 = t1.merge(t2, how='left', on='ImageId')
    # print(t1[t1.EncodedPixels_x == t1.EncodedPixels_y].shape, t1[t1.EncodedPixels_x == t1.EncodedPixels_y].shape[0] / t1.shape[0])


if __name__ == '__main__':
    main()
