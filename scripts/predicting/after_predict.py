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
import pickle as pkl
import lightgbm as lgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, confusion_matrix

from scripts.utils.utils import getListOfFiles
from scripts.utils.metrics import dice, correct_dice
from scripts.utils.mask_functions import mask2rle

from scripts.models.unet_resnet34 import UnetResnet34
import segmentation_models_pytorch as smp
from scripts.dataset.he_dataset import HeSegmentationDataset
from scripts.dataset.raw_dataset import RawSegmentationDataset
from scripts.dataset.raw_pos_neg_dataset import RawPosNegSegmentationDataset
from scripts.dataset.he_pos_neg_dataset import HePosNegSegmentationDataset

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
    parser = argparse.ArgumentParser(description='Predict best of segmentation and classifiers and make some assumes on that result with lgb')

    parser.add_argument('-i', '--input', default='../input/dicom-images-train', help='input eval data directory')
    parser.add_argument('-it', '--input_test', default='../input/dicom-images-test', help='input test data directory')
    parser.add_argument('-id', '--input_df', default='../input/train-rle.csv', help='input train df file')
    parser.add_argument('-ss', '--sample_submission', default='../input/sample_submission.csv', help='sample submission file')

    parser.add_argument('-s', '--seed', default=42, help='seed')
    parser.add_argument('-tfr', '--test_fold_ratio', default=0.2, help='test fold ratio')

    # parser.add_argument('-mp', '--model_path', default='../models/unet_resnet34_1024_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/unet_resnet34_960_v1/', help='path to models directory')
    # parser.add_argument('-mp', '--model_path', default='../models/smp_ur34_1024_v1/', help='path to models directory')
    parser.add_argument('-bs', '--batch_size', default=8, help='size of batches', type=int)
    parser.add_argument('-w', '--workers', default=6, help='data loader wokers count', type=int)
    parser.add_argument('-is', '--image_size', default=1024, help='image size', type=int)

    args = parser.parse_args()

    df = pd.read_csv(args.input_df)
    ids_with_mask = set(df[df[' EncodedPixels'].str.strip() != '-1']['ImageId'].values)
    ids_without_mask = set(df[df[' EncodedPixels'].str.strip() == '-1']['ImageId'].values)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    norm = Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.225, 0.225, 0.225],
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    all_files = getListOfFiles(args.input)
    all_pos_files = [x for x in all_files if os.path.split(x)[1][:-4] in ids_with_mask]
    all_pos_files = np.array(all_pos_files)
    all_neg_files = [x for x in all_files if os.path.split(x)[1][:-4] in ids_without_mask]
    all_neg_files = np.array(all_neg_files)

    _, valid_pos_files = train_test_split(all_pos_files,
                                          test_size=args.test_fold_ratio,
                                          random_state=args.seed)

    _, valid_neg_files = train_test_split(all_neg_files,
                                          test_size=args.test_fold_ratio,
                                          random_state=args.seed)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ur34_1024_v1_checkpoint_name = '../models/unet_resnet34_1024_v1/best_dice_model.t7'
    ur34_1024_v1_base_model = UnetResnet34(in_channels=3, num_classes=1, num_filters=32, pretrained=False, is_deconv=True)
    ur34_1024_v1_val_dataset_base = functools.partial(
        HePosNegSegmentationDataset,
        dcm_pos_files=valid_pos_files,
        dcm_neg_files=valid_neg_files,
        masks_file=args.input_df,
        neg_ratio=None,
    )
    ur34_1024_v1_base_transformations = []

    ur34_960_v1_checkpoint_name = '../models/unet_resnet34_960_v1/best_dice_model.t7'
    ur34_960_v1_base_model = ur34_1024_v1_base_model
    ur34_960_v1_val_dataset_base = ur34_1024_v1_val_dataset_base
    ur34_960_v1_base_transformations = ur34_1024_v1_base_transformations

    smp_ur34_1024_v1_checkpoint_name = '../models/smp_ur34_1024_v1/best_dice_model.t7'
    smp_ur34_1024_v1_base_model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    smp_ur34_1024_v1_val_dataset_base = functools.partial(
        RawPosNegSegmentationDataset,
        dcm_pos_files=valid_pos_files,
        dcm_neg_files=valid_neg_files,
        masks_file=args.input_df,
        neg_ratio=None,
    )
    smp_ur34_1024_v1_base_transformations = [norm]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ur34_1024_v1_val_res, val_targets = predict_seg(
        args,
        ur34_1024_v1_checkpoint_name,
        ur34_1024_v1_base_model,
        ur34_1024_v1_val_dataset_base,
        ur34_1024_v1_base_transformations
    )
    pkl.dump(ur34_1024_v1_val_res,
             open('../p_input/ur34_1024_v1_val_res.pkl', 'wb'),
             protocol=4)
    pkl.dump(val_targets, open('../p_input/rs42_tfp0.2_pos_neg_val_targets.pkl', 'wb'))

    ur34_960_v1_val_res, _ = predict_seg(
        args,
        ur34_960_v1_checkpoint_name,
        ur34_960_v1_base_model,
        ur34_960_v1_val_dataset_base,
        ur34_960_v1_base_transformations
    )
    pkl.dump(ur34_960_v1_val_res,
             open('../p_input/ur34_960_v1_val_res.pkl', 'wb'),
             protocol=4)

    smp_ur34_1024_v1_val_res, _ = predict_seg(
        args,
        smp_ur34_1024_v1_checkpoint_name,
        smp_ur34_1024_v1_base_model,
        smp_ur34_1024_v1_val_dataset_base,
        smp_ur34_1024_v1_base_transformations
    )
    pkl.dump(smp_ur34_1024_v1_val_res,
             open('../p_input/smp_ur34_1024_v1_val_res.pkl', 'wb'),
             protocol=4)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ur34_1024_v1_val_res = pkl.load(open('../p_input/ur34_1024_v1_val_res.pkl', 'rb'))
    ur34_960_v1_val_res = pkl.load(open('../p_input/ur34_960_v1_val_res.pkl', 'rb'))
    smp_ur34_1024_v1_val_res = pkl.load(open('../p_input/smp_ur34_1024_v1_val_res.pkl', 'rb'))
    val_targets = pkl.load(open('../p_input/rs42_tfp0.2_pos_neg_val_targets.pkl', 'rb'))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    eval_files = np.concatenate((valid_pos_files, valid_neg_files))
    valid_df = pd.DataFrame(data={'images': eval_files})
    valid_df['bin_target'] = 0
    valid_df.loc[valid_df.index < len(valid_pos_files), 'bin_target'] = 1
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    clf_preds_files = [
        '../p_input/binary/valid_efficientnet0_1024_d75_he_v1_raw_10xTTA.csv',
        '../p_input/binary/valid_efn5_512_d90_he_v1_raw_10xTTA.csv',
        '../p_input/binary/valid_efn0_1024_d75_add_v1_10xTTA.csv',
        '../p_input/binary/valid_efn0_256_d80_he_cropped_v1.csv',
        '../p_input/binary/valid_efn0_512_aug_dynamic_v1.csv'
    ]

    images_set = set(df['ImageId'].values)
    eval_all_files = [x for x in all_files if os.path.split(x)[1][:-4] in images_set]
    eval_all_files = np.array(eval_all_files)
    for idx, file in enumerate(clf_preds_files):
        tmp_df = pd.read_csv(file)
        eval_all_files_dict = dict(zip(eval_all_files, tmp_df['raw_target'].values))
        eval_files_res = np.array([eval_all_files_dict[x] for x in eval_files])
        valid_df['cl_{}'.format(idx)] = eval_files_res
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    tmp_preds = np.stack((ur34_1024_v1_val_res, ur34_960_v1_val_res)).mean(axis=0)
    mean_val_res = np.stack((smp_ur34_1024_v1_val_res, tmp_preds)).mean(axis=0)
    del tmp_preds
    gc.collect()

    seg_preds = [
        ur34_1024_v1_val_res,
        ur34_960_v1_val_res,
        smp_ur34_1024_v1_val_res,
        mean_val_res
    ]

    for th in tqdm(np.arange(0., 1., 0.05)):
        for idx, seg_pred in enumerate(seg_preds):
            pred_m = (seg_pred > th)
            valid_df['seg_{}_th_{}_count'.format(idx, th)] = pred_m.sum(axis=1).sum(axis=1)
    del pred_m
    gc.collect()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    train_cols = ['cl_{}'.format(i) for i in range(len(clf_preds_files))]
    target_col = 'bin_target'
    train_cols.extend(['seg_{}_th_{}_count'.format(idx, th) for idx in range(len(seg_preds)) for th in np.arange(0., 1., 0.05)])

    param = {
        'bagging_freq': 1,
        'bagging_fraction': 0.38,
        # 'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.33,
        # 'learning_rate': 0.1,
        'learning_rate': 0.01,
        # 'max_depth': 2,
        'max_depth': 6,
        'num_leaves': 63,
        # 'metric':'auc',
        # 'metric':'binary_error',
        # 'min_data_in_leaf': 80,
        # 'min_sum_hessian_in_leaf': 10.0,
        'num_threads': 8,
        # 'tree_learner': 'serial',
        'objective': 'binary',
        'verbosity': -1
    }

    fold_count = 10
    folds = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(valid_df))
    cl_clfs = []

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(valid_df[train_cols].values, valid_df[target_col].values)):
        print("Fold : {}".format(fold_ + 1))
        trn_data = lgb.Dataset(valid_df.iloc[trn_idx][train_cols], label=valid_df.iloc[trn_idx][target_col])
        val_data = lgb.Dataset(valid_df.iloc[val_idx][train_cols], label=valid_df.iloc[val_idx][target_col])
        clf = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
        oof[val_idx] = clf.predict(valid_df.iloc[val_idx][train_cols], num_iteration=clf.best_iteration)
        cl_clfs.append(clf)

    max_acc = np.max([accuracy_score(valid_df[target_col].values, (oof > th)) for th in np.arange(0., 1., 0.01)])
    max_acc_th = np.argmax([accuracy_score(valid_df[target_col].values, (oof > th)) for th in np.arange(0., 1., 0.01)])
    max_acc_th = max_acc_th / 100.
    print('CV log loss score: {}'.format(log_loss(valid_df[target_col].values, oof)))
    print('CV accuracy score: {} th {}'.format(max_acc, max_acc_th))
    print('CV roc auc score: {}'.format(roc_auc_score(valid_df[target_col].values, oof)))
    cl_oof = oof.copy()

    for idx, clf in enumerate(cl_clfs):
        clf.save_model('../p_input/classifier_lgb_{}'.format(idx))
    # orig CLx5
    # CV log loss score: 0.20778928355121096
    # CV accuracy score: 0.9133895131086143
    # CV roc auc score: 0.9589424926597145
    # th x20 over segx3+1mean
    # CV log loss score: 0.1886846117676869
    # CV accuracy score: 0.9293071161048689 th 0.51
    # CV roc auc score: 0.9666573858459045
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    dices = []
    thrs = np.arange(0., 1., 0.01)
    for idx, th in tqdm(enumerate(thrs), total=len(thrs)):
        preds_m = (mean_val_res > th).astype(np.int8)
        preds_m[cl_oof <= max_acc_th] = 0
        dices.append(correct_dice(val_targets, preds_m))
        print('curr {} dice {}'.format(th, dices[-1]))
    dices = np.array(dices)
    eval_dice_score = dices.max()
    eval_best_thrs = thrs[np.argmax(dices)]

    print('Eval best dice score {} best treshold {}'.format(eval_dice_score, eval_best_thrs))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Eval best dice score 0.8655028939247131 best treshold 0.46
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # dices = []
    # tresholds = []
    # thrs = np.arange(0., 1., 0.01)
    # for i in tqdm(range(len(valid_pos_files))):
    #     cur_dices = []
    #     for th in thrs:
    #         preds_m = (mean_val_res[i:i+1] > th).astype(np.int8)
    #         cur_dices.append(correct_dice(val_targets[i:i+1], preds_m))
    #     best_idx = np.argmax(cur_dices)
    #     dices.append(cur_dices[best_idx])
    #     tresholds.append(thrs[best_idx])
    # dices = np.array(dices)
    #
    # print('Eval best possible dice score {}'.format(np.mean(dices)))
    # # Eval best possible dice score 0.6842488646507263
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # from sklearn.metrics import mean_squared_error
    #
    # pos_sub_df = valid_df[:len(valid_pos_files)].copy()
    # pos_sub_df['th'] = tresholds
    # train_cols = ['cl_{}'.format(i) for i in range(len(clf_preds_files))]
    # train_cols.extend(['seg_{}_th_{}_count'.format(idx, th) for idx in range(len(seg_preds)) for th in np.arange(0., 1., 0.05)])
    # target_col = 'th'
    #
    # param = {
    #     'bagging_freq': 1,
    #     'bagging_fraction': 0.38,
    #     # 'boost_from_average':'false',
    #     'boost': 'gbdt',
    #     'feature_fraction': 0.33,
    #     # 'learning_rate': 0.1,
    #     'learning_rate': 0.01,
    #     # 'max_depth': 2,
    #     'max_depth': 6,
    #     'num_leaves': 63,
    #     'metric':'mse',
    #     # 'metric':'binary_error',
    #     # 'min_data_in_leaf': 80,
    #     # 'min_sum_hessian_in_leaf': 10.0,
    #     'num_threads': 8,
    #     # 'tree_learner': 'serial',
    #     # 'objective': 'binary',
    #     'objective': 'regression',
    #     'verbosity': -1
    # }
    #
    # fold_count = 5
    # folds = KFold(n_splits=fold_count, shuffle=True, random_state=args.seed)
    # oof = np.zeros(len(pos_sub_df))
    # clfs = []
    #
    # for fold_, (trn_idx, val_idx) in enumerate(folds.split(np.arange(len(pos_sub_df)))):
    #     print("Fold : {}".format(fold_ + 1))
    #     trn_data = lgb.Dataset(pos_sub_df.iloc[trn_idx][train_cols], label=pos_sub_df.iloc[trn_idx][target_col])
    #     val_data = lgb.Dataset(pos_sub_df.iloc[val_idx][train_cols], label=pos_sub_df.iloc[val_idx][target_col])
    #     clf = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    #     oof[val_idx] = clf.predict(pos_sub_df.iloc[val_idx][train_cols], num_iteration=clf.best_iteration)
    #     clfs.append(clf)
    #
    # # max_acc = np.max([accuracy_score(pos_sub_df[target_col].values, (oof > th)) for th in np.arange(0., 1., 0.01)])
    # # max_acc_th = np.argmax([accuracy_score(valid_df[target_col].values, (oof > th)) for th in np.arange(0., 1., 0.01)])
    # # max_acc_th = max_acc_th / 100.
    # print('CV mse score: {}'.format(mean_squared_error(pos_sub_df[target_col].values, oof)))
    # # CV mse score: 0.08751432656237994
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # neg_preds = np.zeros((len(valid_neg_files)))
    # for clf in clfs:
    #     neg_preds += clf.predict(valid_df[len(valid_pos_files):][train_cols], num_iteration=clf.best_iteration)
    # neg_preds = neg_preds / len(clfs)
    # th_preds = np.concatenate((oof, neg_preds))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # preds = []
    # for idx, pr in enumerate(mean_val_res):
    #     preds.append((pr > th_preds[idx] if cl_oof[idx] > max_acc_th else np.zeros_like(pr, dtype=bool)).astype(np.uint8))
    # preds = np.array(preds)
    # final_dice = correct_dice(val_targets, preds)
    # print("final dice {}".format(final_dice))
    # # final dice 0.8659218549728394
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # clf_preds_1 = pd.read_csv('../p_input/binary/valid_efficientnet0_1024_d75_he_v1_raw_10xTTA.csv')
    # clf_preds_2 = pd.read_csv('../p_input/binary/valid_efn5_512_d90_he_v1_raw_10xTTA.csv')
    # clf_preds_3 = pd.read_csv('../p_input/binary/valid_efn0_1024_d75_add_v1_10xTTA.csv')
    # clf_preds_4 = pd.read_csv('../p_input/binary/valid_efn0_256_d80_he_cropped_v1.csv')
    # clf_preds_5 = pd.read_csv('../p_input/binary/valid_efn0_512_aug_dynamic_v1.csv')
    #
    # clf_tmp_1 = np.stack((
    #     clf_preds_1['raw_target'].values,
    #     clf_preds_2['raw_target'].values,
    #     clf_preds_3['raw_target'].values
    # )).mean(axis=0)
    #
    # clf_tmp_2 = np.stack((
    #     clf_preds_4['raw_target'].values,
    #     clf_preds_5['raw_target'].values,
    # )).mean(axis=0)
    #
    # clf_mean = np.stack((
    #     clf_tmp_1,
    #     clf_tmp_2,
    # )).mean(axis=0)
    #
    # eval_all_files_dict = dict(zip(eval_all_files, clf_mean))
    # eval_files_res = np.array([eval_all_files_dict[x] for x in eval_files])
    #
    # cur_eval_preds = mean_val_res.copy()
    # cur_eval_preds[eval_files_res <= 0.356] = 0
    # preds_m = (cur_eval_preds > 0.65).astype(np.int8)
    # preds_m = np.array([filter_small_masks(x, 1500) for x in tqdm(preds_m)])
    # need_zeroing_for = (np.sum(preds_m, axis=1).sum(axis=1) == 0)
    #
    # dices = []
    # thrs = np.arange(0., 1., 0.01)
    # for i in tqdm(thrs):
    #     preds_m = (cur_eval_preds > i).astype(np.int8)
    #     preds_m[need_zeroing_for] = 0
    #     dices.append(correct_dice(val_targets, preds_m))
    #     print('curr {} dice {}'.format(i, dices[-1]))
    # dices = np.array(dices)
    # eval_dice_score = dices.max()
    # eval_best_thrs = thrs[np.argmax(dices)]
    #
    # print('Eval best dice score {} best treshold {}'.format(eval_dice_score, eval_best_thrs))

    # mean_val_res cl0.365 base0.8 f250
    # Eval best dice score 0.8630527257919312 best treshold 0.47
    # mean_val_res cl0.356 base0.65 f1500
    # Eval best dice score 0.8626716136932373 best treshold 0.44

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    del seg_preds
    del ur34_1024_v1_val_res
    del ur34_960_v1_val_res
    del smp_ur34_1024_v1_val_res
    del val_targets
    del mean_val_res
    gc.collect()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_all_files = getListOfFiles(args.input_test)
    ss = pd.read_csv(args.sample_submission)
    ss_images_set = set(ss['ImageId'].values)
    test_all_files = [x for x in test_all_files if (os.path.split(x)[1][:-4]) in ss_images_set]
    test_all_files = np.array(test_all_files)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ur34_1024_v1_test_dataset_base = functools.partial(
        HeSegmentationDataset,
        dcm_files=test_all_files,
    )
    ur34_1024_v1_test_res, _ = predict_seg(
        args,
        ur34_1024_v1_checkpoint_name,
        ur34_1024_v1_base_model,
        ur34_1024_v1_test_dataset_base,
        ur34_1024_v1_base_transformations
    )
    pkl.dump(ur34_1024_v1_test_res,
             open('../p_input/ur34_1024_v1_test_res.pkl', 'wb'),
             protocol=4)

    ur34_960_v1_test_dataset_base = ur34_1024_v1_test_dataset_base
    ur34_960_v1_test_res, _ = predict_seg(
        args,
        ur34_960_v1_checkpoint_name,
        ur34_960_v1_base_model,
        ur34_960_v1_test_dataset_base,
        ur34_960_v1_base_transformations
    )
    pkl.dump(ur34_960_v1_test_res,
             open('../p_input/ur34_960_v1_test_res.pkl', 'wb'),
             protocol=4)

    smp_ur34_1024_v1_test_dataset_base = functools.partial(
        RawSegmentationDataset,
        dcm_files=test_all_files,
    )

    smp_ur34_1024_v1_test_res, _ = predict_seg(
        args,
        smp_ur34_1024_v1_checkpoint_name,
        smp_ur34_1024_v1_base_model,
        smp_ur34_1024_v1_test_dataset_base,
        smp_ur34_1024_v1_base_transformations
    )
    pkl.dump(smp_ur34_1024_v1_test_res,
             open('../p_input/smp_ur34_1024_v1_test_res.pkl', 'wb'),
             protocol=4)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ur34_1024_v1_test_res = pkl.load(open('../p_input/ur34_1024_v1_test_res.pkl', 'rb'))
    ur34_960_v1_test_res = pkl.load(open('../p_input/ur34_960_v1_test_res.pkl', 'rb'))
    smp_ur34_1024_v1_test_res = pkl.load(open('../p_input/smp_ur34_1024_v1_test_res.pkl', 'rb'))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_df = pd.DataFrame(data={'images': test_all_files})
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    clf_preds_files = [
        '../p_input/binary/efficientnet0_1024_d75_he_v1_raw_10xTTA.csv',
        '../p_input/binary/efn5_512_d90_he_v1_raw_10xTTA.csv',
        '../p_input/binary/efn0_1024_d75_add_v1_10xTTA.csv',
        '../p_input/binary/efn0_256_d80_he_cropped_v1.csv',
        '../p_input/binary/efn0_512_aug_dynamic_v1.csv'
    ]

    for idx, file in enumerate(clf_preds_files):
        tmp_df = pd.read_csv(file)
        test_df['cl_{}'.format(idx)] = tmp_df['raw_target'].values
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    tmp_preds = np.stack((ur34_1024_v1_test_res, ur34_960_v1_test_res)).mean(axis=0)
    mean_test_res = np.stack((smp_ur34_1024_v1_test_res, tmp_preds)).mean(axis=0)
    del tmp_preds
    gc.collect()

    seg_preds = [
        ur34_1024_v1_test_res,
        ur34_960_v1_test_res,
        smp_ur34_1024_v1_test_res,
        mean_test_res
    ]

    for th in tqdm(np.arange(0., 1., 0.05)):
        for idx, seg_pred in enumerate(seg_preds):
            pred_m = (seg_pred > th)
            test_df['seg_{}_th_{}_count'.format(idx, th)] = pred_m.sum(axis=1).sum(axis=1)
    del pred_m
    gc.collect()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_cols = ['cl_{}'.format(i) for i in range(len(clf_preds_files))]
    train_cols.extend(['seg_{}_th_{}_count'.format(idx, th) for idx in range(len(seg_preds)) for th in np.arange(0., 1., 0.05)])

    fold_count = 10
    clfs = [lgb.Booster(model_file='../p_input/classifier_lgb_{}'.format(idx)) for idx in range(fold_count)]
    test_cl_preds = np.zeros((len(test_df)))
    for clf in clfs:
        test_cl_preds += clf.predict(test_df[train_cols], num_iteration=clf.best_iteration)
    test_cl_preds = test_cl_preds / len(clfs)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    args.output = '../submissions/lgb_cl_0.51_ur34_1024_v1_ur34_960_v1_smp_ur34_1024_v1_0.46.csv.gz'
    res_preds = (mean_test_res > 0.46).astype(np.uint8)
    res_preds[test_cl_preds <= 0.51] = 0

    res_rle = [mask2rle(x.T * 255, x.shape[0], x.shape[1]) if x.sum() > 0 else '-1' for x in tqdm(res_preds)]

    res_df = pd.DataFrame(data={
        'ImageId': [os.path.split(x)[1][:-4] for x in test_all_files],
        'EncodedPixels': res_rle
    })
    res_df.to_csv(args.output, index=False, compression='gzip')

    os.system('kaggle competitions submit siim-acr-pneumothorax-segmentation -f {} -m "{}"'.format(args.output, "{}_fold_seed_{}".format(args.test_fold_ratio, args.seed)))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    main()
