import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import functools

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from scripts.utils.utils import getListOfFiles

# from scripts.binary.models.binary_classifier_efficientnet import BinaryClassifyEfficientNet
# from scripts.binary.dataset.binary_raw_he_dataset import BinaryRawHEDataset
from scripts.binary.models.binary_classifier_efficientnet_add import BinaryClassifyEfficientNetAdd
from scripts.binary.dataset.binary_raw_add_dataset import BinaryRawAddDataset

from scripts.predicting.tta_add_predictor import TTAAddPredictor

from albumentations import (
    Resize,
    Normalize,
    HorizontalFlip,
    Compose,
    Rotate
)

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description='Evaluate binary efficientnet')

    parser.add_argument('-i', '--input', default='../input/dicom-images-train', help='input eval data directory')
    parser.add_argument('-it', '--input_test', default='../input/dicom-images-test', help='input test data directory')
    parser.add_argument('-id', '--input_df', default='../input/train-rle.csv', help='input train df file')
    parser.add_argument('-ss', '--sample_submission', default='../input/sample_submission.csv', help='sample submission file')
    parser.add_argument('-o', '--output', default='../p_input/binary/efn0_1024_d75_add_v1_10xTTA.csv', help='output file')
    parser.add_argument('-vo', '--validation_output', default='../p_input/binary/valid_efn0_1024_d75_add_v1_10xTTA.csv', help='output file')
    parser.add_argument('-kf', '--kfolds', default=6, help='kfold splitting')
    parser.add_argument('-s', '--seed', default=42, help='seed')

    parser.add_argument('-mp', '--model_path', default='../models/binary/efficientnet0_is1024_dropout75_add', help='path to models directory')
    parser.add_argument('-bs', '--batch_size', default=8, help='size of batches', type=int)
    parser.add_argument('-w', '--workers', default=8, help='data loader wokers count', type=int)

    parser.add_argument('-is', '--image_size', default=1024, help='image size', type=int)

    args = parser.parse_args()

    df = pd.read_csv(args.input_df)
    df['target'] = (df[' EncodedPixels'].str.strip() != '-1').astype(np.uint8)
    targets_dict = dict(zip(df['ImageId'].values, df['target'].values))
    images_set = set(df['ImageId'].values)

    all_files = getListOfFiles(args.input)
    all_files = [x for x in all_files if (os.path.split(x)[1][:-4]) in images_set]
    all_targets = [targets_dict[os.path.split(x)[1][:-4]] for x in all_files]

    all_files = np.array(all_files)
    all_targets = np.array(all_targets)


    test_all_files = getListOfFiles(args.input_test)
    ss = pd.read_csv(args.sample_submission)
    ss_images_set = set(ss['ImageId'].values)

    test_all_files = [x for x in test_all_files if (os.path.split(x)[1][:-4]) in ss_images_set]
    test_all_files = np.array(test_all_files)


    current_size = int(args.image_size)
    norm = Normalize(
        mean=[0.5],
        std=[0.225],
    )
    base_transforms = [Resize(current_size, current_size), norm]

    full_eval_targets = np.zeros((len(all_files)), dtype=np.uint8)
    full_eval_preds = np.zeros((len(all_files)), dtype=np.float32)
    full_test_preds = np.zeros((len(test_all_files)), dtype=np.float32)

    folds = KFold(n_splits=args.kfolds, shuffle=False, random_state=args.seed)
    for fold_idx, (_, eval_idx) in enumerate(folds.split(np.arange(len(all_files)))):
        eval_images = all_files[eval_idx]
        eval_targets = all_targets[eval_idx]

        # # # # # # # # # Model loading # # # # # # # # #
        current_model_dir = os.path.join(args.model_path, 'kfold_{}'.format(fold_idx))
        checkpoint = torch.load(os.path.join(current_model_dir, 'best_f1_model.t7'))
        # checkpoint = torch.load(os.path.join(current_model_dir, 'best_loss_model.t7'))
        model = BinaryClassifyEfficientNetAdd(
            N=0,
            add_size=3,
            pretrained=False,
            dropout=0.75
        )
        model = model.cuda()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        eval_dataset_base = functools.partial(
            BinaryRawAddDataset,
            files=eval_images,
            targets=eval_targets,
        )

        eval_predictor = TTAAddPredictor(
            model=model,
            ds_base=eval_dataset_base,
            is_validation=True,
            batch_size=args.batch_size,
            workers=args.workers,
            base_transforms=base_transforms
        )

        eval_predictor.put([HorizontalFlip(always_apply=True)], None)
        eval_predictor.put([Rotate(limit=(-15,-15), always_apply=True)], None)
        eval_predictor.put([Rotate(limit=(15,15), always_apply=True)], None)

        eval_predictor.put(
            [Rotate(limit=(-15,-15), always_apply=True),
             HorizontalFlip(always_apply=True)],
            None
        )

        eval_predictor.put(
            [Rotate(limit=(15,15), always_apply=True),
             HorizontalFlip(always_apply=True)],
            None
        )

        eval_predictor.base_transforms = [norm]

        eval_predictor.put([Resize(int(current_size * 1.25), int(current_size * 1.25))], None)
        eval_predictor.put([Resize(int(current_size * 0.75), int(current_size * 0.75))], None)

        eval_predictor.put(
            [HorizontalFlip(always_apply=True),
             Resize(int(current_size * 1.25), int(current_size * 1.25))],
            None
        )

        eval_predictor.put(
            [HorizontalFlip(always_apply=True),
             Resize(int(current_size * 0.75), int(current_size * 0.75))],
            None
        )

        eval_preds = []
        eval_targets = []
        for pred, targets in tqdm(eval_predictor):
            eval_preds.extend(pred)
            eval_targets.extend(targets)
        eval_preds = np.array(eval_preds)
        eval_targets = np.array(eval_targets)

        full_eval_targets[eval_idx] = eval_targets
        full_eval_preds[eval_idx] = eval_preds

        test_dataset_base = functools.partial(
            BinaryRawAddDataset,
            files=test_all_files,
            targets=np.zeros((len(test_all_files)), dtype=np.uint8),
        )

        test_predictor = TTAAddPredictor(
            model=model,
            ds_base=test_dataset_base,
            is_validation=True,
            batch_size=args.batch_size,
            workers=args.workers,
            base_transforms=base_transforms
        )

        test_predictor.put([HorizontalFlip(always_apply=True)], None)
        test_predictor.put([Rotate(limit=(-15,-15), always_apply=True)], None)
        test_predictor.put([Rotate(limit=(15,15), always_apply=True)], None)

        test_predictor.put(
            [Rotate(limit=(-15,-15), always_apply=True),
             HorizontalFlip(always_apply=True)],
            None
        )

        test_predictor.put(
            [Rotate(limit=(15,15), always_apply=True),
             HorizontalFlip(always_apply=True)],
            None
        )

        test_predictor.base_transforms = [norm]

        test_predictor.put([Resize(int(current_size * 1.25), int(current_size * 1.25))], None)
        test_predictor.put([Resize(int(current_size * 0.75), int(current_size * 0.75))], None)

        test_predictor.put(
            [HorizontalFlip(always_apply=True),
             Resize(int(current_size * 1.25), int(current_size * 1.25))],
            None
        )

        test_predictor.put(
            [HorizontalFlip(always_apply=True),
             Resize(int(current_size * 0.75), int(current_size * 0.75))],
            None
        )

        test_preds = []
        for pred, _ in tqdm(test_predictor):
            test_preds.extend(pred)
        test_preds = np.array(test_preds)

        full_test_preds += test_preds

    full_test_preds = full_test_preds / args.kfolds

    valid_df = pd.DataFrame(
        {'ImageId': [os.path.split(x)[1][:-4] for x in all_files],
         'raw_target': full_eval_preds}
    )
    valid_df.to_csv(args.validation_output, index=False)

    result_df = pd.DataFrame(
        {'ImageId': [os.path.split(x)[1][:-4] for x in test_all_files],
        'raw_target': full_test_preds}
    )
    result_df.to_csv(args.output, index=False)

    f1_scores = []
    accuracies = []
    thrs = np.arange(0, 1.01, 0.01)
    for i in tqdm(thrs):
        preds_m = (full_eval_preds > i).astype(np.int8)
        f1_scores.append(f1_score(full_eval_targets, preds_m))
        accuracies.append(accuracy_score(full_eval_targets, preds_m))
    f1_scores = np.array(f1_scores)
    eval_f1_score = f1_scores.max()
    eval_best_thrs = thrs[np.argmax(f1_scores)]

    accuracies = np.array(accuracies)
    eval_acc_score = accuracies.max()
    eval_best_acc_treshold = thrs[np.argmax(accuracies)]

    print('Eval best f1 score {} best treshold {}'.format(eval_f1_score, eval_best_thrs))
    print('Eval best acc score {} best treshold {}'.format(eval_acc_score, eval_best_acc_treshold))

    # 1xTTA
    # Eval best f1 score 0.7759452936444087 best treshold 0.37
    # Eval best acc score 0.8986416861826698 best treshold 0.48
    # 10xTTA
    # Eval best f1 score 0.7839156744095257 best treshold 0.32
    # Eval best acc score 0.8999531615925058 best treshold 0.42

    # Comb
    # Eval best f1 score 0.7956061200470773 best treshold 0.33
    # Eval best acc score 0.905480093676815 best treshold 0.46
    # Comb x3
    # Eval best f1 score 0.7996860898567786 best treshold 0.32
    # Eval best acc score 0.9076346604215456 best treshold 0.44

    tmp1 = pd.read_csv('../p_input/binary/valid_efn0_1024_d75_add_v1_10xTTA.csv')
    tmp2 = pd.read_csv('../p_input/binary/valid_efn5_512_d90_he_v1_raw_10xTTA.csv')
    tmp3 = pd.read_csv('../p_input/binary/valid_efficientnet0_1024_d75_he_v1_raw_10xTTA.csv')

    full_eval_preds = np.stack((tmp1.raw_target.values,
                                tmp2.raw_target.values,
                                tmp3.raw_target.values,)).mean(axis=0)




if __name__ == '__main__':
    main()
