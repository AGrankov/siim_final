import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import functools

from scripts.utils.utils import getListOfFiles

from scripts.binary.models.binary_classifier_efficientnet import BinaryClassifyEfficientNet
from scripts.binary.dataset.binary_raw_he_dataset import BinaryRawHEDataset

from scripts.predicting.tta_predictor import TTAPredictor

from albumentations import (
    Resize,
    HorizontalFlip,
    Compose,
    Rotate
)

tqdm.monitor_interval = 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate binary efficientnet')

    parser.add_argument('-it', '--input_test', default='../input/dicom-images-test', help='input test data directory')
    parser.add_argument('-ss', '--sample_submission', default='../input/sample_submission.csv', help='sample submission file')

    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('-kf', '--kfolds', required=True, help='kfold splitting', type=int)
    parser.add_argument('-mp', '--model_path', required=True, help='path to models directory')
    parser.add_argument('-ms', '--model_size', required=True, help='Size of efficientnet model', type=int)
    parser.add_argument('-is', '--image_size', required=True, help='image size', type=int)

    parser.add_argument('-bs', '--batch_size', default=4, help='size of batches', type=int)
    parser.add_argument('-w', '--workers', default=4, help='data loader wokers count', type=int)

    args = parser.parse_args()

    current_size = int(args.image_size)

    test_all_files = getListOfFiles(args.input_test)
    ss = pd.read_csv(args.sample_submission)
    ss_images_set = set(ss['ImageId'].values)

    test_all_files = [x for x in test_all_files if (os.path.split(x)[1][:-4]) in ss_images_set]
    test_all_files = np.array(test_all_files)

    full_test_preds = np.zeros((len(test_all_files)), dtype=np.float32)

    for fold_idx in range(args.kfolds):
        print('Evaluate fold {}'.format(fold_idx))
        # # # # # # # # # Model loading # # # # # # # # # # # # # # # # # # # # # #
        current_model_dir = os.path.join(args.model_path, 'kfold_{}'.format(fold_idx))
        checkpoint = torch.load(os.path.join(current_model_dir, 'best_f1_model.t7'))
        model = BinaryClassifyEfficientNet(N=args.model_size)
        model = model.cuda()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        test_dataset_base = functools.partial(
            BinaryRawHEDataset,
            files=test_all_files,
            targets=np.zeros((len(test_all_files)), dtype=np.uint8),
        )

        test_predictor = TTAPredictor(
            model=model,
            ds_base=test_dataset_base,
            batch_size=args.batch_size,
            workers=args.workers,
            base_transforms=[Resize(current_size, current_size)],
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

        test_predictor.base_transforms = []

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

    base_output_dir = os.path.split(args.output)[0]
    if not os.path.isdir(base_output_dir):
        os.makedirs(base_output_dir)

    result_df = pd.DataFrame(
        {'ImageId': [os.path.split(x)[1][:-4] for x in test_all_files],
        'raw_target': full_test_preds}
    )
    result_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
