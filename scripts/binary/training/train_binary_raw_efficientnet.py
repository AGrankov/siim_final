import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gc

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from scripts.utils.utils import getListOfFiles

from scripts.utils.torch_train_utils import progress_bar
from scripts.train.adamW import AdamW

from scripts.binary.models.binary_classifier_efficientnet import BinaryClassifyEfficientNet
from scripts.binary.dataset.binary_raw_he_dataset import BinaryRawHEDataset

from albumentations import (
    Resize,
    Normalize,
    HorizontalFlip,
    Crop,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    Rotate
)

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True


def sigmoid(x):
  return 1/(1+np.exp(-x))


def train(epoch, model, trainloader, optimizer, criterion, batch_accumulation=1):
    print('\nStart training epoch: %d' % epoch)
    model.train()
    train_loss = 0

    train_targets = []
    train_predicts = []

    print('Epoch {} optimizer LR {}'.format(epoch, optimizer.param_groups[0]['lr']))

    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        train_targets.extend(targets.numpy())
        inputs, targets = inputs.cuda(async=True), targets.float().unsqueeze(1).cuda(async=True)

        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()

        if ((batch_idx+1)%batch_accumulation == 0) or ((batch_idx+1) == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()
        train_predicts.extend(out.detach().cpu().numpy().flatten())
        train_f1_score = f1_score(train_targets, (np.array(train_predicts) > 0).astype(np.int8))

        progress_bar(batch_idx,
                     len(trainloader),
                     'Loss: {l:.8f} f1 score: {f:.4f}'.format(l = train_loss/(batch_idx+1), f = train_f1_score)
                     )
    train_targets = np.array(train_targets)
    print('Train targets pos ratio {}'.format(np.sum(train_targets) / len(train_targets)))


def validation(epoch, model, valloader, best_loss, best_f1, criterion, base_model_dir):
    model.eval()
    val_loss = 0

    valid_preds = []
    valid_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            valid_targets.extend(targets.numpy())
            inputs, targets = inputs.cuda(async=True), targets.float().unsqueeze(1).cuda(async=True)

            out = model(inputs)
            loss = criterion(out, targets)

            val_loss += loss.item()
            valid_preds.extend(out.detach().cpu().numpy().flatten())
            val_f1_score = f1_score(valid_targets, (np.array(valid_preds) > 0).astype(np.int8))

            progress_bar(batch_idx,
                         len(valloader),
                         'Loss: {l:.8f} f1 score: {f:.4f}'.format(l = val_loss/(batch_idx+1), f = val_f1_score)
                         )

    valid_targets = np.array(valid_targets)
    print('Valid targets pos ratio {}'.format(np.sum(valid_targets) / len(valid_targets)))

    val_loss = val_loss/(batch_idx+1)
    if val_loss < best_loss:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'loss': val_loss,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(base_model_dir, 'best_loss_model.t7'))
        best_loss = val_loss

    simple_f1_score = f1_score(valid_targets, (np.array(valid_preds) > 0).astype(np.int8))
    valid_preds = sigmoid(np.array(valid_preds))

    f1_scores = []
    accuracies = []
    thrs = np.arange(0.01, 1, 0.01)
    for i in tqdm(thrs):
        preds_m = (valid_preds > i).astype(np.int8)
        f1_scores.append(f1_score(valid_targets, preds_m))
        accuracies.append(accuracy_score(valid_targets, preds_m))
    f1_scores = np.array(f1_scores)
    val_f1_score = f1_scores.max()
    best_acc = np.array(accuracies).max()

    print('Epoch {} best f1 score {} best acc {}'.format(epoch, val_f1_score, best_acc))


    if val_f1_score > best_f1:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'f1_score': val_f1_score,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(base_model_dir, 'best_f1_model.t7'))
        best_f1 = val_f1_score

    return best_loss, best_f1


def main():
    parser = argparse.ArgumentParser(description='Train binary efficientnet')

    parser.add_argument('-i', '--input', default='../input/dicom-images-train', help='input data directory')
    parser.add_argument('-id', '--input_df', default='../input/train-rle.csv', help='input train df file')
    # parser.add_argument('-kf', '--kfolds', default=6, help='kfold splitting')
    # parser.add_argument('-kf', '--kfolds', default=4, help='kfold splitting')
    parser.add_argument('-kf', '--kfolds', default=6, help='kfold splitting')
    parser.add_argument('-s', '--seed', default=42, help='seed')

    parser.add_argument('-mp', '--model_path', default='../models/binary/', help='path to models directory')
    # parser.add_argument('-m', '--model_name', default='efficientnet0_is1024_dropout75', help='name of trained model')
    # parser.add_argument('-m', '--model_name', default='efficientnet5_is512_dropout90_he', help='name of trained model')
    parser.add_argument('-m', '--model_name', default='efficientnet2_is256_dropout87_he', help='name of trained model')
    parser.add_argument('-lr', '--learning_rate', default=0.0005, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.0001, help='weight decay')
    # parser.add_argument('-e', '--epochs', default=20, help='epochs count', type=int)
    parser.add_argument('-e', '--epochs', default=60, help='epochs count', type=int)
    parser.add_argument('-w', '--workers', default=6, help='data loader wokers count', type=int)

    parser.add_argument('-f', '--folds', required=True, help='make training on folds')
    # parser.add_argument('-f', '--folds', default='0,1,2', help='make training on folds')
    # parser.add_argument('-f', '--folds', default='3,4,5', help='make training on folds')

    # parser.add_argument('-bs', '--batch_size', default=4, help='size of batches', type=int)
    # parser.add_argument('-is', '--image_size', default=1024, help='image size', type=int)
    # parser.add_argument('-bs', '--batch_size', default=4, help='size of batches', type=int)
    # parser.add_argument('-is', '--image_size', default=512, help='image size', type=int)
    parser.add_argument('-bs', '--batch_size', default=16, help='size of batches', type=int)
    parser.add_argument('-is', '--image_size', default=256, help='image size', type=int)
    parser.add_argument('-ba', '--batch_accumulation', default=1, help='size accumulated batches', type=int)

    args = parser.parse_args()

    base_model_dir = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(base_model_dir):
        os.makedirs(base_model_dir)

    df = pd.read_csv(args.input_df)
    df['target'] = (df[' EncodedPixels'].str.strip() != '-1').astype(np.uint8)
    df = df[['ImageId', 'target']]
    df = df.drop_duplicates()
    targets_dict = dict(zip(df['ImageId'].values, df['target'].values))
    images_set = set(df['ImageId'].values)

    all_files = getListOfFiles(args.input)
    all_files = [x for x in all_files if (os.path.split(x)[1][:-4]) in images_set]
    all_targets = [targets_dict[os.path.split(x)[1][:-4]] for x in all_files]

    all_files = np.array(all_files)
    all_targets = np.array(all_targets)

    current_size = int(args.image_size)

    train_transforms = Compose([
        Rotate(limit=20, p=1.),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.3),

        OneOf([RandomSizedCrop(min_max_height=(960, 1024),
                                height=current_size,
                                width=current_size,
                                p=1),
              Resize(current_size, current_size, p=1)], p=1),
        HorizontalFlip(p=0.5),
    ])

    valid_transforms = Compose([
        Resize(current_size, current_size),
    ])

    folds = KFold(n_splits=args.kfolds, shuffle=False, random_state=args.seed)
    # folds = StratifiedKFold(n_splits=6, shuffle=False, random_state=split_seed)
    # for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df.index.values, train_df.std_class)):

    run_folds = [int(x) for x in args.folds.split(',')]

    for fold_idx, (trn_idx, val_idx) in enumerate(folds.split(np.arange(len(all_files)))):
        if fold_idx not in run_folds:
            continue

        train_images = all_files[trn_idx]
        train_targets = all_targets[trn_idx]

        valid_images = all_files[val_idx]
        valid_targets = all_targets[val_idx]

        train_ds = BinaryRawHEDataset(
            files=train_images,
            targets=train_targets,
            transform=train_transforms,
        )

        valid_ds = BinaryRawHEDataset(
            files=valid_images,
            targets=valid_targets,
            transform=valid_transforms,
        )

        trainloader = torch.utils.data.DataLoader(train_ds,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

        valloader = torch.utils.data.DataLoader(
                                                valid_ds,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)

        current_model_dir = os.path.join(base_model_dir, 'kfold_{}'.format(fold_idx))
        if not os.path.isdir(current_model_dir):
            os.makedirs(current_model_dir)

        model = BinaryClassifyEfficientNet(N=2, pretrained=True, dropout=0.87)

        criterion = torch.nn.modules.BCEWithLogitsLoss()

        best_loss = np.finfo(np.float32).max
        best_f1 = 0

        start_epoch = 0

        model = model.cuda()
        # model = torch.nn.DataParallel(model).cuda()

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.80)
        lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.87)

        for epoch in range(start_epoch, args.epochs):
            lr_sch.step(epoch)
            train(epoch, model, trainloader, optimizer, criterion, args.batch_accumulation)
            best_loss, best_f1 = validation(epoch, model, valloader, best_loss, best_f1, criterion, current_model_dir)

        print ('==> Best loss: {0:.8f}'.format(best_loss))
        print ('==> Best f1: {0:.8f}'.format(best_f1))


if __name__ == '__main__':
    main()
