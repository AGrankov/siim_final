import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gc
import pydicom

from sklearn.model_selection import train_test_split

from scripts.utils.utils import getListOfFiles
from scripts.utils.custom_losses import BCEDiceLoss, BCELogDiceLoss, DiceLoss, LogDiceLoss
from scripts.utils.torch_train_utils import progress_bar
from scripts.utils.metrics import dice, dice_torch
from scripts.train.adamW import AdamW, RAdam

from scripts.models.unet_resnet34 import UnetResnet34
from scripts.dataset.he_dataset import HeSegmentationDataset

from albumentations import (
    Resize,
    Normalize,
    PadIfNeeded,
    HorizontalFlip,
    Crop,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    Rotate,
    HueSaturationValue,
    CLAHE,
    RandomContrast,
    RandomBrightness,
    RandomGamma,
    Blur,
    MedianBlur,
    GaussianBlur,
    JpegCompression
)

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True


def train(epoch, model, trainloader, optimizer, criterion, batch_accumulation=1):
    print('\nStart training epoch: %d' % epoch)
    model.train()
    train_loss = 0

    print('Epoch {} optimizer LR {}'.format(epoch, optimizer.param_groups[0]['lr']))

    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(async=True), targets.float().cuda(async=True)

        out = model(inputs)
        loss = criterion(out, targets.unsqueeze(1))
        loss.backward()

        if (batch_idx+1)%batch_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()

        progress_bar(batch_idx,
                     len(trainloader),
                     'Loss: {l:.8f}'.format(l = train_loss/(batch_idx+1))
                     )

def validation(epoch, model, valloader, best_loss, best_dice, criterion, base_model_dir):
    model.eval()
    val_loss = 0
    val_dice = 0

    thrs = np.arange(0.01, 1, 0.01)
    dices = [[]] * len(thrs)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.cuda(async=True), targets.float().cuda(async=True)

            out = model(inputs)
            loss = criterion(out, targets.unsqueeze(1))

            val_loss += loss.item()
            out = out.detach()
            out = torch.sigmoid(out[:, 0])

            for i, th in enumerate(thrs):
                preds_m = (out > th).float()
                dices[i].extend(dice_torch(targets, preds_m).cpu().numpy())

            val_dice = np.array(dices).mean(axis=1).max()

            progress_bar(batch_idx,
                         len(valloader),
                         'Loss: {l:.8f} dice score: {f:.4f}'.format(l = val_loss/(batch_idx+1), f = val_dice)
                         )

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

    if val_dice > best_dice:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'dice_score': val_dice,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(base_model_dir, 'best_dice_model.t7'))
        best_dice = val_dice

    return best_loss, best_dice


def main():
    parser = argparse.ArgumentParser(description='Train unet efficientnet')

    parser.add_argument('-i', '--input', default='../input/dicom-images-train', help='input data directory')
    parser.add_argument('-id', '--input_df', default='../input/train-rle.csv', help='input train df file')

    parser.add_argument('-s', '--seed', default=42, help='seed')
    parser.add_argument('-tfr', '--test_fold_ratio', default=0.2, help='test fold ratio')

    parser.add_argument('-mp', '--model_path', default='../models/', help='path to models directory')
    parser.add_argument('-m', '--model_name', default='unet_resnet34_1024_v1', help='name of trained model')

    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=1e-4, help='weight decay')
    parser.add_argument('-e', '--epochs', default=150, help='epochs count', type=int)
    parser.add_argument('-fe', '--freeze_epochs', default=30, help='epochs count of encoder freeze state', type=int)

    parser.add_argument('-w', '--workers', default=6, help='data loader wokers count', type=int)
    parser.add_argument('-is', '--image_size', default=1024, help='image size', type=int)
    # parser.add_argument('-is', '--image_size', default=960, help='image size', type=int)
    parser.add_argument('-bs', '--batch_size', default=4, help='size of batches', type=int)
    parser.add_argument('-ba', '--batch_accumulation', default=1, help='size accumulated batches', type=int)

    args = parser.parse_args()

    base_model_dir = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(base_model_dir):
        os.makedirs(base_model_dir)

    df = pd.read_csv(args.input_df)
    ids_with_mask = set(df[df[' EncodedPixels'].str.strip() != '-1']['ImageId'].values)

    current_size = int(args.image_size)

    train_transforms = Compose([
        Rotate(limit=20, p=0.5),
        OneOf([
            ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=1),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.5),

        HueSaturationValue(p=0.25),
        CLAHE(p=0.25),
        RandomContrast(p=0.25),
        RandomBrightness(p=0.25),
        RandomGamma(p=0.25),

        OneOf([
            Blur(p=1),
            MedianBlur(p=1),
            GaussianBlur(p=1),
            JpegCompression(quality_lower=50, quality_upper=100, p=1),
        ], p=0.25),

        OneOf([
            RandomSizedCrop(min_max_height=(960, 1024),
            # RandomSizedCrop(min_max_height=(768, 960),
                            height=current_size,
                            width=current_size,
                            p=1),
            Resize(current_size, current_size, p=1)
        ], p=1),

        HorizontalFlip(p=0.5),
    ])

    valid_transforms = Compose([
        Resize(current_size, current_size),
    ])

    all_files = getListOfFiles(args.input)
    all_files = [x for x in all_files if os.path.split(x)[1][:-4] in ids_with_mask]
    all_files = np.array(all_files)

    train_files, valid_files = train_test_split(all_files,
                                        test_size=args.test_fold_ratio,
                                        random_state=args.seed)

    train_ds = HeSegmentationDataset(
        dcm_files=train_files,
        masks_file=args.input_df,
        transform=train_transforms,
    )

    valid_ds = HeSegmentationDataset(
        dcm_files=valid_files,
        masks_file=args.input_df,
        transform=valid_transforms,
    )

    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    valloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = UnetResnet34(
        in_channels=3,
        num_classes=1,
        num_filters=32,
        pretrained=True,
        is_deconv=True
    )

    criterion = BCEDiceLoss()
    # criterion = BCELogDiceLoss()
    # criterion = DiceLoss()
    # criterion = LogDiceLoss()

    best_loss = np.finfo(np.float32).max
    best_iou = 0

    model = torch.nn.DataParallel(model).cuda()

    # model.module.freeze_encoder()

    optimizer = RAdam(
                        # params,
                        model.parameters(),
                        # filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        )

    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    for epoch in range(0, args.epochs):
        # if epoch == args.freeze_epochs:
        #     model.module.unfreeze_encoder()

        lr_sch.step(epoch)
        train(epoch, model, trainloader, optimizer, criterion, args.batch_accumulation)
        best_loss, best_iou = validation(epoch, model, valloader, best_loss, best_iou, criterion, base_model_dir)

    print ('==> Best loss: {0:.8f}'.format(best_loss))
    print ('==> Best iou: {0:.8f}'.format(best_iou))






if __name__ == '__main__':
    main()
