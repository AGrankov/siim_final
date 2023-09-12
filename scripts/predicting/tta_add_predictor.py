import numpy as np
from albumentations import Compose
import torch


class TTAAddPredictor:
    def __init__(self, model, ds_base, batch_size, base_transforms, workers=6):
        self.model = model
        self.ds_base = ds_base
        self.batch_size = batch_size
        self.workers = workers
        self.base_transforms = base_transforms

        self.ttas = []
        self.datasets = []
        self.loaders = []
        self.iterators = []

        self.put([], None)

    def put(self, predict_transforms, back_transforms):
        self.ttas.append((predict_transforms, back_transforms))

        cur_transforms = Compose(predict_transforms + self.base_transforms)
        cur_ds = self.ds_base(transform=cur_transforms)

        cur_loader = torch.utils.data.DataLoader(cur_ds,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.workers,
                                                 pin_memory=False
                                                 )

        self.datasets.append(cur_ds)
        self.loaders.append(cur_loader)
        self.iterators.append(enumerate(cur_loader))

    def reset(self):
        self.iterators = [enumerate(loader) for loader in self.loaders]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loaders[0])

    def __next__(self):
        predictions = []
        targets = None
        for tta_idx in range(len(self.ttas)):
            _, vars = next(self.iterators[tta_idx])
            if len(vars) == 3:
                inp, add, target = vars
                if tta_idx == 0:
                    targets = target.numpy()
            else:
                inp, add = vars

            with torch.no_grad():
                inp = inp.cuda()
                add = add.cuda()
                out = self.model(inp, add)
                y_pred = torch.sigmoid(out[:, 0].detach()).cpu().numpy()

            if self.ttas[tta_idx][1] is not None:
                y_pred = np.array([self.ttas[tta_idx][1](image=x)['image'] for x in y_pred])
            predictions.append(y_pred)

        result = np.mean(np.array(predictions), axis=0)
        return result, targets
