import numpy as np
from albumentations import Compose

import torch

class TTAPredictor:
    def __init__(self,
                 model,
                 ds_base,
                 batch_size,
                 base_transforms,
                 composition_operation=np.mean,
                 put_deafult=True,
                 workers=6,
                 ):
        self.model = model
        self.ds_base = ds_base
        self.batch_size = batch_size
        self.workers = workers
        self.base_transforms = base_transforms

        self.ttas = []
        self.datasets = []
        self.loaders = []
        self.iterators = []
        self.composition_operation = composition_operation

        if put_deafult:
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
            if type(vars) == list:
                inp, target = vars
                if tta_idx == 0:
                    targets = target.numpy()
            else:
                inp = vars

            with torch.no_grad():
                inp = inp.cuda(async=True)
                out = self.model(inp)
                y_pred = torch.sigmoid(out[:, 0].detach()).cpu().numpy()

            if self.ttas[tta_idx][1] is not None:
                y_pred = np.array([self.ttas[tta_idx][1](image=x)['image'] for x in y_pred])
            predictions.append(y_pred)

        result = self.composition_operation(np.array(predictions), axis=0)
        return result, targets
