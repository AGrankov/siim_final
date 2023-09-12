import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import relu_fn

class BinaryClassifyEfficientNetAdd(nn.Module):
    def __init__(self, N, add_size, pretrained=False, dropout=None, mid_fc=256):
        super().__init__()

        model_name = 'efficientnet-b{}'.format(int(N))

        if pretrained:
            self.encoder = EfficientNet.from_pretrained(model_name)
        else:
            self.encoder = EfficientNet.from_name(model_name)

        self.encoder._conv_stem = nn.Conv2d(
            1,
            self.encoder._conv_stem.out_channels,
            kernel_size=3,
            stride=2,
            bias=False
        )
        self.encoder._dropout = dropout
        self.encoder._fc = nn.Linear(self.encoder._conv_head.out_channels+add_size, mid_fc)

        self.last_fc = nn.Linear(mid_fc, 1)

    def forward(self, x, add):
        x = self.encoder.extract_features(x)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.encoder._dropout:
            x = F.dropout(x, p=self.encoder._dropout, training=self.training)

        x = torch.cat((x, add), 1)
        x = relu_fn(self.encoder._fc(x))
        if self.encoder._dropout:
            x = F.dropout(x, p=self.encoder._dropout, training=self.training)
        x = self.last_fc(x)
        return x

if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable

    net = BinaryClassifyEfficientNetAdd(N=4, add_size=2, pretrained=True)

    with torch.no_grad():
        x = Variable(torch.rand(1, 1, 1024, 1024))
        add = Variable(torch.rand(1, 2))

        now = time.time()
        predictions = net(x, add)
        later = time.time()

        print(later - now)
