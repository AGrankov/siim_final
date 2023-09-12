from torch import nn
from efficientnet_pytorch import EfficientNet

class BinaryClassifyEfficientNet(nn.Module):
    def __init__(self, N, pretrained=False, dropout=None):
        super().__init__()

        model_name = 'efficientnet-b{}'.format(int(N))

        if pretrained:
            self.encoder = EfficientNet.from_pretrained(model_name)
        else:
            self.encoder = EfficientNet.from_name(model_name)

        # self.encoder._conv_stem = nn.Conv2d(
        #     1,
        #     self.encoder._conv_stem.out_channels,
        #     kernel_size=3,
        #     stride=2,
        #     bias=False
        # )
        self.encoder._dropout = dropout
        self.encoder._fc = nn.Linear(self.encoder._conv_head.out_channels, 1)

    def forward(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable

    net = BinaryClassifyEfficientNet(N=4, pretrained=True)

    with torch.no_grad():
        x = Variable(torch.rand(1, 1, 1024, 1024))

        now = time.time()
        predictions = net(x)
        later = time.time()

        print(later - now)
