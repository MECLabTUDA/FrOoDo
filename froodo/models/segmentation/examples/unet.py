import torch, torchvision
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        kernel_size = (3, 3)
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_ch,
                out_channels=out_ch,
                padding=1,
                padding_mode="reflect",
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=out_ch,
                out_channels=out_ch,
                padding=1,
                padding_mode="reflect",
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        chs=(
            3,
            32,
            64,
            128,
            256,
            512,
        ),
        mcdo=False,
    ):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)
        self.mcdo = mcdo

    def forward(self, x):
        ftrs = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            ftrs.append(x)
            if not i == len(self.enc_blocks) - 1:
                x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(
        self,
        chs=(256, 128, 64, 32, 16),
        mcdo=False,
        p=0.3,
    ):
        super().__init__()
        self.chs = chs
        self.p = p
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    chs[i], chs[i + 1], kernel_size=(2, 2), stride=(2, 2)
                )
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.extra_blocks = nn.ModuleList(
            [Block(chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.mcdo = mcdo
        self.do = nn.Dropout(p=self.p)

    def forward(self, x, encoder_features):

        """

        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            ftrs.append(x)
            if not i == len(self.enc_blocks) - 1:
                x = self.pool(x)
        return ftrs
        """
        ftrs = []
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            if self.mcdo:
                x = self.do(x)

            enc_ftrs = self.crop(encoder_features[i], x)

            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
            ftrs.append(x)
        return ftrs

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(
        self,
        enc_chs=(3, 16, 32, 64, 128),
        dec_chs=(128, 64, 32, 16),
        num_class=4,
        retain_dim=True,
        out_sz=(300, 300),
        mcdo=False,
        fix=False,
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs, mcdo=mcdo)
        self.p = 0.3
        self.decoder = Decoder(dec_chs, mcdo=mcdo, p=self.p)
        self.head = torch.nn.Sequential(
            nn.Conv2d(dec_chs[-1], num_class, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_class),
        )
        self.retain_dim = retain_dim
        self.out_sz = out_sz
        self.between_block = Block(enc_chs[-1], enc_chs[-1])
        self.fix = fix
        self.mcdo = mcdo
        self.do = nn.Dropout(p=self.p)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        lowest = enc_ftrs[::-1][0]
        if self.fix:
            lowest = self.between_block(lowest)
        dec_ftrs = self.decoder(lowest, enc_ftrs[::-1][1:])
        out = self.head(dec_ftrs[-1])
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out, *(enc_ftrs + dec_ftrs)
