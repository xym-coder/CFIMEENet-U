import torch
import torch.nn as nn
import torch.nn.functional as F
from create_module.GCE import GCE
from create_module.MEEM import MEEM

class block(nn.Module):
    def __init__(self, inplanes, planes, dcn=False):
        super(block, self).__init__()
        self.dcn = dcn

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.resid = None
        if inplanes != planes:
            self.resid = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class SwitchLayer(nn.Module):

    def __init__(self, num, chunk_channel, gcbam_group=4, fusion_type='conv1x1'):
        super().__init__()
        self.num = num
        self.fusion_type = fusion_type

        self.individual_gcbams = nn.ModuleList([
            GCE(channel=chunk_channel, group=gcbam_group)
            for _ in range(num * num)
        ])

        total_channel = chunk_channel * num
        if fusion_type == 'conv1x1':
            self.fusion = nn.Conv2d(total_channel, total_channel, kernel_size=1)
        elif fusion_type == 'mlp':
            self.fusion = MLPMixer(total_channel, num)
        else:
            self.fusion = nn.Identity()

    def forward(self, xs):

        numofeature = self.num
        splitxs = []

        for i in range(numofeature):
            splitxs.append(list(torch.chunk(xs[i], numofeature, dim=1)))

        for i in range(numofeature):
            h, w = splitxs[i][i].shape[2:]
            tmp = []

            for j in range(numofeature):
                chunk = splitxs[j][i]

                if i > j:

                    chunk = F.avg_pool2d(chunk, kernel_size=2 ** (i - j))
                elif i < j:

                    chunk = F.interpolate(chunk, (h, w), mode='bilinear', align_corners=False)

                chunk = self.individual_gcbams[j * numofeature + i](chunk)
                tmp.append(chunk)

            concatenated = torch.cat(tmp, dim=1)
            xs[i] = self.fusion(concatenated)
        return xs

class MLPMixer(nn.Module):

    def __init__(self, channel, num_chunks):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_channel = channel // num_chunks

        # Channel mixing MLP
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.chunk_channel, self.chunk_channel * 2),
            nn.GELU(),
            nn.Linear(self.chunk_channel * 2, self.chunk_channel)
        )

        self.token_mixer = nn.Sequential(
            nn.Linear(num_chunks, num_chunks * 2),
            nn.GELU(),
            nn.Linear(num_chunks * 2, num_chunks)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_chunks, self.chunk_channel, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # (B, num_chunks, H, W, chunk_channel)
        x = self.channel_mixer(x)
        x = x.permute(0, 1, 4, 2, 3)  # (B, num_chunks, chunk_channel, H, W)
        x = x.permute(0, 2, 3, 4, 1)  # (B, chunk_channel, H, W, num_chunks)
        x = self.token_mixer(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, num_chunks, chunk_channel, H, W)
        x = x.view(B, C, H, W)
        return x

class FSNet(nn.Module):
    def __init__(self, channels=64, numofblocks=4, layers=[1,2,3,4], dcn=False, gcbam_group=4, fusion_type='conv1x1'):
        super(FSNet, self).__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.layers = layers
        self.blocks = nn.ModuleList()
        self.steps = nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),

            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        self.MEEM = MEEM(
            in_dim=64,
            hidden_dim=32,
            width=3
        )

        self.switch_layer_2 = SwitchLayer(num=2, chunk_channel=channels, gcbam_group=gcbam_group,
                                          fusion_type=fusion_type)
        self.switch_layer_3 = SwitchLayer(num=3, chunk_channel=channels, gcbam_group=gcbam_group,
                                          fusion_type=fusion_type)

        for idx, l in enumerate(layers):

            self.steps.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                )
            )

            next_channels = self.channels * l

            for i in range(l):
                tmp = [block(channels, next_channels, dcn=False)]
                for j in range(self.numofblocks-1):
                    #tmp.append(block(next_channels, next_channels, dcn=dcn, use_fdconv=False))
                    tmp.append(block(next_channels, next_channels, dcn=dcn))
                self.blocks.append(nn.Sequential(*tmp))
            channels = next_channels

    def forward(self, x):
        x = self.stem(x)

        x1 = self.steps[0](x)
        x1 = self.blocks[0](x1)
        c0 = self.MEEM(x1)

        x2 = self.steps[1](x1)
        x1 = self.blocks[1](x1)
        x2 = self.blocks[2](x2)

        x3 = self.steps[2](x2)
        x1, x2 = self.switch_layer_2([x1, x2])
        x1 = self.blocks[3](x1)
        x2 = self.blocks[4](x2)
        x3 = self.blocks[5](x3)
        # 派生分支4(超低分辨率)
        x4 = self.steps[3](x3)
        x1, x2, x3 = self.switch_layer_3([x1, x2, x3])
        x1 = self.blocks[6](x1)
        x2 = self.blocks[7](x2)
        x3 = self.blocks[8](x3)
        x4 = self.blocks[9](x4)
        return x1, x2, x3, x4, c0

def FSNet_M(pretrained=False):
    model = FSNet()
    return model

if __name__ == "__main__":
    img = torch.rand((4, 3, 640, 640))
    model = FSNet_M(pretrained=True)
    x1, x2, x3, x4 = model(img)
    print(x1.shape, x2.shape, x3.shape, x4.shape)
    