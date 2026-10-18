import torch
from torch import nn

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)

        return x + edge

class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()

        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        identity = x
        mid = self.in_conv(x)

        out = mid

        for i in range(self.width - 1):
            mid = self.pool(mid)

            mid = self.mid_conv[i](mid)
            edge = self.edge_enhance[i](mid)
            out = torch.cat([out,edge], dim=1)

        out = self.out_conv(out)

        return out+identity

if __name__ == '__main__':
    MEEM = MEEM(in_dim=64,hidden_dim=32)
    input = torch.randn(1, 64, 128, 128)
    output = MEEM(input)

    