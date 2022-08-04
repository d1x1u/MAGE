from torch import nn


class MLPHead_SimCLR(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size, use_swish=False):
        super().__init__()
        
        self.linear1 = nn.Linear(in_channels, mlp_hidden_size)
        self.bn1 = nn.BatchNorm1d(mlp_hidden_size)
        self.act = nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(mlp_hidden_size, projection_size)
        self.bn2 = nn.BatchNorm1d(projection_size)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.bn2(out)

        return out


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size, use_swish=False):
        super().__init__()

        self.linear1 = nn.Linear(in_channels, mlp_hidden_size)
        self.bn1 = nn.BatchNorm1d(mlp_hidden_size)
        self.act = nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(mlp_hidden_size, projection_size)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out


class MLP2dHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_channel, projection_size, use_swish=False):
        super().__init__()

        self.linear1 = nn.Conv2d(in_channels, mlp_hidden_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(mlp_hidden_channel)
        self.act = nn.SiLU(inplace=True) if use_swish else nn.ReLU(inplace=True)
        self.linear2 = nn.Conv2d(mlp_hidden_channel, projection_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out
