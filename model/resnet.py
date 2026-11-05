import torch.nn as nn
from torch.nn import functional as F
import math

class ResNet(nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, hidden, kernel_size=1)
        self.conv1_1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(hidden, hidden*2, kernel_size=1)

        self.conv2_1 = nn.Conv2d(hidden*2, hidden * 2, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(hidden*2, hidden * 2, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(hidden*2, hidden * 2, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv2d(hidden*2, hidden * 2, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(hidden*2, hidden * 4, kernel_size=1)

        self.conv3_1 = nn.Conv2d(hidden * 4, hidden * 4, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(hidden * 4, hidden * 4, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(hidden * 4, hidden * 4, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(hidden * 4, hidden * 4, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(hidden*4, hidden * 8, kernel_size=1)

        self.conv4_1 = nn.Conv2d(hidden * 8, hidden * 8, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(hidden * 8, hidden * 8, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(hidden * 8, hidden * 8, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(hidden * 8, hidden * 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden * 8, out_channel, kernel_size=1)

    def forward(self, data):
        data = data.permute(0, 3, 1, 2).contiguous()
        out = F.relu(self.conv0(data))
        temp = out
        out = F.relu(self.conv1_1(out))
        out = F.relu(self.conv1_2(out))
        out = out + temp
        temp = out
        out = F.relu(self.conv1_3(out))
        out = F.relu(self.conv1_4(out))
        out = out + temp
        out = F.relu(self.conv1(out))
        temp = out
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = out + temp
        temp = out
        out = F.relu(self.conv2_3(out))
        out = F.relu(self.conv2_4(out))
        out = out + temp
        out = F.relu(self.conv2(out))
        temp = out
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = out + temp
        temp = out
        out = F.relu(self.conv3_3(out))
        out = F.relu(self.conv3_4(out))
        out = out + temp
        out = F.relu(self.conv3(out))
        temp = out
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = out + temp
        temp = out
        out = F.relu(self.conv4_3(out))
        out = F.relu(self.conv4_4(out))
        out = out + temp
        out = self.conv4(out)
        return out.permute(0, 2, 3, 1).contiguous()
        # return out


