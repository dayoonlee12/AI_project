import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channel, mid_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.out_channel = mid_channel * 4
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, self.out_channel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)

        if downsample or in_channel != self.out_channel or stride != 1:
            self.downsample_layers = nn.Sequential(
                nn.Conv2d(in_channel, self.out_channel, 1, stride=stride),
                nn.BatchNorm2d(self.out_channel)
            )
        else:
            self.downsample_layers = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample_layers is not None:
            identity = self.downsample_layers(identity)

        out += identity
        out = self.relu(out)
        return out

class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        h = w = [32, 16, 8]
        self.block = Bottleneck
        fwd1 = nn.Sequential(self.block(3,64,1),self.block(64*4,64)) #out dim = 64*4 = 256
        out_dim1 = 64*4
        fwd2 = nn.Sequential(self.block(64*4,128,2),self.block(128*4,128)) #out dim = 128*4 = 512
        out_dim2 = 128*4
        fwd3 = nn.Sequential(self.block(128*4,256,2),self.block(256*4,256)) #out dim = 256*4 = 1024
        out_dim3 = 256*4
        fwd = [fwd1,fwd2,fwd3]
        self.fwd = nn.ModuleList(fwd)#->forward후 shape?[1, 1024, 8, 8]

        #8->16으로 / 16->32로 upsample해서 최종적으로는 original input과 같은 size 만들어주기!(upsampe array는 뒤에서부터 진행)
        upsample = [nn.Upsample(size=(32,32),mode='bilinear'),nn.Upsample(size=(16,16),mode='bilinear')] 
        self.upsample = nn.ModuleList(upsample) 
        bwd1 = nn.Sequential(self.block(out_dim3+out_dim2,64),self.block(64*4,64))#->256
        bwd2 = nn.Sequential(self.block(256+out_dim1,128),self.block(128*4,128))
        back = [bwd2,bwd1]
        self.back  = nn.ModuleList(back)
        self.final = nn.Conv2d(128*4, 3, kernel_size = 1, bias = False)

    def forward(self, x):
        out = x
        outputs = []
        for i in range(len(self.fwd)):
            out = self.fwd[i](out)
            if i != len(self.fwd) - 1:
                outputs.append(out)
        #print(f"After forward:{out.shape}")
        for i in range(len(self.back) - 1, -1, -1):
            out = self.upsample[i](out) #일단 upsample 8*8*1024->conv2 output과 concat 하기 위해선: 16*16*1024로 upsample 후 + 16*16*512->16*16*1536(lateral connection)
            out = torch.cat((out, outputs[i]), 1)
            out = self.back[i](out)
        out = self.final(out)
        out += x #마지막엔 x와 더해줌으로써 residual connection
        return out