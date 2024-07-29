#Resnet 50
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




class ResNet50(nn.Module):
  def __init__(self,num_classes=10):
    super(ResNet50,self).__init__()
    self.inplanes = 64
    self.block = Bottleneck
    self.AvgPool = nn.AvgPool2d(4)

    self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=3,stride=1,padding=1,bias=False)
    self.bn = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    ## conv1 통과후->32*32*64->spatial size 그대로!!
    self.layer1 = self._make_layer(64,num_blocks=3,stride=1)

    #layer2 부터는 feature map의 spatial size도 절반씩 감소시켜줄거임!
    self.layer2 = self._make_layer(128,num_blocks=4,stride=2)
    self.layer3 = self._make_layer(256,num_blocks=6,stride=2)
    self.layer4 = self._make_layer(512,num_blocks=3,stride=2)
    #spatial size 작아서 굳이 pooling안해도 될듯?
    self.fc = nn.Linear(512*4,num_classes)

  def _make_layer(self,planes,num_blocks,stride):
    layers = []
    #spatial size를 줄일때: layer의 첫번째 블록만 size 줄이면 됨!
    layers.append(Bottleneck(self.inplanes,planes,stride))
    self.inplanes = planes*4
    for i in range(num_blocks-1):
      layers.append(Bottleneck(self.inplanes,planes))
    return nn.Sequential(*layers)

  def forward(self,x):
     out = self.conv1(x)
     out = self.bn(out)
     out = self.relu(out)
     out = self.layer1(out)
     #print(f'after layer1:{out.shape}')
     out = self.layer2(out)
     #print(f'after layer2:{out.shape}')
     out = self.layer3(out)
     #print(f'after layer3:{out.shape}')
     out = self.layer4(out)
     #print(f'after layer4:{out.shape}')
     out = self.AvgPool(out)
     out = out.view(out.size(0),-1)
     out = self.fc(out)
     #print(out.shape)
     #print("---------end-------------")
     return out
