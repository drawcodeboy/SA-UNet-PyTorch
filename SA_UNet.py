'''
author : Doby
contact : dobylive01@gmail.com
'''

import torch
from torch import nn
import torch.nn.functional as F

'''
아래 class는 사용하지 않고, 아래의 함수 dropBlock을 사용한다.
'''
class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob=0.9, sync_channel=False):
        super(DropBlock, self).__init__()
        '''
        block_size : feature에서 drop 시킬 block의 크기, 반드시 홀수여야 함.
        keep_prob : 계속 activation 시킬 Probability
        논문에서는 Keep Probability에 대해 학습을 하면서 1부터 알맞는 값까지
        선형적으로 학습하여 적합한 p값을 찾아야 한다 하지만, 그 부분은 구현이 꽤
        어려워서 0.9를 default로 모델링을 한다.
        '''
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channel = sync_channel
        self.padding_size = (self.block_size-1)//2

    def getGamma(self, feat_size):
        '''
        Gamma의 역할
        Traditional한 Dropout에서 1-keep_prob을 통해 베르누이 분포를 계산하여
        Dropout을 진행하는데 이 부분이 DropBlock에서는 영역적으로
        block_size에 따라 Drop하기 때문에 이 부분을 고려하여 아래와 같이
        Gamma를 계산하여 베르누이 분포의 Drop 시킬 확률 값으로 넘기도록 한다.
        '''
        return (1.0-self.keep_prob)/(self.block_size**2)*(feat_size**2)/((feat_size-self.block_size+1)**2)

    def dropMask(self, feat_size):
        '''
        여기서 (1-keep_prob)기반 Gamma를 토대로 Bernoulli를 쓰는 것이기 때문에
        1이 Drop할 Center Pixel이다.
        '''
        mask = torch.distributions.Bernoulli(probs=self.getGamma(feat_size)).sample((feat_size, feat_size)).to('cuda' if torch.cuda.is_available() else 'cpu')
        return mask

    def outOfRegion(self, mask_pixel):
        '''
        Mask할 Region들이 Block Size로 인해 feature map을 넘어가지 않도록
        feature map 내에서 fully하게 Drop할 수 있도록 테두리 부분에
        Mask 픽셀이 있는 경우 이를 제거 한다.
        '''
        mask_pixel[0:self.padding_size, :] = 0.
        mask_pixel[-self.padding_size:, :] = 0.
        mask_pixel[:, 0:self.padding_size] = 0.
        mask_pixel[:, -self.padding_size:] = 0.
        return mask_pixel

    def getRegion(self, mask_region):
        masking_li = []
        for i in range(0, mask_region.shape[0]):
            for j in range(0, mask_region.shape[1]):
                if mask_region[i][j] == 1.0:
                    masking_li.append((i, j))

        for (i, j) in masking_li:
            mask_region[i-self.padding_size:i+self.padding_size+1,j-self.padding_size:j+self.padding_size+1]=1.0
        return mask_region
        
    def forward(self, x):
        feat_size = x.shape[-1]
        n_channels = x.shape[-3]
        
        if self.sync_channel:
            '''
            논문 실험에서는 channel 다 통합한 같은 Masking보다
            독립적으로 하는 게 더 성능이 좋다해서 이 부분은 사용 안 할 듯
            '''
            mask = torch.where(\
                    self.getRegion(\
                        self.outOfRegion(\
                            self.dropMask(feat_size))) == 1, 0, 1).float()
            x = x * mask
            return x
        else:
            # Channel에 따라 독립적으로 Dropout
            mask = torch.stack(
                [torch.where(\
                    self.getRegion(\
                        self.outOfRegion(\
                            self.dropMask(feat_size))) == 1, 0, 1).float()\
                                 for _ in range(n_channels)], dim=0)
            x = x * mask
            return x

def dropBlock(input, block_size, keep_prob=0.9):
    N, C, H, W = input.size()
    block_size = min(block_size, W, H)
    gamma = (1.0-keep_prob)*H*W / ((block_size**2) * ((H - block_size + 1) * (W - block_size + 1)))
    noise = torch.empty((N, C, H - block_size + 1, W - block_size + 1), dtype=input.dtype, device=input.device)
    noise.bernoulli_(gamma)

    noise = F.pad(noise, [block_size // 2] * 4, value=0)
    noise = F.max_pool2d(noise, stride=(1, 1), kernel_size=(block_size, block_size), padding=block_size // 2)
    noise = 1 - noise
    return input * noise

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        maxPool = torch.max(x, dim=-3)[0].unsqueeze(dim=-3)
        avgPool = torch.mean(x, dim=-3).unsqueeze(dim=-3)
        concat = torch.cat([maxPool, avgPool], dim=-3)
        SA = self.conv(concat)
        x = x * SA
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, block_size, keep_prob):
        super().__init__()

        # Block Size는 이미지 사이즈의 10%인 22로 설정
        self.keep_prob = keep_prob
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.drop1 = DropBlock(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)
        self.drop2 = DropBlock(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        if self.training == True:
            x = dropBlock(x, 
                          block_size=(x.shape[-1]//5 if x.shape[-1]//5%2==1 else x.shape[-1]//5+1), 
                          keep_prob=self.keep_prob)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.training == True:
            x = dropBlock(x, 
                          block_size=(x.shape[-1]//5 if x.shape[-1]//5%2==1 else x.shape[-1]//5+1), 
                          keep_prob=self.keep_prob)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters, block_size, keep_prob):
        super(EncoderBlock, self).__init__()

        self.convBlk = ConvBlock(in_filters, out_filters, block_size=block_size, keep_prob=keep_prob)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.convBlk(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters, block_size, keep_prob):
        super(DecoderBlock, self).__init__()

        self.transposeConv = nn.ConvTranspose2d(in_filters, out_filters, kernel_size=2, stride=2)
        self.convBlk = ConvBlock(in_filters, out_filters, block_size=block_size, keep_prob=keep_prob)
        
    def forward(self, x, skip):
        x = self.transposeConv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convBlk(x)
        
        return x

class SA_UNet(nn.Module):
    def __init__(self, channel, block_size, keep_prob=0.9):
        super(SA_UNet, self).__init__()

        # Constracting Path block_size = block_size, keep_prob = keep_prob
        self.keep_prob = keep_prob
        self.e1 = EncoderBlock(channel, 16, block_size=block_size, keep_prob=keep_prob)
        self.e2 = EncoderBlock(16, 32, block_size=block_size, keep_prob=keep_prob)
        self.e3 = EncoderBlock(32, 64, block_size=block_size, keep_prob=keep_prob)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.drop1 = DropBlock(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(128)

        # Bridge
        self.sam = SpatialAttentionModule()

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.drop2 = DropBlock(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(128)

        # Expanding Path
        self.d1 = DecoderBlock(128, 64, block_size=block_size, keep_prob=keep_prob)
        self.d2 = DecoderBlock(64, 32, block_size=block_size, keep_prob=keep_prob)
        self.d3 = DecoderBlock(32, 16, block_size=block_size, keep_prob=keep_prob)

        self.convOut = nn.Conv2d(16, 1, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, b = self.e3(p2)

        b = self.conv1(b)
        if self.training == True:
            b = dropBlock(b, 
                          block_size=(b.shape[-1]//5 if b.shape[-1]//5%2==1 else b.shape[-1]//5+1), 
                          keep_prob=self.keep_prob)
        b = self.bn1(b)
        b = self.relu(b)

        b = self.sam(b)

        b = self.conv2(b)
        if self.training == True:
            b = dropBlock(b, 
                          block_size=(b.shape[-1]//5 if b.shape[-1]//5%2==1 else b.shape[-1]//5+1), 
                          keep_prob=self.keep_prob)
        b = self.bn2(b)
        b = self.relu(b)

        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.convOut(d3)
        output = self.sigmoid(output)

        return output
