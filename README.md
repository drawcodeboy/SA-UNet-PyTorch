# SA-UNet Implementation in PyTorch
* SA-UNet Impleneted By PyTorch
* DropBlock is also implemented so that <code>block_size</code> and <code>keep_prob</code> that are important in the paper can be set to hyperparameters.
* PyTorch를 통해 SA-UNet을 구현하였습니다.
* DropBlock 또한 구현하여, 논문에서 중요하게 여기는 <code>block_size</code>, <code>keep_prob</code>을 hyperparameter로 설정할 수 있도록 합니다.
```
model = SA_UNet(channel=1, block_size=22, keep_prob=0.9)
```
* * *
# Configuration
![image](https://github.com/drawcodeboy/SA-UNet-Implementation/assets/84033023/d0801cef-b352-435c-a6e5-fe2717c78f9f)
* * *
# DropBlock Activation/Deactivation
* When calling <code>model.train()</code>, <code>model.val()</code>, <b>DropBlock is enabled/deactivated</b>, make the same configuration as Dropout and Batch Normalization.
* <code>model.train()</code>, <code>model.eval()</code> 호출 시 <b>DropBlock이 활성화/비활성화</b> 되도록 Dropout, Batch Normalization과 같은 구성을 합니다.
## <code>model.train()</code>, DropBlock ⭕
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 224, 224]             160
         DropBlock-2         [-1, 16, 224, 224]               0
       BatchNorm2d-3         [-1, 16, 224, 224]              32
              ReLU-4         [-1, 16, 224, 224]               0
            Conv2d-5         [-1, 16, 224, 224]           2,320
         DropBlock-6         [-1, 16, 224, 224]               0
       BatchNorm2d-7         [-1, 16, 224, 224]              32
              ReLU-8         [-1, 16, 224, 224]               0
         ConvBlock-9         [-1, 16, 224, 224]               0
        MaxPool2d-10         [-1, 16, 112, 112]               0
     EncoderBlock-11  [[-1, 16, 224, 224], [-1, 16, 112, 112]]               0
           Conv2d-12         [-1, 32, 112, 112]           4,640
        DropBlock-13         [-1, 32, 112, 112]               0
      BatchNorm2d-14         [-1, 32, 112, 112]              64
             ReLU-15         [-1, 32, 112, 112]               0
           Conv2d-16         [-1, 32, 112, 112]           9,248
        DropBlock-17         [-1, 32, 112, 112]               0
      BatchNorm2d-18         [-1, 32, 112, 112]              64
             ReLU-19         [-1, 32, 112, 112]               0
        ConvBlock-20         [-1, 32, 112, 112]               0
        MaxPool2d-21           [-1, 32, 56, 56]               0
     EncoderBlock-22  [[-1, 32, 112, 112], [-1, 32, 56, 56]]               0
           Conv2d-23           [-1, 64, 56, 56]          18,496
        DropBlock-24           [-1, 64, 56, 56]               0
      BatchNorm2d-25           [-1, 64, 56, 56]             128
             ReLU-26           [-1, 64, 56, 56]               0
           Conv2d-27           [-1, 64, 56, 56]          36,928
        DropBlock-28           [-1, 64, 56, 56]               0
      BatchNorm2d-29           [-1, 64, 56, 56]             128
             ReLU-30           [-1, 64, 56, 56]               0
        ConvBlock-31           [-1, 64, 56, 56]               0
        MaxPool2d-32           [-1, 64, 28, 28]               0
     EncoderBlock-33  [[-1, 64, 56, 56], [-1, 64, 28, 28]]               0
           Conv2d-34          [-1, 128, 28, 28]          73,856
        DropBlock-35          [-1, 128, 28, 28]               0
      BatchNorm2d-36          [-1, 128, 28, 28]             256
             ReLU-37          [-1, 128, 28, 28]               0
           Conv2d-38            [-1, 1, 28, 28]              99
SpatialAttentionModule-39          [-1, 128, 28, 28]               0
           Conv2d-40          [-1, 128, 28, 28]         147,584
        DropBlock-41          [-1, 128, 28, 28]               0
      BatchNorm2d-42          [-1, 128, 28, 28]             256
             ReLU-43          [-1, 128, 28, 28]               0
  ConvTranspose2d-44           [-1, 64, 56, 56]          32,832
           Conv2d-45           [-1, 64, 56, 56]          73,792
        DropBlock-46           [-1, 64, 56, 56]               0
      BatchNorm2d-47           [-1, 64, 56, 56]             128
             ReLU-48           [-1, 64, 56, 56]               0
           Conv2d-49           [-1, 64, 56, 56]          36,928
        DropBlock-50           [-1, 64, 56, 56]               0
      BatchNorm2d-51           [-1, 64, 56, 56]             128
             ReLU-52           [-1, 64, 56, 56]               0
        ConvBlock-53           [-1, 64, 56, 56]               0
     DecoderBlock-54           [-1, 64, 56, 56]               0
  ConvTranspose2d-55         [-1, 32, 112, 112]           8,224
           Conv2d-56         [-1, 32, 112, 112]          18,464
        DropBlock-57         [-1, 32, 112, 112]               0
      BatchNorm2d-58         [-1, 32, 112, 112]              64
             ReLU-59         [-1, 32, 112, 112]               0
           Conv2d-60         [-1, 32, 112, 112]           9,248
        DropBlock-61         [-1, 32, 112, 112]               0
      BatchNorm2d-62         [-1, 32, 112, 112]              64
             ReLU-63         [-1, 32, 112, 112]               0
        ConvBlock-64         [-1, 32, 112, 112]               0
     DecoderBlock-65         [-1, 32, 112, 112]               0
  ConvTranspose2d-66         [-1, 16, 224, 224]           2,064
           Conv2d-67         [-1, 16, 224, 224]           4,624
        DropBlock-68         [-1, 16, 224, 224]               0
      BatchNorm2d-69         [-1, 16, 224, 224]              32
             ReLU-70         [-1, 16, 224, 224]               0
           Conv2d-71         [-1, 16, 224, 224]           2,320
        DropBlock-72         [-1, 16, 224, 224]               0
      BatchNorm2d-73         [-1, 16, 224, 224]              32
             ReLU-74         [-1, 16, 224, 224]               0
        ConvBlock-75         [-1, 16, 224, 224]               0
     DecoderBlock-76         [-1, 16, 224, 224]               0
           Conv2d-77          [-1, 1, 224, 224]              17
          Sigmoid-78          [-1, 1, 224, 224]               0
================================================================
Total params: 483,252
Trainable params: 483,252
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 7615.28
Params size (MB): 1.84
Estimated Total Size (MB): 7617.32
----------------------------------------------------------------
```
## <code>model.eval()</code>, DropBlock ❌
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 224, 224]             160
       BatchNorm2d-2         [-1, 16, 224, 224]              32
              ReLU-3         [-1, 16, 224, 224]               0
            Conv2d-4         [-1, 16, 224, 224]           2,320
       BatchNorm2d-5         [-1, 16, 224, 224]              32
              ReLU-6         [-1, 16, 224, 224]               0
         ConvBlock-7         [-1, 16, 224, 224]               0
         MaxPool2d-8         [-1, 16, 112, 112]               0
      EncoderBlock-9  [[-1, 16, 224, 224], [-1, 16, 112, 112]]               0
           Conv2d-10         [-1, 32, 112, 112]           4,640
      BatchNorm2d-11         [-1, 32, 112, 112]              64
             ReLU-12         [-1, 32, 112, 112]               0
           Conv2d-13         [-1, 32, 112, 112]           9,248
      BatchNorm2d-14         [-1, 32, 112, 112]              64
             ReLU-15         [-1, 32, 112, 112]               0
        ConvBlock-16         [-1, 32, 112, 112]               0
        MaxPool2d-17           [-1, 32, 56, 56]               0
     EncoderBlock-18  [[-1, 32, 112, 112], [-1, 32, 56, 56]]               0
           Conv2d-19           [-1, 64, 56, 56]          18,496
      BatchNorm2d-20           [-1, 64, 56, 56]             128
             ReLU-21           [-1, 64, 56, 56]               0
           Conv2d-22           [-1, 64, 56, 56]          36,928
      BatchNorm2d-23           [-1, 64, 56, 56]             128
             ReLU-24           [-1, 64, 56, 56]               0
        ConvBlock-25           [-1, 64, 56, 56]               0
        MaxPool2d-26           [-1, 64, 28, 28]               0
     EncoderBlock-27  [[-1, 64, 56, 56], [-1, 64, 28, 28]]               0
           Conv2d-28          [-1, 128, 28, 28]          73,856
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31            [-1, 1, 28, 28]              99
SpatialAttentionModule-32          [-1, 128, 28, 28]               0
           Conv2d-33          [-1, 128, 28, 28]         147,584
      BatchNorm2d-34          [-1, 128, 28, 28]             256
             ReLU-35          [-1, 128, 28, 28]               0
  ConvTranspose2d-36           [-1, 64, 56, 56]          32,832
           Conv2d-37           [-1, 64, 56, 56]          73,792
      BatchNorm2d-38           [-1, 64, 56, 56]             128
             ReLU-39           [-1, 64, 56, 56]               0
           Conv2d-40           [-1, 64, 56, 56]          36,928
      BatchNorm2d-41           [-1, 64, 56, 56]             128
             ReLU-42           [-1, 64, 56, 56]               0
        ConvBlock-43           [-1, 64, 56, 56]               0
     DecoderBlock-44           [-1, 64, 56, 56]               0
  ConvTranspose2d-45         [-1, 32, 112, 112]           8,224
           Conv2d-46         [-1, 32, 112, 112]          18,464
      BatchNorm2d-47         [-1, 32, 112, 112]              64
             ReLU-48         [-1, 32, 112, 112]               0
           Conv2d-49         [-1, 32, 112, 112]           9,248
      BatchNorm2d-50         [-1, 32, 112, 112]              64
             ReLU-51         [-1, 32, 112, 112]               0
        ConvBlock-52         [-1, 32, 112, 112]               0
     DecoderBlock-53         [-1, 32, 112, 112]               0
  ConvTranspose2d-54         [-1, 16, 224, 224]           2,064
           Conv2d-55         [-1, 16, 224, 224]           4,624
      BatchNorm2d-56         [-1, 16, 224, 224]              32
             ReLU-57         [-1, 16, 224, 224]               0
           Conv2d-58         [-1, 16, 224, 224]           2,320
      BatchNorm2d-59         [-1, 16, 224, 224]              32
             ReLU-60         [-1, 16, 224, 224]               0
        ConvBlock-61         [-1, 16, 224, 224]               0
     DecoderBlock-62         [-1, 16, 224, 224]               0
           Conv2d-63          [-1, 1, 224, 224]              17
          Sigmoid-64          [-1, 1, 224, 224]               0
================================================================
Total params: 483,252
Trainable params: 483,252
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 7659.69
Params size (MB): 1.84
Estimated Total Size (MB): 7661.72
----------------------------------------------------------------
```
* * *
# Reference
Paper 1: [SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation](https://arxiv.org/abs/2004.03696) \
Paper 2: [DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890)
