import oneflow as flow
import oneflow.nn as flownn
import oneflow.nn.functional as F
# import torch as flow
# import torch.nn as flownn
# import torch.nn.functional as F

in_channels = 10
bs = 2
out_channels = 2
conv_weight = flow.randn(out_channels, in_channels, 3, 3)
bn_weight = flow.randn(in_channels)
bn_bias = flow.rand(in_channels)
input_hw = 5
bn_running_var = flow.rand(in_channels)
bn_running_mean = flow.randn(in_channels)
eps = 1e-5

class N1(flownn.Module):
    def __init__(self):
        super(N1, self).__init__()
        self.bn1 = flownn.BatchNorm2d(in_channels, eps=eps)
        self.conv = flownn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1.bias.data = bn_bias
        self.bn1.weight.data = bn_weight
        self.bn1.running_var = bn_running_var
        self.bn1.running_mean = bn_running_mean
        self.conv.weight.data = conv_weight


    def forward(self, x):
        x = self.bn1(x)
        x = self.conv(x)
        return x

class N2(flownn.Module):
    def __init__(self):
        super(N2, self).__init__()
        self.conv = flownn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        # new_conv_weight = raw_conv_weight*bn_weight/bn_running_var
        self.conv.weight.data = conv_weight*bn_weight.view(1, in_channels, 1, 1)/flow.sqrt(bn_running_var.view(1, in_channels, 1, 1)+eps)
        # new bias
        bias = -bn_weight*bn_running_mean/flow.sqrt(bn_running_var+eps) + bn_bias
        conv_bias = bias.view(1, in_channels, 1, 1).expand(-1, -1, input_hw, input_hw)
        self.constant = flow.nn.functional.conv2d(x=conv_bias, weight=conv_weight, padding=[1, 1], stride=[1, 1], dilation=[1, 1], channel_pos="channels_first")
        # self.constant = flow.nn.functional.conv2d(input=conv_bias, weight=conv_weight, padding=1, stride=1, dilation=1)


    def forward(self, x):
        x = self.conv(x)+self.constant.cuda()
        return x



n1 = N1().cuda().eval()
n2 = N2().cuda().eval()
input = flow.randn(bs, in_channels, input_hw, input_hw).cuda()
y1 = n1(input)
y2 = n2(input)
print(n1(input))
print(n2(input))
print(y1-y2)