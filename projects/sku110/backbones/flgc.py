import torch
import torch.nn as nn


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(context, probs):
        binarized = (probs == torch.max(probs, dim=1, keepdim=True)[0]).float()
        context.save_for_backward(binarized)
        return binarized

    @staticmethod
    def backward(context, gradient_output):
        binarized, = context.saved_tensors
        gradient_output[binarized == 0] = 0
        return gradient_output


class Flgc2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=8, bias=True):
        super().__init__()
        self.in_channels_in_group_assignment_map = nn.Parameter(torch.Tensor(in_channels, groups))
        nn.init.normal_(self.in_channels_in_group_assignment_map)
        self.out_channels_in_group_assignment_map = nn.Parameter(torch.Tensor(out_channels, groups))
        nn.init.normal_(self.out_channels_in_group_assignment_map)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias)
        self.binarize = Binarize.apply

    def forward(self, x):
        map = torch.mm(self.binarize(torch.softmax(self.out_channels_in_group_assignment_map, dim=1)),
                       torch.t(self.binarize(torch.softmax(self.in_channels_in_group_assignment_map, dim=1))))
        return nn.functional.conv2d(x, self.conv.weight * map[:, :, None, None], self.conv.bias,
                                    self.conv.stride, self.conv.padding, self.conv.dilation)


if __name__ == '__main__':
    x = torch.randn(4, 3, 7, 7)
    conv = Flgc2d(3, 16, 3, padding=1, groups=4)
    out = conv(x)
    print(out.shape)