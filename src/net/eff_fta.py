from torch import nn


class SF_Module(nn.Module):
    def __init__(self, input_num, n_channel, reduction, limitation):
        super(SF_Module, self).__init__()
        # Fuse Layer
        self.f_avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.f_bn = nn.BatchNorm1d(n_channel)
        self.f_linear = nn.Sequential(
            nn.Linear(n_channel, max(n_channel // reduction, limitation)), nn.SELU()
        )
        # Select Layer
        self.s_linear = nn.ModuleList(
            [
                nn.Linear(max(n_channel // reduction, limitation), n_channel)
                for _ in range(input_num)
            ]
        )

    def forward(self, x):
        # x [3, bs, c, h, w]
        fused = None
        for x_s in x:
            if fused is None:
                fused = x_s
            else:
                fused = fused + x_s
        # [bs, c, h, w]
        fused = self.f_avg(fused)  # bs,c,1,1
        fused = fused.view(fused.shape[0], fused.shape[1])
        # fused = self.f_bn(fused)
        fused = self.f_linear(fused)

        masks = []
        for i in range(len(x)):
            masks.append(self.s_linear[i](fused))
        # [3, bs, c]
        mask_stack = torch.stack(masks, dim=-1)  # bs, c, 3
        mask_stack = nn.Softmax(dim=-2)(mask_stack)

        selected = None
        for i, x_s in enumerate(x):
            mask = mask_stack[:, :, i][:, :, None, None]  # bs,c,1,1
            x_s = x_s * mask
            if selected is None:
                selected = x_s
            else:
                selected = selected + x_s
        # [bs, c, h,w]
        return selected


class FTA_Module(nn.Module):
    def __init__(self, shape, kt, kf):
        super(FTA_Module, self).__init__()
        self.r_cn = MBConvBlock(shape[2], 1, shape[3], 1, 1, True)
        self.ta_cn1 = MBConv1DBlock(shape[2], 1, shape[3], kt, 1, True)
        self.ta_cn2 = MBConv1DBlock(shape[3], 1, shape[3], kt, 1, True)
        self.ta_cn3 = MBConvBlock(shape[2], 1, shape[3], 3, 1, True)
        self.ta_cn4 = MBConvBlock(shape[3], 1, shape[3], 5, 1, True)

        self.fa_cn1 = MBConv1DBlock(shape[2], 1, shape[3], kf, 1, True)
        self.fa_cn2 = MBConv1DBlock(shape[3], 1, shape[3], kf, 1, True)
        self.fa_cn3 = MBConvBlock(shape[2], 1, shape[3], 3, 1, True)
        self.fa_cn4 = MBConvBlock(shape[3], 1, shape[3], 5, 1, True)

    def forward(self, x):
        x_r = self.r_cn(x)

        a_t = torch.mean(x, dim=-2)
        a_t = self.ta_cn1(a_t)
        a_t = self.ta_cn2(a_t)
        a_t = nn.Softmax(dim=-1)(a_t)
        a_t = a_t.unsqueeze(dim=-2)
        x_t = self.ta_cn3(x)
        x_t = self.ta_cn4(x_t)
        x_t = x_t * a_t

        a_f = torch.mean(x, dim=-1)
        a_f = self.fa_cn1(a_f)
        a_f = self.fa_cn2(a_f)
        a_f = nn.Softmax(dim=-1)(a_f)
        a_f = a_f.unsqueeze(dim=-1)
        x_f = self.fa_cn3(x)
        x_f = self.fa_cn4(x_f)
        x_f = x_f * a_f

        return x_r, x_t, x_f


class FTAnet(nn.Module):
    def __init__(self, freq_bin=360, time_segment=128):
        super(FTAnet, self).__init__()

        # fta_module
        self.fta_1 = FTA_Module((freq_bin, time_segment, 1, 32), 3, 3)
        self.fta_2 = FTA_Module((freq_bin // 2, time_segment // 2, 32, 64), 3, 3)
        self.fta_3 = FTA_Module((freq_bin // 4, time_segment // 4, 64, 128), 3, 3)
        self.fta_4 = FTA_Module((freq_bin // 4, time_segment // 4, 128, 256), 3, 3)
        # self.fta_5 = FTA_Module((freq_bin // 2, time_segment // 2, 128, 64), 3, 3)
        # self.fta_6 = FTA_Module((freq_bin, time_segment, 64, 32), 3, 3)
        # self.fta_7 = FTA_Module((freq_bin, time_segment, 32, 1), 3, 3)

        # sf_module
        self.sf_1 = SF_Module(3, 32, 4, 4)
        self.sf_2 = SF_Module(3, 64, 4, 4)
        self.sf_3 = SF_Module(3, 128, 4, 4)
        self.sf_4 = SF_Module(3, 256, 4, 4)
        # self.sf_5 = SF_Module(3, 64, 4, 4)
        # self.sf_6 = SF_Module(3, 32, 4, 4)
        # self.sf_7 = SF_Module(3, 1, 4, 4)

        # maxpool
        self.mp_1 = nn.MaxPool2d((2, 2), (2, 2))
        self.mp_2 = nn.MaxPool2d((2, 2), (2, 2))
        # self.up_1 = nn.Upsample(scale_factor=2)
        # self.up_2 = nn.Upsample(scale_factor=2)

        self.head = nn.Sequential(
            nn.Conv2d(256, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.SELU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(1280, 1025),
        )

    def forward(self, x):
        x_r, x_t, x_f = self.fta_1(x)
        x = self.sf_1([x_r, x_t, x_f])
        x = self.mp_1(x)

        x_r, x_t, x_f = self.fta_2(x)
        x = self.sf_2([x_r, x_t, x_f])
        x = self.mp_2(x)

        x_r, x_t, x_f = self.fta_3(x)
        x = self.sf_3([x_r, x_t, x_f])

        x_r, x_t, x_f = self.fta_4(x)
        x = self.sf_4([x_r, x_t, x_f])

        out = self.head(x)
        out = self.classifier(out)

        return out


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(
        self,
        in_channels,
        expand_ratio,
        out_channels,
        kernel_size,
        # padding,
        stride,
        id_skip=False,
        image_size=None,
    ):
        super().__init__()

        # Expansion phase (Inverted Bottleneck)
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.padding = padding
        self.stride = stride
        inp = in_channels
        # inp = self._block_args.input_filters  # number of input channels
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            # Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = nn.Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            # self._bn0 = nn.BatchNorm2d(
            #     num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            # )
            self._bn0 = nn.BatchNorm2d(num_features=oup)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = kernel_size
        s = stride
        # p = padding
        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            padding=(k - 1) // 2,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(num_features=oup)
        # image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        # if self.has_se:
        #     Conv2d = get_same_padding_conv2d(image_size=(1, 1))
        #     num_squeezed_channels = max(
        #         1, int(self._block_args.input_filters * self._block_args.se_ratio)
        #     )
        #     self._se_reduce = Conv2d(
        #         in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
        #     )
        #     self._se_expand = Conv2d(
        #         in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
        #     )

        # Pointwise convolution phase
        final_oup = out_channels
        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = nn.Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(num_features=final_oup)
        # self._bn2 = nn.BatchNorm2d(
        #     num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        # )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        # if self.has_se:
        #     x_squeezed = F.adaptive_avg_pool2d(x, 1)
        #     x_squeezed = self._se_reduce(x_squeezed)
        #     x_squeezed = self._swish(x_squeezed)
        #     x_squeezed = self._se_expand(x_squeezed)
        #     x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = (
            self.in_channels,
            self.out_channels,
        )
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class MBConv1DBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(
        self,
        in_channels,
        expand_ratio,
        out_channels,
        kernel_size,
        # padding,
        stride,
        id_skip=False,
        image_size=None,
    ):
        super().__init__()
        # self._block_args = block_args
        # self._bn_mom = (
        #     1 - global_params.batch_norm_momentum
        # )  # pytorch's difference from tensorflow
        # self._bn_eps = global_params.batch_norm_epsilon
        # self._bn_mom = 1 - 0.99
        # self._bn_eps = 1e-3
        # self.has_se = (self._block_args.se_ratio is not None) and (
        #     0 < self._block_args.se_ratio <= 1
        # )
        # self.id_skip = (
        #     block_args.id_skip
        # )  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.padding = padding
        self.stride = stride
        inp = in_channels
        # inp = self._block_args.input_filters  # number of input channels
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            # Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = nn.Conv1d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            # self._bn0 = nn.BatchNorm2d(
            #     num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            # )
            self._bn0 = nn.BatchNorm1d(num_features=oup)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = kernel_size
        s = stride
        # p = padding
        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = nn.Conv1d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=s,
            padding=(k - 1) // 2,
            bias=False,
        )
        self._bn1 = nn.BatchNorm1d(num_features=oup)
        # image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        # if self.has_se:
        #     Conv2d = get_same_padding_conv2d(image_size=(1, 1))
        #     num_squeezed_channels = max(
        #         1, int(self._block_args.input_filters * self._block_args.se_ratio)
        #     )
        #     self._se_reduce = Conv2d(
        #         in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
        #     )
        #     self._se_expand = Conv2d(
        #         in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
        #     )

        # Pointwise convolution phase
        final_oup = out_channels
        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = nn.Conv1d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm1d(num_features=final_oup)
        # self._bn2 = nn.BatchNorm2d(
        #     num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        # )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        # if self.has_se:
        #     x_squeezed = F.adaptive_avg_pool2d(x, 1)
        #     x_squeezed = self._se_reduce(x_squeezed)
        #     x_squeezed = self._swish(x_squeezed)
        #     x_squeezed = self._se_expand(x_squeezed)
        #     x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = (
            self.in_channels,
            self.out_channels,
        )
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


"""utils.py - Helper functions for building the model and for loading model parameters.
   These helper functions are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo


################################################################################
# Help functions for model architecture
################################################################################

# GlobalParams and BlockArgs: Two namedtuples
# Swish and MemoryEfficientSwish: Two implementations of the method
# round_filters and round_repeats:
#     Functions to calculate params for scaling model width and depth ! ! !
# get_width_and_height_from_size and calculate_output_image_size
# drop_connect: A structural design
# get_same_padding_conv2d:
#     Conv2dDynamicSamePadding
#     Conv2dStaticSamePadding
# get_same_padding_maxPool2d:
#     MaxPool2dDynamicSamePadding
#     MaxPool2dStaticSamePadding
#     It's an additional function, not used in EfficientNet,
#     but can be used in other model (such as EfficientDet).

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "width_coefficient",
        "depth_coefficient",
        "image_size",
        "dropout_rate",
        "num_classes",
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "drop_connect_rate",
        "depth_divisor",
        "min_depth",
        "include_top",
    ],
)

# Parameters for an individual model block
BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "num_repeat",
        "kernel_size",
        "stride",
        "expand_ratio",
        "input_filters",
        "output_filters",
        "se_ratio",
        "id_skip",
    ],
)

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

# Swish activation function
if hasattr(nn, "SiLU"):
    Swish = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, "p must be in range of [0,1]"

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


# Note:
# The following 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names ! ! !


def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
    The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(
            iw / sw
        )  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        image_size=None,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


def get_same_padding_maxPool2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding.
    """
    if image_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
    The padding is operated in forward function by calculating dynamically.
    """

    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = (
            [self.kernel_size] * 2
            if isinstance(self.kernel_size, int)
            else self.kernel_size
        )
        self.dilation = (
            [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        )

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = (
            [self.kernel_size] * 2
            if isinstance(self.kernel_size, int)
            else self.kernel_size
        )
        self.dilation = (
            [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        )

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        return x


################################################################################
# Helper functions for loading model params
################################################################################

# BlockDecoder: A Class for encoding and decoding BlockArgs
# efficientnet_params: A function to query compound coefficient
# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# url_map and url_map_advprop: Dicts of url_map for pretrained weights
# load_pretrained_weights: A function to load pretrained weights


class BlockDecoder(object):
    """Block Decoder for readability,
    straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert ("s" in options and len(options["s"]) == 1) or (
            len(options["s"]) == 2 and options["s"][0] == options["s"][1]
        )

        return BlockArgs(
            num_repeat=int(options["r"]),
            kernel_size=int(options["k"]),
            stride=[int(options["s"][0])],
            expand_ratio=int(options["e"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            se_ratio=float(options["se"]) if "se" in options else None,
            id_skip=("noskip" not in block_string),
        )

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.

        Args:
            block (namedtuple): A BlockArgs type argument.

        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
            "r%d" % block.num_repeat,
            "k%d" % block.kernel_size,
            "s%d%d" % (block.strides[0], block.strides[1]),
            "e%s" % block.expand_ratio,
            "i%d" % block.input_filters,
            "o%d" % block.output_filters,
        ]
        if 0 < block.se_ratio <= 1:
            args.append("se%s" % block.se_ratio)
        if block.id_skip is False:
            args.append("noskip")
        return "_".join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.

        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.

        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.

    Args:
        model_name (str): Model name to be queried.

    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
        "efficientnet-b8": (2.2, 3.6, 672, 0.5),
        "efficientnet-l2": (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    image_size=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    num_classes=1000,
    include_top=True,
):
    """Create BlockArgs and GlobalParams for efficientnet model.

    Args:
        width_coefficient (float)
        depth_coefficient (float)
        image_size (int)
        dropout_rate (float)
        drop_connect_rate (float)
        num_classes (int)

        Meaning as the name suggests.

    Returns:
        blocks_args, global_params.
    """

    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_args = [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.

    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.

    Returns:
        blocks_args, global_params
    """
    if model_name.startswith("efficientnet"):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s
        )
    else:
        raise NotImplementedError(
            "model name is not pre-defined: {}".format(model_name)
        )
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


# train with Standard methods
# check more details in paper(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)
url_map = {
    "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
    "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
    "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
    "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
    "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
    "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
    "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
    "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
}

# train with Adversarial Examples(AdvProp)
# check more details in paper(Adversarial Examples Improve Image Recognition)
url_map_advprop = {
    "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth",
    "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth",
    "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth",
    "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth",
    "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth",
    "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth",
    "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth",
    "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth",
    "efficientnet-b8": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth",
}

# TODO: add the petrained weights url map of 'efficientnet-l2'


def load_pretrained_weights(
    model, model_name, weights_path=None, load_fc=True, advprop=False, verbose=True
):
    """Loads pretrained weights from weights path or download using url.

    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        advprop (bool): Whether to load pretrained weights
                        trained with advprop (valid when weights_path is None).
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert (
            not ret.missing_keys
        ), "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)
    else:
        state_dict.pop("_fc.weight")
        state_dict.pop("_fc.bias")
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(
            ["_fc.weight", "_fc.bias"]
        ), "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)
    assert (
        not ret.unexpected_keys
    ), "Missing keys when loading pretrained weights: {}".format(ret.unexpected_keys)

    if verbose:
        print("Loaded pretrained weights for {}".format(model_name))
