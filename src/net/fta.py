# FTANet
import torch
import torch.nn as nn
import torch.nn.functional as F


class SF_Module(nn.Module):
    def __init__(self, input_num, n_channel, reduction, limitation):
        super(SF_Module, self).__init__()
        # Fuse Layer
        self.f_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.f_bn = nn.BatchNorm1d(n_channel)
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
        fused = self.f_bn(fused)
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
        self.bn = nn.BatchNorm2d(shape[2])
        self.r_cn = nn.Sequential(nn.Conv2d(shape[2], shape[3], (1, 1)), nn.ReLU())
        self.ta_cn1 = nn.Sequential(
            nn.Conv1d(shape[2], shape[3], kt, padding=(kt - 1) // 2), nn.SELU()
        )
        self.ta_cn2 = nn.Sequential(
            nn.Conv1d(shape[3], shape[3], kt, padding=(kt - 1) // 2), nn.SELU()
        )
        self.ta_cn3 = nn.Sequential(
            nn.Conv2d(shape[2], shape[3], 3, padding=1), nn.SELU()
        )
        self.ta_cn4 = nn.Sequential(
            nn.Conv2d(shape[3], shape[3], 5, padding=2), nn.SELU()
        )

        self.fa_cn1 = nn.Sequential(
            nn.Conv1d(shape[2], shape[3], kf, padding=(kf - 1) // 2), nn.SELU()
        )
        self.fa_cn2 = nn.Sequential(
            nn.Conv1d(shape[3], shape[3], kf, padding=(kf - 1) // 2), nn.SELU()
        )
        self.fa_cn3 = nn.Sequential(
            nn.Conv2d(shape[2], shape[3], 3, padding=1), nn.SELU()
        )
        self.fa_cn4 = nn.Sequential(
            nn.Conv2d(shape[3], shape[3], 5, padding=2), nn.SELU()
        )

    def forward(self, x):
        x = self.bn(x)
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
        self.bn_layer = nn.BatchNorm2d(1)
        # bm
        self.bm_layer = nn.Sequential(
            nn.Conv2d(1, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 1, (5, 1), stride=(5, 1)),
            nn.SELU(),
        )

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
            SamePadConv2d(256, 1280, kernel_size=1),
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
        x = self.bn_layer(x)
        bm = x
        bm = self.bm_layer(bm)

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

        # x = self.up_1(x)
        # x_r, x_t, x_f = self.fta_5(x)
        # x = self.sf_5([x_r, x_t, x_f])
        # x = self.up_2(x)
        # x_r, x_t, x_f = self.fta_6(x)
        # x = self.sf_6([x_r, x_t, x_f])

        # x_r, x_t, x_f = self.fta_7(x)
        # x = self.sf_7([x_r, x_t, x_f])

        # output_pre = torch.cat([bm, x], dim=2)
        # output = nn.Softmax(dim=-2)(output_pre)
        out = self.head(x)
        out = self.classifier(out)

        return out
        # return output, output_pre


class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    It means if we have kernel_size=5 and we want to leave conv output
    be the same size as input we need to define how many pads should we have.
    In TF we have option padding='same' but in Pytorch we need to provide
    number of paddings. So we class is analog of TF padding='same'.
    As you can see we don't have `padding` as input to this class, since
    it we be calc automaticaly.
    """

    def __init__(
        self,
        inp,
        oup,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__(
            inp, oup, kernel_size, stride, 0, dilation, groups, bias, padding_mode
        )

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(
            0, (out_rows - 1) * stride + effective_filter_size_rows - in_
        )
        padding_rows = max(
            0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_
        )
        rows_odd = padding_rows % 2 != 0
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(
            x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0]
        )
        padding_cols, cols_odd = self.get_pad_odd(
            x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1]
        )

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )
