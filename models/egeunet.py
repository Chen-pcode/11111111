import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
import math


class DepthWiseConv2d(nn.Module):
    """
    深度可分离卷积。

    计算流程分成两步：
    1. `conv1` 做逐通道卷积，每个输入通道各自卷积，不发生通道混合。
    2. `conv2` 用 1x1 卷积做通道融合，把逐通道提取到的局部模式重新组合。

    这种结构比标准卷积更省参数和计算量，但当前文件里并没有实际用到它，
    更像是作者实验过程中保留下来的通用模块。
    """

    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=dim_in
        )
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r"""
    来自 ConvNeXt 的 LayerNorm 实现，兼容两种张量排布：
    1. `channels_last` : [B, H, W, C]
    2. `channels_first`: [B, C, H, W]

    本项目中的特征图基本都是 `channels_first`，所以大多数地方都会显式指定
    `data_format='channels_first'`。自己实现这一版的主要原因是 PyTorch 默认的
    `F.layer_norm` 更适合最后一个维度是通道的情况。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 在通道维上做标准化：对每个空间位置的各通道值做归一化。
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class group_aggregation_bridge(nn.Module):
    """
    Group Aggregation Bridge, 简称 GAB。

    这是论文里用来改造 skip connection 的关键模块。普通 U-Net 会直接把 encoder
    的低层特征送到 decoder，而这里会额外引入：
    1. 更高层的语义特征 `xh`
    2. 当前层的低层细节特征 `xl`
    3. 当前尺度的辅助分割预测 `mask`

    模块想解决的问题：
    1. 低层特征细节多，但语义不够强，直接跳连容易把背景噪声也带回 decoder。
    2. 高层特征语义强，但空间分辨率低，单独用它又不够精细。
    3. 辅助预测 `mask` 可以当作一种显式引导，让桥接模块更关注疑似病灶区域。

    实现方式：
    1. 先把高层特征投影到和低层特征相同的通道规模。
    2. 上采样到和低层特征相同的空间尺寸。
    3. 按通道切成 4 组，每组配一个不同膨胀率的 depth-wise 卷积分支。
    4. 每组都拼接同一张 mask，让 mask 参与特征筛选。
    5. 最后把 4 组结果拼回去，再通过 1x1 卷积压回 `dim_xl` 个通道。
    """

    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()

        # 把高层特征通道数映射到低层特征通道数，方便后续逐组融合。
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)

        # 高层和低层特征都会被分成 4 组。这里的 group_size 实际等于每组里
        # 一份高层子特征 + 一份低层子特征拼起来后的通道数，不包含额外的 mask 通道。
        group_size = dim_xl // 2

        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                dilation=d_list[0],
                groups=group_size + 1
            )
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                dilation=d_list[1],
                groups=group_size + 1
            )
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                dilation=d_list[2],
                groups=group_size + 1
            )
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(
                group_size + 1,
                group_size + 1,
                kernel_size=3,
                stride=1,
                padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                dilation=d_list[3],
                groups=group_size + 1
            )
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )

    def forward(self, xh, xl, mask):
        """
        参数含义：
        - xh: 高层特征，语义强，分辨率低。形状一般是 [B, C_high, H_small, W_small]
        - xl: 低层特征，分辨率高，细节丰富。形状一般是 [B, C_low, H, W]
        - mask: 当前尺度的辅助分割图，形状是 [B, 1, H, W]

        返回值：
        - 融合后的 skip 特征，形状与 `xl` 对齐，即 [B, C_low, H, W]
        """
        if hasattr(self, 'frequency'):
            xl = self.frequency(xl, mask)
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)

        # 切成 4 组，分别走不同膨胀率分支，构造多尺度感受野。
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)

        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    """
    Grouped Multi-axis Hadamard Product Attention, 简称 GHPA。

    这是 EGE-UNet 的核心特征提取模块。它把输入通道平均切成 4 组，分别在不同轴向上
    做注意力或局部卷积，然后再拼接回来。

    4 个分支分别是：
    1. xy 分支：在二维空间平面上建模，相当于关注 H-W 平面的空间模式。
    2. zx 分支：把宽度维调到前面，沿“通道-高度”关系建模。
    3. zy 分支：把高度维调到前面，沿“通道-宽度”关系建模。
    4. dw 分支：做局部 depth-wise 卷积，补充普通局部感受野信息。

    名字里的 Hadamard Product 指逐元素相乘，也就是：
    特征图 * 学习到的注意力权重图
    这种做法比全量自注意力更轻量，适合做医学图像分割中的高效 backbone/block。
    """

    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        # xy 分支的可学习模板。初始尺寸较小，forward 时会插值到当前特征图尺寸。
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1)
        )

        # zx 分支模板：后面会把张量重排后送入 1D depth-wise 卷积。
        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1)
        )

        # zy 分支模板，与 zx 类似，只是对应的重排轴不同。
        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1)
        )

        # 局部分支：先做一次 1x1 调整，再做 depth-wise 3x3 卷积。
        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        # 拼回 4 组后，再做一次 depth-wise + point-wise 输出到目标通道数。
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        """
        输入:
        - x: [B, C, H, W]

        输出:
        - [B, dim_out, H, W]

        注意这里不会改变空间尺寸，只改变特征表达和输出通道数。
        """
        x = self.norm1(x)

        # 平均切成 4 组，每组处理不同类型的空间/通道关系。
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        # xy 分支：直接在 H-W 平面上学习一个二维权重图。
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(
            F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True)
        )

        # zx 分支：
        # 先把张量从 [B, C, H, W] 变成 [B, W, C, H]，这样对每个宽度位置来说，
        # 后续 1D 卷积会在“通道-高度”展开的结构上建模。
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)
        ).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)

        # zy 分支：
        # 把张量改成 [B, H, C, W]，在“通道-宽度”关系上建模。
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)
        ).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        # dw 分支：保留普通卷积擅长的局部纹理和邻域信息。
        x4 = self.dw(x4)

        # 把 4 组不同感受方式得到的特征重新合并。
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class EGEUNet(nn.Module):
    """
    EGE-UNet 主网络。

    它整体还是 U-Net 的编码器-解码器框架，但做了两处关键增强：
    1. 在深层编码/解码模块中引入 GHPA，提升多轴向特征建模能力。
    2. 在 skip connection 上引入 GAB，并结合深监督预测引导特征融合。

    参数说明：
    - num_classes: 分割类别数。当前仓库默认是 1，对应二分类分割。
    - input_channels: 输入图像通道数，皮肤镜图像默认是 3。
    - c_list: 6 个阶段的通道数配置，决定模型宽度。
    - bridge: 是否启用 GAB 模块。
    - gt_ds: 是否启用深监督。

    前向输出：
    - 当 `gt_ds=True` 时，返回 `(gt_pre_tuple, out)`：
      `gt_pre_tuple` 是 5 个不同尺度的辅助预测图，`out` 是最终预测图。
    - 当 `gt_ds=False` 时，只返回最终预测图。
    """

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True,
             freq_mode='off', freq_gate='mask', freq_stages=(1, 2, 3)):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds

        # 编码器前 3 层用普通卷积，主要提取低层边缘、纹理和局部结构。
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )

        # 编码器后 3 层用 GHPA，在计算量较可控的阶段引入更强的上下文建模。
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        if bridge:
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')

        if gt_ds:
            # 5 个辅助预测头，对应 decoder 的 5 个尺度。
            # 它们输出 1 通道概率图，训练时参与深监督损失。
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')

        # 解码器前 3 层仍然使用 GHPA，浅层则退回普通卷积，兼顾效率和恢复细节。
        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        # 编码器和解码器都使用 GroupNorm，而不是 BatchNorm。
        # 医学图像训练里 batch size 往往偏小，GroupNorm 对小 batch 更稳定。
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        # 最后一层把通道数映射到类别数。
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

        if freq_mode not in {'off', 'low', 'high', 'both', 'spatial'}:
            raise ValueError('Invalid freq_mode')
        if freq_mode != 'off':
            if not bridge or not gt_ds:
                raise ValueError('The frequency path requires bridge=True and gt_ds=True')
            if not freq_stages or len(set(freq_stages)) != len(freq_stages) or any(s not in range(1, 6) for s in freq_stages):
                raise ValueError('freq_stages must contain unique stage numbers from 1 to 5')
            from models.frequency_bridge import FrequencyBridge
            # Attach after baseline initialization to preserve shared initial weights.
            for stage in freq_stages:
                getattr(self, f'GAB{stage}').frequency = FrequencyBridge(
                    c_list[stage - 1], mode=freq_mode, gate=freq_gate)

    def _init_weights(self, m):
        """
        权重初始化策略。

        这里基本遵循卷积网络常见初始化方式：
        - Linear: 截断正态分布
        - Conv1d/Conv2d: 接近 He 初始化
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        输入:
        - x: [B, 3, H, W]

        输出:
        - gt_ds=True:
          (
              (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1),
              out0
          )
          其中每个输出都已经经过 sigmoid，值域是 [0, 1]。

        - gt_ds=False:
          out0

        尺度变化路径：
        输入 HxW
        -> encoder1 后池化为 H/2
        -> encoder2 后池化为 H/4
        -> encoder3 后池化为 H/8
        -> encoder4 后池化为 H/16
        -> encoder5 后池化为 H/32
        -> encoder6 保持 H/32
        -> decoder 再逐级恢复到原图大小
        """

        # ---------------- Encoder ----------------
        # 每一级都做：
        # 卷积/注意力模块 -> GroupNorm -> GELU -> MaxPool
        # 并把池化后的结果保存成 t1~t5，作为后续 skip feature。
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # [B, c0, H/2,  W/2]

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # [B, c1, H/4,  W/4]

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # [B, c2, H/8,  W/8]

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # [B, c3, H/16, W/16]

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # [B, c4, H/32, W/32]

        # 最深层不再做池化，保持 H/32，作为瓶颈语义特征。
        out = F.gelu(self.encoder6(out))
        t6 = out  # [B, c5, H/32, W/32]

        # ---------------- Decoder stage 5 ----------------
        # 这一层还在 H/32 尺度。
        # 先解码得到 out5，再用 GAB 融合 t6 / t5 / gt_pre5，最后相加。
        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # [B, c4, H/32, W/32]
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)  # [B, 1, H/32, W/32]
            t5 = self.GAB5(t6, t5, gt_pre5)
            # 为了和原图标签直接算 loss，这里把辅助预测上采样回原图大小。
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)
        else:
            t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5)

        # ---------------- Decoder stage 4 ----------------
        # 从 H/32 上采样到 H/16。
        out4 = F.gelu(
            F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        )  # [B, c3, H/16, W/16]
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)  # [B, 1, H/16, W/16]
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4)

        # ---------------- Decoder stage 3 ----------------
        # 从 H/16 上采样到 H/8。
        out3 = F.gelu(
            F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        )  # [B, c2, H/8, W/8]
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)  # [B, 1, H/8, W/8]
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3)

        # ---------------- Decoder stage 2 ----------------
        # 从 H/8 上采样到 H/4。
        out2 = F.gelu(
            F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        )  # [B, c1, H/4, W/4]
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)  # [B, 1, H/4, W/4]
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2)

        # ---------------- Decoder stage 1 ----------------
        # 从 H/4 上采样到 H/2。
        out1 = F.gelu(
            F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        )  # [B, c0, H/2, W/2]
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)  # [B, 1, H/2, W/2]
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1)

        # 最终 1x1 卷积后再上采样 2 倍，恢复到原图分辨率。
        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True)

        if self.gt_ds:
            # 当前实现里模型内部已经做了 sigmoid，因此外部损失函数必须接收概率值，
            # 不能再换成 BCEWithLogitsLoss 这类要求原始 logits 的损失。
            return (
                torch.sigmoid(gt_pre5),
                torch.sigmoid(gt_pre4),
                torch.sigmoid(gt_pre3),
                torch.sigmoid(gt_pre2),
                torch.sigmoid(gt_pre1)
            ), torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)
