import torch
from torch import nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class FrequencyBridge(nn.Module):
    """
    FrequencyBridge：频域桥接模块，替代原版EGE-UNet中的GAB
    核心思想：利用小波变换DWT将特征图分解为低频分量(轮廓/大体区域) + 高频分量(边缘/细节)
    结合辅助预测mask的不确定性做门控gate，自适应筛选高低频修正特征，再逆小波IDWT融合回原图空间
    用于U-Net的skip connection，用来融合encoder低层特征 + 高层语义引导(mask_logits)
    参数：
        channels: 输入特征图通道数
        mode: 工作模式
            "low"：只处理低频分量
            "high"：只处理高频分量
            "both"：同时处理低频+高频（默认推荐）
            "spatial"：不做小波变换，仅空间域模拟同参数量的分支，用作消融实验
        gate: 门控类型
            "none": 无门控，权重固定为1
            "feature": 用高低频能量特征学习门控
            "mask": 基于辅助分割预测mask的不确定性生成门控（论文主模式）
    """
    def __init__(self, channels, mode="both", gate="mask"):
        super().__init__()
        # 校验输入参数合法性
        if mode not in {"low", "high", "both", "spatial"}:
            raise ValueError("mode must be low, high, both or spatial")
        if gate not in {"none", "feature", "mask"}:
            raise ValueError("gate must be none, feature or mask")

        self.mode = mode
        self.gate_kind = gate

        # 如果不是spatial消融模式，初始化Haar小波正/逆变换
        if mode != "spatial":
            self.dwt = DWTForward(J=1, wave="haar", mode="zero")   # 1级小波分解
            self.idwt = DWTInverse(wave="haar", mode="zero")       # 小波逆变换，重建图像

        # 低频分支：深度卷积，对低频轮廓特征做局部建模；mode=high时不需要低频分支
        if mode != "high":
            self.low_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        # 高频分支：深度卷积，处理3个方向的高频细节（水平、垂直、对角）
        if mode != "low":
            self.high_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
            # gate不是none时，用1×1卷积融合3路描述子生成门控权重
            if gate != "none":
                self.gate = nn.Conv2d(3, channels, 1)

        # 可学习缩放系数：控制修正项correction对原始特征的贡献强度
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # 可视化开关：capture=True时，保存中间特征图，用于后续画图分析
        self.capture = False
        self.last_maps = {}

    def _gate(self, low_energy, high_energy, mask_logits):
        """
        门控权重计算子函数：生成自适应权重，控制高频修正信息流入
        输入：
            low_energy: 低频能量图 [B,1,Hw,Ww]
            high_energy: 高频能量图 [B,1,Hw,Ww]
            mask_logits: 辅助分割预测的原始logits（未sigmoid）
        返回：
            gate: 门控权重 [B,C,Hw,Ww]，0~1之间，sigmoid输出
            uncertainty: 不确定性图，来自mask预测置信度
        """
        uncertainty = torch.zeros_like(low_energy)
        if self.gate_kind == "mask":
            # mask模式：利用预测概率p计算不确定性 4*p*(1-p)，p越接近0.5不确定性越高
            # mask_logits是logits，先detach阻止梯度回传，再sigmoid得到概率p
            p = mask_logits.detach().sigmoid()
            # 将不确定性图缩放至小波分解后的特征尺寸
            uncertainty = F.adaptive_avg_pool2d(4 * p * (1 - p), low_energy.shape[-2:])

        if self.gate_kind == "none":
            # 无门控，权重全部为1，不做筛选
            return torch.ones_like(low_energy), uncertainty

        # 拼接三路描述子：低频能量、高频能量、预测不确定性
        descriptor = torch.cat((low_energy, high_energy, uncertainty), dim=1)
        # 1×1卷积 + sigmoid得到0~1门控权重
        return self.gate(descriptor).sigmoid(), uncertainty

    def forward(self, x, mask_logits):
        """
        前向传播
        输入：
            x: encoder低层特征图 [B, C, H, W]
            mask_logits: 当前尺度辅助分割头输出logits（未sigmoid），用来计算不确定性门控
        返回：
            out: 经过频域门控增强后的skip特征，shape和输入x完全一致 [B,C,H,W]
        """
        height, width = x.shape[-2:]

        if self.mode == "spatial":
            # 消融实验：不使用小波变换，仅在空间域模拟高低频分支，参数量和"both"模式保持一致
            low = x
            high = x.unsqueeze(2)
        else:
            # 【Haar小波1级分解】
            # low:低频分量 [B,C,H/2,W/2]，代表大体轮廓
            # details[0]:高频细节 [B,C,3,H/2,W/2]，3个维度分别是水平、垂直、对角高频系数
            low, details = self.dwt(x)
            high = details[0]

        # 计算能量图：绝对值求平均，表征该位置信号强弱
        low_energy = low.abs().mean(1, keepdim=True)
        # high在通道后多了bands维度，需要在bands+通道维度求平均
        high_energy = high.abs().mean((1, 2)).unsqueeze(1)

        # low_delta：低频分支卷积得到的修正特征；mode=high则低频修正置0
        low_delta = torch.zeros_like(low) if self.mode == "high" else self.low_dw(low)
        gate = torch.ones_like(low_energy)
        uncertainty = torch.zeros_like(low_energy)
        high_delta = torch.zeros_like(high)

        if self.mode != "low":
            batch, channels, bands, h, w = high.shape
            # 高频分支处理：3个方向(bands)共享同一个depthwise卷积
            # 维度重排，把bands合并到batch维度，共用一套卷积权重
            packed = high.permute(0, 2, 1, 3, 4).reshape(batch * bands, channels, h, w)
            filtered = self.high_dw(packed)
            # 还原维度：[B*bands,C,h,w] → [B,C,bands,h,w]
            high_delta = filtered.reshape(batch, bands, channels, h, w).permute(0, 2, 1, 3, 4)

            # spatial模式下，使用修正后的特征重新计算能量
            if self.mode == "spatial":
                low_energy = low_delta.abs().mean(1, keepdim=True)
                high_energy = high_delta.abs().mean((1, 2)).unsqueeze(1)

            # 调用门控函数，得到gate权重与不确定性图
            gate, uncertainty = self._gate(low_energy, high_energy, mask_logits)
            # 门控作用于高频修正项，对不同位置高频细节做自适应抑制/增强
            high_delta = high_delta * gate.unsqueeze(2)

            if self.mode == "high":
                # 只使用高频模式：低频修正置0，避免IDWT梯度报错
                low_delta = high_delta[:, :, 0] * 0.0

        # 合并高低频修正项
        if self.mode == "spatial":
            # 空间消融模式，不需要逆小波，直接相加
            correction = low_delta + high_delta.squeeze(2)
        else:
            # 逆小波变换IDWT，把low_delta和high_delta重建回原始空间分辨率
            # 切片取原图尺寸，防止边界padding带来多余像素
            correction = self.idwt((low_delta, [high_delta]))[..., :height, :width]

        # 残差连接：原始特征 + alpha缩放后的频域修正特征
        out = x + self.alpha * correction

        # 保存中间可视化特征图（capture打开才生效）
        if self.capture:
            self.last_maps = {
                "low_energy": low_energy.detach().cpu(),
                "high_energy": high_energy.detach().cpu(),
                "uncertainty": uncertainty.detach().cpu(),
                "gate": gate.mean(1, keepdim=True).detach().cpu(),
                "correction": (self.alpha * correction).abs().mean(1, keepdim=True).detach().cpu(),
            }
        return out
