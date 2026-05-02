import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """简化的 Decoder 模块用于 MKD (Masked Knowledge Distillation)"""
    
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 输入通道数（通常是教师特征通道数）
            out_channels: 输出通道数（通常是学生特征通道数）
        """
        super(Decoder, self).__init__()
        
        # 简化的解码器结构
        self.decoder = nn.Sequential(
            # 第一层：扩展通道
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            
            # 第二层：保持通道数
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            
            # 第三层：压缩到目标通道数
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C_in, H, W]
        Returns:
            解码后的特征 [B, C_out, H, W]
        """
        return self.decoder(x)


if __name__ == "__main__":
    # 测试代码
    print("测试 Decoder 模块...")
    
    # 创建测试数据
    batch_size = 4
    in_channels = 512  # 教师通道数
    out_channels = 256  # 学生通道数
    h, w = 32, 32
    
    x = torch.randn(batch_size, in_channels, h, w)
    
    # 创建 Decoder
    decoder = Decoder(in_channels, out_channels)
    
    # 前向传播
    y = decoder(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"预期输出形状: [{batch_size}, {out_channels}, {h}, {w}]")
    
    # 验证输出形状
    assert y.shape == (batch_size, out_channels, h, w), "输出形状不匹配"
    
    print("\n✓ Decoder 模块测试通过！")
