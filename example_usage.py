"""
时序特征聚合器使用示例

这个脚本展示了如何使用 TemporalFeatureAggregator 模块,
包括基本使用、步态识别应用示例,以及可视化输出。
"""

import torch
import torch.nn as nn
from temporal_feature_aggregator import (
    TemporalFeatureAggregator,
    BasicConv1d,
    FocalConv2d,
    clones
)


def example_basic_usage():
    """
    示例 1: 基本使用
    演示如何创建和使用 TemporalFeatureAggregator
    """
    print("=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)
    
    # 创建模块
    tfa = TemporalFeatureAggregator(
        in_channels=256,    # 特征维度
        squeeze=4,          # 压缩比例
        part_num=16         # 部位数量
    )
    
    # 打印模型信息
    print(f"\n模型结构:")
    print(f"- 输入通道数: 256")
    print(f"- 压缩比例: 4 (隐藏层维度: 64)")
    print(f"- 部位数量: 16")
    
    # 准备输入数据
    # shape: [部位数, 批次大小, 通道数, 时间步数]
    batch_size = 8
    time_steps = 30
    x = torch.randn(16, batch_size, 256, time_steps)
    
    print(f"\n输入数据:")
    print(f"- Shape: {x.shape}")
    print(f"- 含义: [16个部位, {batch_size}个样本, 256维特征, {time_steps}个时间步]")
    
    # 前向传播
    with torch.no_grad():  # 不计算梯度,仅推理
        output = tfa(x)
    
    print(f"\n输出数据:")
    print(f"- Shape: {output.shape}")
    print(f"- 含义: [16个部位, {batch_size}个样本, 256维特征]")
    print(f"- 说明: 时序维度已被聚合")
    
    # 计算参数量
    total_params = sum(p.numel() for p in tfa.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n✓ 基本使用示例完成\n")


def example_gait_recognition():
    """
    示例 2: 步态识别应用
    构建一个完整的步态识别模型
    """
    print("=" * 60)
    print("示例 2: 步态识别应用")
    print("=" * 60)
    
    class GaitRecognitionModel(nn.Module):
        """
        步态识别模型
        
        架构:
        输入 → TFA → 全局池化 → 全连接层 → 输出
        """
        
        def __init__(self, in_channels=256, part_num=16, num_classes=100):
            super().__init__()
            
            # 时序特征聚合器
            self.tfa = TemporalFeatureAggregator(
                in_channels=in_channels,
                squeeze=4,
                part_num=part_num
            )
            
            # 分类头
            self.classifier = nn.Linear(in_channels * part_num, num_classes)
            
            self.part_num = part_num
            self.in_channels = in_channels
        
        def forward(self, x):
            """
            前向传播
            
            参数:
                x: [batch, parts, channels, time_steps]
            
            返回:
                logits: [batch, num_classes]
            """
            batch_size = x.size(0)
            
            # 调整维度顺序: [batch, parts, channels, time] → [parts, batch, channels, time]
            x = x.permute(1, 0, 2, 3)
            
            # 时序聚合: [parts, batch, channels, time] → [parts, batch, channels]
            x = self.tfa(x)
            
            # 展平: [parts, batch, channels] → [batch, parts * channels]
            x = x.permute(1, 0, 2).contiguous()
            x = x.view(batch_size, -1)
            
            # 分类
            logits = self.classifier(x)
            
            return logits
    
    # 创建模型
    num_classes = 100  # 假设有100个不同的人
    model = GaitRecognitionModel(
        in_channels=256,
        part_num=16,
        num_classes=num_classes
    )
    
    print(f"\n步态识别模型:")
    print(f"- 输入: [batch, 16部位, 256通道, 30时间步]")
    print(f"- 输出: [batch, {num_classes}类别] (识别{num_classes}个不同的人)")
    
    # 模拟输入数据
    batch_size = 4
    input_data = torch.randn(batch_size, 16, 256, 30)
    
    print(f"\n模拟数据:")
    print(f"- {batch_size}个行走视频序列")
    print(f"- 每个视频: 16个人体部位 × 256维特征 × 30帧")
    
    # 前向传播
    with torch.no_grad():
        output = model(input_data)
        predictions = output.argmax(dim=1)
    
    print(f"\n推理结果:")
    print(f"- 输出logits shape: {output.shape}")
    print(f"- 预测的身份ID: {predictions.tolist()}")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n完整模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n✓ 步态识别示例完成\n")


def example_component_tests():
    """
    示例 3: 组件测试
    测试各个子组件的功能
    """
    print("=" * 60)
    print("示例 3: 组件测试")
    print("=" * 60)
    
    # 测试 BasicConv1d
    print("\n测试 BasicConv1d:")
    conv1d = BasicConv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    x = torch.randn(4, 64, 30)  # [batch, channels, length]
    y = conv1d(x)
    print(f"- 输入: {x.shape} → 输出: {y.shape}")
    print(f"- 说明: 1D卷积将通道数从64变为128")
    
    # 测试 FocalConv2d
    print("\n测试 FocalConv2d:")
    focal_conv = FocalConv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        halving=1,  # 将高度分成2部分
        padding=1
    )
    x = torch.randn(4, 3, 64, 44)  # [batch, channels, height, width]
    y = focal_conv(x)
    print(f"- 输入: {x.shape} → 输出: {y.shape}")
    print(f"- 说明: 焦点卷积,halving=1将高度维度分成2部分处理")
    
    # 测试 clones 函数
    print("\n测试 clones 函数:")
    module = nn.Linear(10, 10)
    module_list = clones(module, 5)
    print(f"- 创建了 {len(module_list)} 个独立的Linear层")
    print(f"- 每个层的参数是独立的 (深拷贝)")
    # 验证参数独立性
    param1 = list(module_list[0].parameters())[0]
    param2 = list(module_list[1].parameters())[0]
    print(f"- 参数地址不同: {param1.data_ptr() != param2.data_ptr()}")
    
    print("\n✓ 组件测试完成\n")


def example_visualization():
    """
    示例 4: 特征可视化
    可视化TFA的输出特征
    """
    print("=" * 60)
    print("示例 4: 特征分析")
    print("=" * 60)
    
    # 创建模块
    tfa = TemporalFeatureAggregator(in_channels=64, squeeze=4, part_num=4)
    
    # 准备输入
    x = torch.randn(4, 2, 64, 20)  # 4个部位,2个样本,64通道,20时间步
    
    print(f"\n输入数据:")
    print(f"- Shape: {x.shape}")
    print(f"- 均值: {x.mean().item():.4f}")
    print(f"- 标准差: {x.std().item():.4f}")
    
    # 前向传播
    with torch.no_grad():
        output = tfa(x)
    
    print(f"\n输出数据:")
    print(f"- Shape: {output.shape}")
    print(f"- 均值: {output.mean().item():.4f}")
    print(f"- 标准差: {output.std().item():.4f}")
    
    # 分析每个部位的输出
    print(f"\n各部位特征分析:")
    for i in range(4):
        part_feature = output[i]  # [2, 64]
        print(f"  部位 {i+1}:")
        print(f"    - 特征范围: [{part_feature.min().item():.2f}, {part_feature.max().item():.2f}]")
        print(f"    - 特征均值: {part_feature.mean().item():.4f}")
    
    print("\n✓ 特征分析完成\n")


def example_performance_analysis():
    """
    示例 5: 性能分析
    分析不同配置下的计算性能
    """
    print("=" * 60)
    print("示例 5: 性能分析")
    print("=" * 60)
    
    import time
    
    configs = [
        {"in_channels": 128, "squeeze": 4, "part_num": 8},
        {"in_channels": 256, "squeeze": 4, "part_num": 16},
        {"in_channels": 512, "squeeze": 4, "part_num": 16},
    ]
    
    print("\n不同配置的性能对比:\n")
    print(f"{'配置':<30} {'参数量':>15} {'推理时间(ms)':>18}")
    print("-" * 65)
    
    for config in configs:
        # 创建模型
        tfa = TemporalFeatureAggregator(**config)
        
        # 计算参数量
        params = sum(p.numel() for p in tfa.parameters())
        
        # 准备输入
        x = torch.randn(
            config["part_num"],
            4,  # batch_size
            config["in_channels"],
            30  # time_steps
        )
        
        # 预热
        with torch.no_grad():
            _ = tfa(x)
        
        # 计时
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = tfa(x)
        elapsed = (time.time() - start) * 10  # 转换为ms
        
        config_str = f"C={config['in_channels']}, P={config['part_num']}"
        print(f"{config_str:<30} {params:>12,}   {elapsed:>15.2f}")
    
    print("\n✓ 性能分析完成\n")


def main():
    """
    运行所有示例
    """
    print("\n" + "="*60)
    print("时序特征聚合器 (TFA) 使用示例")
    print("="*60 + "\n")
    
    # 运行各个示例
    example_basic_usage()
    example_gait_recognition()
    example_component_tests()
    example_visualization()
    example_performance_analysis()
    
    print("="*60)
    print("所有示例运行完成!")
    print("="*60)
    print("\n总结:")
    print("1. TFA 可以有效地聚合时序特征")
    print("2. 适用于步态识别等视频分析任务")
    print("3. 模块化设计,易于集成到现有模型")
    print("4. 可以根据需求调整参数配置")
    print("\n更多详细信息请查看 TEMPORAL_FEATURE_AGGREGATOR_README.md")
    print()


if __name__ == "__main__":
    main()
