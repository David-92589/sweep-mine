"""
时序特征聚合器 (Temporal Feature Aggregator)

这个模块实现了一个用于步态识别或视频分析的时序特征聚合网络。
主要用于处理时序序列数据,通过多尺度时序模板(Multi-scale Temporal Block, MTB)
来捕获不同时间范围内的特征模式,并进行自适应的特征加权和聚合。

主要组件:
1. BasicConv1d: 基础的1D卷积层
2. FocalConv2d: 焦点2D卷积层,用于处理分割的特征
3. TemporalFeatureAggregator: 时序特征聚合器主模块

作者说明:
这段代码通常用于步态识别(Gait Recognition)领域,特别是处理序列化的人体部位特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def clones(module, N):
    """
    产生N个相同的层(深拷贝)
    
    这个函数用于创建多个独立的神经网络层副本,每个副本都有独立的参数。
    在TemporalFeatureAggregator中,用于为每个人体部位创建独立的卷积网络。
    
    参数:
        module: 要复制的PyTorch模块
        N: 复制的数量
        
    返回:
        nn.ModuleList: 包含N个独立模块副本的列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BasicConv1d(nn.Module):
    """
    基础1D卷积模块
    
    这是一个简单的1D卷积层包装器,去除了偏置项(bias=False)。
    用于构建更复杂的时序卷积网络。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        **kwargs: 其他卷积参数(如padding, stride等)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量, shape为 [batch, channels, length]
            
        返回:
            卷积后的张量
        """
        ret = self.conv(x)
        return ret


class FocalConv2d(nn.Module):
    """
    焦点2D卷积模块
    
    这个模块通过将输入在高度维度上分割成多个部分,分别进行卷积,然后再拼接。
    这种设计可以让网络关注不同高度区域的特征,类似于"焦点"机制。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int or tuple): 卷积核大小
        halving (int): 分割次数,输入会被分成 2^halving 个部分
        **kwargs: 其他卷积参数
    
    示例:
        如果halving=1,输入高度h会被分成2部分
        如果halving=2,输入高度h会被分成4部分
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量, shape为 [batch, channels, height, width]
            
        返回:
            处理后的张量,应用了LeakyReLU激活
        """
        h = x.size(2)  # 获取高度维度
        split_size = int(h // 2**self.halving)  # 计算每个分割块的大小
        z = x.split(split_size, 2)  # 在高度维度上分割
        # 对每个分割块分别进行卷积,然后在高度维度上拼接
        z = torch.cat([self.conv(_) for _ in z], 2)
        return F.leaky_relu(z, inplace=True)


class TemporalFeatureAggregator(nn.Module):
    """
    时序特征聚合器 (Temporal Feature Aggregator, TFA)
    
    这个模块的核心功能是对时序特征进行智能聚合,主要应用于步态识别等视频分析任务。
    
    ## 主要功能:
    
    1. **多尺度时序建模**: 使用两个不同的时序模板块(MTB1和MTB2)来捕获不同时间尺度的特征
       - MTB1: 使用3x1卷积和3x1池化,捕获较小时间窗口的特征
       - MTB2: 使用3x3卷积和5x5池化,捕获较大时间窗口的特征
    
    2. **自适应特征加权**: 通过卷积网络学习每个时间步的重要性权重(使用Sigmoid函数)
    
    3. **部位独立处理**: 为每个人体部位(part)创建独立的卷积网络,允许不同部位有不同的时序模式
    
    4. **特征聚合**: 结合平均池化和最大池化,然后通过学习到的权重进行加权
    
    ## 工作流程:
    
    输入: [p, n, c, s] - p个部位, n个样本, c个通道, s个时间步
    
    对于每个MTB:
    1. 使用1D卷积网络学习注意力权重 → Sigmoid激活 → 得到scores
    2. 使用池化操作提取时序特征 → 得到feature
    3. 将feature与scores相乘 → 得到加权后的特征
    
    最后: 将MTB1和MTB2的输出相加,然后进行时序最大池化,得到最终的聚合特征
    
    ## 参数说明:
    
    参数:
        in_channels (int): 输入特征的通道数
        squeeze (int): 压缩比例,用于计算隐藏层维度 (hidden_dim = in_channels // squeeze)
        part_num (int): 人体部位数量,默认为16
    
    输入格式:
        x: shape为 [p, n, c, s] 的张量
           - p: 部位数量 (如16个人体部位)
           - n: batch size (样本数量)
           - c: 通道数 (特征维度)
           - s: 时序长度 (时间步数)
    
    输出格式:
        shape为 [p, n, c] 的张量,表示每个部位每个样本的聚合特征
    
    ## 应用场景:
    
    这个模块特别适合处理:
    - 步态识别: 不同人体部位在行走过程中的时序特征
    - 动作识别: 时序视频特征的聚合
    - 任何需要对时序特征进行自适应加权聚合的任务
    """
    
    def __init__(self, in_channels, squeeze=4, part_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)  # 隐藏层维度,用于降维
        self.part_num = part_num
        
        # ==================== MTB1 (多尺度时序块 1) ====================
        # 使用3x1卷积核,捕获较小的时序窗口
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),  # 降维
            nn.LeakyReLU(inplace=True),  # 激活
            BasicConv1d(hidden_dim, in_channels, 1)  # 升维回原通道数
        )
        # 为每个部位创建独立的卷积网络
        self.conv1d3x1 = clones(conv3x1, part_num)
        # 使用3x1的池化窗口
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)
        
        # ==================== MTB2 (多尺度时序块 2) ====================
        # 使用3x3卷积核,捕获较大的时序窗口
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),  # 降维
            nn.LeakyReLU(inplace=True),  # 激活
            BasicConv1d(hidden_dim, in_channels, 3, padding=1)  # 升维,使用3x1卷积
        )
        # 为每个部位创建独立的卷积网络
        self.conv1d3x3 = clones(conv3x3, part_num)
        # 使用5x1的池化窗口(更大的感受野)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

    def forward(self, x):
        """
        前向传播 - 执行时序特征聚合
        
        参数:
            x: 输入张量, shape为 [p, n, c, s]
               - p: 部位数量
               - n: batch size
               - c: 通道数
               - s: 时序长度
        
        返回:
            聚合后的特征, shape为 [p, n, c]
        
        处理流程:
        1. 将输入按部位分割
        2. 对每个部位分别处理:
           a. MTB1分支: 3x1卷积生成权重 + 3x1池化提取特征
           b. MTB2分支: 3x3卷积生成权重 + 5x1池化提取特征
        3. 将两个分支的加权特征相加
        4. 对时序维度进行最大池化,得到最终特征
        """
        p, n, c, s = x.size()
        # 将输入按部位维度分割,每个元素shape为 [1, n, c, s]
        feature = x.split(1, 0)
        # 重塑为 [p*n, c, s] 以便进行1D卷积
        x = x.view(-1, c, s)
        
        # ==================== MTB1 处理流程 ====================
        # 步骤1: 使用卷积网络为每个部位生成注意力权重
        logits3x1 = torch.cat([
            conv(_.squeeze(0)).unsqueeze(0)  # 对每个部位独立处理
            for conv, _ in zip(self.conv1d3x1, feature)
        ], 0)
        # 步骤2: 使用Sigmoid将logits转换为0-1之间的权重分数
        scores3x1 = torch.sigmoid(logits3x1)
        
        # 步骤3: 使用池化操作提取时序特征(模板函数)
        # 结合平均池化和最大池化,捕获不同的统计特性
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)  # 恢复形状
        # 步骤4: 使用学习到的权重对特征进行加权
        feature3x1 = feature3x1 * scores3x1
        
        # ==================== MTB2 处理流程 ====================
        # 步骤1: 使用卷积网络为每个部位生成注意力权重
        logits3x3 = torch.cat([
            conv(_.squeeze(0)).unsqueeze(0)  # 对每个部位独立处理
            for conv, _ in zip(self.conv1d3x3, feature)
        ], 0)
        # 步骤2: 使用Sigmoid将logits转换为0-1之间的权重分数
        scores3x3 = torch.sigmoid(logits3x3)
        
        # 步骤3: 使用池化操作提取时序特征(模板函数)
        # 使用更大的池化窗口(5x1),捕获更长时间范围的特征
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)  # 恢复形状
        # 步骤4: 使用学习到的权重对特征进行加权
        feature3x3 = feature3x3 * scores3x3
        
        # ==================== 最终聚合 ====================
        # 将两个时序尺度的特征相加,然后在时序维度上进行最大池化
        # max(-1)[0] 表示在最后一个维度(时序维度)上取最大值
        ret = (feature3x1 + feature3x3).max(-1)[0]
        
        return ret


# ==================== 代码总结 ====================
"""
## 这个代码做了什么事情?

这段代码实现了一个**时序特征聚合器**(Temporal Feature Aggregator),主要用于处理
视频或序列数据中的时序特征,特别是在**步态识别**领域。

### 核心思想:

1. **多尺度时序建模**: 
   - 不同的时序模式需要不同的时间窗口来捕获
   - MTB1使用小窗口(3x1)捕获快速变化
   - MTB2使用大窗口(5x1)捕获慢速变化

2. **注意力机制**:
   - 不是所有时间步都同等重要
   - 使用卷积网络学习每个时间步的重要性权重
   - 通过Sigmoid函数将权重归一化到0-1之间

3. **部位独立建模**:
   - 人体不同部位的运动模式不同(如腿部vs手臂)
   - 为每个部位创建独立的参数,提高表达能力

4. **特征融合**:
   - 结合平均池化和最大池化,捕获均值和极值信息
   - 将多尺度特征相加,最后通过最大池化聚合时序信息

### 应用场景:

- **步态识别**: 识别人的行走模式,用于身份识别
- **动作识别**: 分析视频中的动作序列
- **行为分析**: 理解时序行为模式

### 输入输出:

- 输入: [部位数, 批次数, 特征维度, 时间步数]
- 输出: [部位数, 批次数, 特征维度]

通过这个模块,时序信息被智能地聚合成固定维度的特征表示,可以用于后续的分类或识别任务。
"""
