"""
torch.Tensor.transpose() 函数详细示例
========================

功能与语法:
-----------
transpose() 用于交换张量的两个维度。
语法: tensor.transpose(dim0, dim1) 或 torch.transpose(tensor, dim0, dim1)

参数说明:
---------
- dim0 (int): 要交换的第一个维度
- dim1 (int): 要交换的第二个维度

返回值说明:
-----------
返回一个新的张量视图(view),其中指定的两个维度被交换。
注意:返回的是视图,不是复制,修改返回值会影响原张量。
"""

import torch

print("=" * 60)
print("示例1: 基本的二维张量转置")
print("=" * 60)
# 创建一个2x3的张量
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"原始张量形状: {tensor_2d.shape}")
print(f"原始张量:\n{tensor_2d}")

# 交换维度0和维度1
transposed = tensor_2d.transpose(0, 1)
print(f"\n转置后形状: {transposed.shape}")
print(f"转置后张量:\n{transposed}")

# 等价于 .T 属性(仅适用于2D张量)
transposed_T = tensor_2d.T
print(f"\n使用.T属性结果:\n{transposed_T}")

print("\n" + "=" * 60)
print("示例2: 三维张量的维度交换")
print("=" * 60)
# 创建一个2x3x4的三维张量
tensor_3d = torch.arange(24).reshape(2, 3, 4)
print(f"原始张量形状: {tensor_3d.shape}")
print(f"原始张量:\n{tensor_3d}")

# 交换维度0和维度2
transposed_02 = tensor_3d.transpose(0, 2)
print(f"\n交换维度0和2后的形状: {transposed_02.shape}")
print(f"结果:\n{transposed_02}")

# 交换维度1和维度2
transposed_12 = tensor_3d.transpose(1, 2)
print(f"\n交换维度1和2后的形状: {transposed_12.shape}")
print(f"结果:\n{transposed_12}")

print("\n" + "=" * 60)
print("示例3: 验证返回的是视图(view)")
print("=" * 60)
original = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"原始张量:\n{original}")

# 转置
transposed = original.transpose(0, 1)
print(f"\n转置后:\n{transposed}")

# 修改转置后的张量
transposed[0, 0] = 999
print(f"\n修改转置张量的[0,0]位置为999")
print(f"转置张量:\n{transposed}")
print(f"原始张量也被修改:\n{original}")

print("\n" + "=" * 60)
print("示例4: 使用负数索引")
print("=" * 60)
tensor = torch.arange(24).reshape(2, 3, 4)
print(f"原始张量形状: {tensor.shape}")

# -1表示最后一个维度, -2表示倒数第二个维度
transposed_neg = tensor.transpose(-1, -2)
print(f"\n交换最后两个维度后的形状: {transposed_neg.shape}")
print(f"这等价于 transpose(1, 2): {tensor.transpose(1, 2).shape}")

print("\n" + "=" * 60)
print("示例5: 批量图像处理中的应用")
print("=" * 60)
# 模拟一个批量的图像数据: (batch_size, channels, height, width)
batch_images = torch.randn(4, 3, 28, 28)
print(f"原始图像批次形状 (NCHW格式): {batch_images.shape}")
print("N=batch_size, C=channels, H=height, W=width")

# 某些库需要 (batch_size, height, width, channels) 格式
# 先交换C和H
temp = batch_images.transpose(1, 2)  # (4, 28, 3, 28)
# 再交换C和W
hwc_format = temp.transpose(2, 3)  # (4, 28, 28, 3)
print(f"\n转换为NHWC格式: {hwc_format.shape}")

# 更简洁的方法是使用permute
hwc_format_2 = batch_images.permute(0, 2, 3, 1)
print(f"使用permute的结果: {hwc_format_2.shape}")

print("\n" + "=" * 60)
print("示例6: transpose vs permute vs contiguous")
print("=" * 60)
original = torch.arange(12).reshape(3, 4)
print(f"原始张量:\n{original}")
print(f"是否连续: {original.is_contiguous()}")

# transpose返回视图,可能不连续
transposed = original.transpose(0, 1)
print(f"\n转置后是否连续: {transposed.is_contiguous()}")

# 如果需要连续的张量,使用contiguous()
contiguous_transposed = transposed.contiguous()
print(f"调用contiguous()后是否连续: {contiguous_transposed.is_contiguous()}")

print("\n" + "=" * 60)
print("示例7: 链式调用多次transpose")
print("=" * 60)
tensor = torch.arange(120).reshape(2, 3, 4, 5)
print(f"原始形状: {tensor.shape}")

# 多次调用transpose
result = tensor.transpose(0, 1).transpose(2, 3)
print(f"先交换dim0和dim1,再交换dim2和dim3: {result.shape}")

# 注意:连续的transpose可以抵消
back_to_original = tensor.transpose(0, 1).transpose(0, 1)
print(f"\n两次交换同样的维度会恢复原状: {back_to_original.shape}")
print(f"是否与原张量相同: {torch.equal(back_to_original, tensor)}")

print("\n" + "=" * 60)
print("示例8: 实际应用 - 矩阵乘法")
print("=" * 60)
# 假设有一个权重矩阵 (input_features, output_features)
weight = torch.randn(5, 3)
# 输入数据 (batch_size, input_features)
input_data = torch.randn(2, 5)

print(f"权重矩阵形状: {weight.shape}")
print(f"输入数据形状: {input_data.shape}")

# 标准的矩阵乘法: input @ weight 需要 (2,5) @ (5,3) = (2,3)
output = input_data @ weight
print(f"\n直接矩阵乘法结果形状: {output.shape}")

# 如果权重存储为转置形式 (output_features, input_features)
weight_T = torch.randn(3, 5)
print(f"\n转置的权重形状: {weight_T.shape}")
# 需要先转置才能相乘
output_2 = input_data @ weight_T.transpose(0, 1)
print(f"转置后再相乘的结果形状: {output_2.shape}")
