import torch

batch_size = 1
seq_len = 4
num_experts = 8
top_k = 2

# ============ 模拟 MoE 路由过程 ============

# 1. router_logits: 每个 token 对所有专家的原始评分
# shape: (batch_size * seq_len, num_experts) = (4, 8)
router_logits = torch.tensor(
    [
        [2.1, 0.5, 1.8, 3.2, 0.9, 1.2, 2.5, 1.0],  # token 0 的评分
        [1.5, 3.0, 0.8, 1.1, 2.8, 1.9, 0.6, 2.2],  # token 1 的评分
        [0.7, 1.4, 2.9, 1.6, 0.5, 3.5, 2.0, 1.1],  # token 2 的评分
        [2.7, 1.0, 1.5, 0.9, 3.1, 2.4, 1.8, 0.6],  # token 3 的评分
    ]
)

print("=" * 60)
print("router_logits (每个 token 对所有专家的评分):")
print(f"shape: {router_logits.shape}")
print(router_logits)

# 2. 选择 top-k 专家
# topk 返回 (values, indices)
routing_weights, router_indices = torch.topk(router_logits, top_k, dim=1)

print("\n" + "=" * 60)
print("router_indices (每个 token 选择的 top-2 专家索引):")
print(f"shape: {router_indices.shape}")
print(router_indices)

print("\n" + "=" * 60)
print("routing_weights (对应的 top-2 评分，归一化前):")
print(f"shape: {routing_weights.shape}")
print(routing_weights)

# 3. 对权重进行 softmax 归一化
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

print("\n" + "=" * 60)
print("routing_weights (softmax 归一化后):")
print(routing_weights)
print(f"每行求和验证: {routing_weights.sum(dim=1)}")

# 4. scatter 操作：将稀疏权重转为密集表示
router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)

print("\n" + "=" * 60)
print("router_weights (scatter 后的密集表示):")
print(f"shape: {router_weights.shape}")
print(router_weights)

print("\n" + "=" * 60)
print("可视化结果:")
print("-" * 60)
for i in range(seq_len):
    print(f"\nToken {i}:")
    print(f"  选中的专家: {router_indices[i].tolist()}")
    print(f"  对应权重: {routing_weights[i][routing_weights[i] > 0].tolist()}")
    print(f"  密集向量: {router_weights[i]}")

"""
Output Log:
============================================================
router_logits (每个 token 对所有专家的评分):
shape: torch.Size([4, 8])
tensor([[2.1000, 0.5000, 1.8000, 3.2000, 0.9000, 1.2000, 2.5000, 1.0000],
        [1.5000, 3.0000, 0.8000, 1.1000, 2.8000, 1.9000, 0.6000, 2.2000],
        [0.7000, 1.4000, 2.9000, 1.6000, 0.5000, 3.5000, 2.0000, 1.1000],
        [2.7000, 1.0000, 1.5000, 0.9000, 3.1000, 2.4000, 1.8000, 0.6000]])

============================================================
router_indices (每个 token 选择的 top-2 专家索引):
shape: torch.Size([4, 2])
tensor([[3, 6],
        [1, 4],
        [5, 2],
        [4, 0]])

============================================================
routing_weights (对应的 top-2 评分，归一化前):
shape: torch.Size([4, 2])
tensor([[3.2000, 2.5000],
        [3.0000, 2.8000],
        [3.5000, 2.9000],
        [3.1000, 2.7000]])

============================================================
routing_weights (softmax 归一化后):
tensor([[0.5614, 0.4386],
        [0.5172, 0.4828],
        [0.5469, 0.4531],
        [0.5345, 0.4655]])
每行求和验证: tensor([1.0000, 1.0000, 1.0000, 1.0000])

============================================================
router_weights (scatter 后的密集表示):
shape: torch.Size([4, 8])
tensor([[0.0000, 0.0000, 0.0000, 0.5614, 0.0000, 0.0000, 0.4386, 0.0000],
        [0.0000, 0.5172, 0.0000, 0.0000, 0.4828, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.4531, 0.0000, 0.0000, 0.5469, 0.0000, 0.0000],
        [0.4655, 0.0000, 0.0000, 0.0000, 0.5345, 0.0000, 0.0000, 0.0000]])

============================================================
可视化结果:
------------------------------------------------------------

Token 0:
  选中的专家: [3, 6]
  对应权重: [0.5614035129547119, 0.4385965168476105]
  密集向量: tensor([0.0000, 0.0000, 0.0000, 0.5614, 0.0000, 0.0000, 0.4386, 0.0000])

Token 1:
  选中的专家: [1, 4]
  对应权重: [0.517241358757019, 0.48275861144065857]
  密集向量: tensor([0.0000, 0.5172, 0.0000, 0.0000, 0.4828, 0.0000, 0.0000, 0.0000])

Token 2:
  选中的专家: [5, 2]
  对应权重: [0.546875, 0.453125]
  密集向量: tensor([0.0000, 0.0000, 0.4531, 0.0000, 0.0000, 0.5469, 0.0000, 0.0000])

Token 3:
  选中的专家: [4, 0]
  对应权重: [0.5344827175140381, 0.46551722288131714]
  密集向量: tensor([0.4655, 0.0000, 0.0000, 0.0000, 0.5345, 0.0000, 0.0000, 0.0000])
"""
