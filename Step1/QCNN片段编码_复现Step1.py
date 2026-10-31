# -*- coding: utf-8 -*-
"""
Röseler 2025 论文复现 · Step 1（最小验证）
《Efficient QCNN for Image Classification: Overcoming Hardware Constraints》
arXiv:2505.05957

目标：验证「片段编码(fragment encoding) + QCNN」这套流程能跑通、能训练。
规模：MNIST 0/1 → resize 到 4×4(16像素) → 4 个 2×2 块 → 4 比特 → QCNN。

片段编码核心（论文式 158 + 命题1）：
  经典 2×2 卷积核把 4 个像素加权求和；量子版用「编码门 E(x) 和权重门 U(w)
  交替连乘」来实现。这里用最简编码：E(x)=RY(x), U(w)=RY(w)。
  8 个单比特门(4像素+4权重交替)作用在同一个比特上，等效于角度累加，
  由命题1可知可合并成 1 个 U3 门（本脚本先直接写 8 个门，效果等价）。
"""
import pennylane as qml
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# ============ 量子设备：4 个比特 ============
n_frag = 4          # 4×4 图像 → 4 个 2×2 块 → 4 比特
dev = qml.device("default.qubit", wires=n_frag)


def encode_fragment(wire, x_block, w_block):
    """
    片段编码：把一个 2×2 块(4个像素)编码进 1 个比特。
    x_block: 4 个像素值（作编码门 E(x) 的角度）
    w_block: 4 个权重角度（作权重门 U(w) 的角度，可训练）
    顺序：E(x0)U(w0)E(x1)U(w1)E(x2)U(w2)E(x3)U(w3)，共 8 个门。
    """
    for i in range(4):
        qml.RY(x_block[i], wires=wire)   # 编码门 E(x)：像素 → 旋转角
        qml.RY(w_block[i], wires=wire)   # 权重门 U(w)：权重 → 旋转角


@qml.qnode(dev, interface="torch")
def qcnn(x, w_enc, w_qcnn):
    """
    x:     (16,) 拉平的 4×4 图像像素
    w_enc: (4,4) 片段编码权重，每块 4 个角度
    w_qcnn:(7,)  QCNN 主体权重（复用你已有的 4 比特 QCNN 结构）
    """
    # ① 片段编码层：16 像素 → 4 个 2×2 块 → 4 比特
    for i in range(n_frag):
        encode_fragment(i, x[i * 4:(i + 1) * 4], w_enc[i])

    # ② QCNN 主体（环形纠缠版）
    # 卷积层 1：RY 权重 + 环形纠缠
    for i in range(4):
        qml.RY(w_qcnn[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[3, 0])   # 环形：补首尾

    # 池化层 1：丢弃比特 1、3，信息压到 0、2
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[3, 2])

    # 卷积层 2：剩余比特 0、2
    qml.RY(w_qcnn[4], wires=0)
    qml.RY(w_qcnn[5], wires=2)
    qml.CNOT(wires=[0, 2])

    # 池化层 2：丢弃比特 2，信息压到 0
    qml.CNOT(wires=[2, 0])

    # 输出：最后分类旋转 + 测比特 0
    qml.RY(w_qcnn[6], wires=0)
    return qml.expval(qml.PauliZ(0))


# ============ 数据：MNIST 0/1，resize 到 4×4 ============
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),                 # [0,1]
        transforms.Resize((4, 4)),             # 4×4 = 16 像素
        transforms.Lambda(lambda t: t.view(-1)),  # 拉平成 (16,)
    ])
    train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)

    # 只取 0 和 1 两类
    def filter01(ds):
        xs, ys = [], []
        for x, y in ds:
            if y in (0, 1):
                xs.append(x)
                ys.append(0.0 if y == 0 else 1.0)
        return torch.stack(xs), torch.tensor(ys)

    X_train, y_train = filter01(train)
    X_test, y_test = filter01(test)
    return X_train, y_train, X_test, y_test


# ============ 训练 ============
def train(epochs=10, lr=0.1, seed=0):
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test = load_data()

    # Step 1 只验证流程，减少样本加速（qnode 逐样本调用较慢）
    X_train, y_train = X_train[:200], y_train[:200]
    X_test, y_test = X_test[:100], y_test[:100]

    # 参数：片段编码权重(4,4) + QCNN 权重(7,)，均用 torch 张量（叶子）
    w_enc = torch.randn(n_frag, 4) * 0.1
    w_enc.requires_grad_(True)
    w_qcnn = torch.randn(7) * 0.1
    w_qcnn.requires_grad_(True)

    opt = torch.optim.Adam([w_enc, w_qcnn], lr=lr)

    for epoch in range(epochs):
        opt.zero_grad()
        # 输出 ⟨Z⟩∈[-1,1]，映射到 [0,1] 与标签 0/1 比较（MSE）
        outs = torch.stack([qcnn(x, w_enc, w_qcnn) for x in X_train])
        preds = (outs + 1) / 2          # [-1,1] -> [0,1]
        loss = F.mse_loss(preds, y_train)
        loss.backward()
        opt.step()
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}", flush=True)

    # 评估
    with torch.no_grad():
        correct = 0
        for x, y in zip(X_test, y_test):
            out = qcnn(x, w_enc, w_qcnn)
            pred = 1 if out.item() > 0 else 0   # ⟨Z⟩>0 判 1，否则判 0
            correct += (pred == int(y.item()))
        acc = correct / len(X_test) * 100
    print(f"\n测试准确率 = {acc:.2f}%  (样本数 {len(X_test)})")
    print("> 随机基线约 50%，超过 50% 即证明片段编码+QCNN 流程跑通")
    return acc


if __name__ == "__main__":
    train()
