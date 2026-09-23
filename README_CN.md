# PhoenixSR

**PhoenixSR: Generative Heterogeneous Distillation Unleashes Efficient Models
for Real-World Super-Resolution**

这是一个用于真实图像超分辨率的匿名研究代码仓库。训练流程基于 BasicSR，
组合了 SwinIR 生成器、RealESRGAN 高阶退化与对抗训练、潜空间 DMD、方向性样本
加权，以及可选的 DINOv3 表征对齐损失（REPA）。

[English](README.md)

## 方法概览

精简后的唯一训练主线是 `basicsr/models/dmd_sr_model.py`：

- 生成器同时使用像素、感知、GAN 与 DMD 损失；
- 完整可训练的 fake-score UNet 拟合生成图像分布；
- 通过 LoRA 微调的 real-score UNet 拟合真实图像分布；
- 标准潜空间 DMD 以两个 score 网络预测的干净潜变量之差构造替代梯度；
- 可选 REPA 将 fake-score 中间特征与冻结的 DINOv3 patch token 对齐；
- 验证指标包含 PSNR、SSIM、NIQE 和无参考 MANIQA。

方向性加权的原始逐样本分数为：

```text
s_i = cosine(真实图像去噪残差, 生成图像去噪残差)
```

因此原始相似度范围是 `[-1, 1]`。代码随后计算全局均值为 1 的
`N * softmax(s / temperature)`，再裁剪到
`[min_weight, max_weight]`。当全局 batch size 为 1 时，softmax 权重恒为 1。

## 代码结构

```text
basicsr/models/dmd_sr_model.py       精简训练模型
basicsr/utils/dmd_util.py            时间步、方向权重与 DMD 数学
basicsr/losses/dmd_repa_loss.py      可选 DINOv3 REPA 损失
basicsr/metrics/maniqa.py            延迟加载的 PyIQA 指标
options/train/OURS/dmd_sr_swinir.yml 唯一主配置
tests/test_dmd_util.py                DMD 工具单元测试
check_anonymity.py                    匿名发布审计
```

参考工程、本地模型仓库、数据集、权重和实验输出不进入匿名发布版本。

## 安装

建议使用 Python 3.10 或 3.11。先安装与本机 CUDA 匹配的 PyTorch，再执行：

```bash
pip install -r requirements.txt
pip install -e .
```

PyIQA 第一次使用指标时会加载对应权重。若环境不能联网且没有缓存，可先在配置中
关闭 MANIQA。

## 外部数据与权重

仓库不提交数据集或预训练权重。请准备以下本地结构，或修改
`options/train/OURS/dmd_sr_swinir.yml` 中对应的相对路径：

```text
datasets/
├── train/HR/
└── RealSR/
    ├── LR/
    └── HR/

pretrained_models/
├── stable-diffusion-2-1/
├── dinov3_vitb16_pretrain_lvd1689m.pth
└── swinir_real_sr_x4.pth

dinov3/
└── dinov3/                         兼容的 DINOv3 源码包
```

Stable Diffusion 目录必须能被 Diffusers 读取；DINOv3 权重必须与
`dino_model_name` 和 `target_dim` 对应。不使用 DINOv3/REPA 时，将
`score.repa.enabled` 设为 `false`。

## 训练

```bash
python basicsr/train.py \
  -opt options/train/OURS/dmd_sr_swinir.yml \
  --launcher none
```

分布式训练可使用 BasicSR 的常规启动方式，但各 rank 的本地 batch size 必须
相同，因为方向性权重会在全部 rank 上做一次全局 softmax。

## 验证

```bash
python tests/test_dmd_util.py
python tests/test_metrics/test_maniqa.py
python -m py_compile \
  basicsr/models/dmd_sr_model.py \
  basicsr/utils/dmd_util.py \
  basicsr/losses/dmd_repa_loss.py \
  basicsr/metrics/maniqa.py
python check_anonymity.py
```

完整训练冒烟测试还需要上述外部资源和 CUDA 环境。

## 匿名发布检查

1. 将姓名、账号、主机名或单位名写入本地
   `.private_terms.txt`，每行一个；该文件已被 Git 忽略。
2. 执行 `python check_anonymity.py`。
3. 新建 Git 仓库，不复用带有个人身份的历史提交。
4. 在当前仓库单独配置匿名作者名和不暴露身份的 no-reply 邮箱。
5. 暂存文件后执行 `python check_anonymity.py --staged`。
6. 推送前人工检查暂存文件列表和远端地址。GitHub 账号名、提交签名、issue 链接
   和 release 附件同样可能暴露身份。

## 许可证与第三方代码

本仓库使用 Apache-2.0 许可证，并基于 BasicSR 及多个开源项目。必须保留的
第三方声明与许可证位于 `LICENSE/` 和 `LICENSE.txt`。这些内容属于上游项目
合规署名，不代表匿名仓库作者身份，不应为了匿名而删除。
