# LLM Training & Reasoning Experiments (Three-Part Project)

## English
This repository collects a three-part set of experiments on large language models (LLMs), covering:  
(1) training a tiny GPT from scratch, (2) optimizer comparisons for math fine-tuning, and (3) **test-time scaling** for mathematical reasoning with **Pass@k** evaluation.

## 中文
本仓库整理了一个三部分的 LLM 实验项目，内容包括：  
（1）从零训练小型 GPT，（2）数学微调中的优化器对比，以及（3）数学推理场景下的 **test-time scaling**（推理阶段多采样与聚合），并用 **Pass@k** 指标进行评估。

---

## Project Structure

## English
- `p1/` pre-training a tiny GPT (from scratch)
- `p2/` fine-tuning + optimizer comparison (SGD / Adam / LoRA)
- `p3/` test-time scaling (multi-sample decoding + Pass@k / voting)
- `assets/` figures used in README

## 中文
- `p1/` 从零预训练小型 GPT
- `p2/` 数学微调与优化器对比（SGD / Adam / LoRA）
- `p3/` test-time scaling（多采样推理 + Pass@k / 投票聚合）
- `assets/` README 中使用的图片

## Part I — Pre-train a GPT on Shakespeare (NanoGPT-based)

## English
**Goal:** build hands-on understanding of *autoregressive language modeling* by pre-training a GPT-style model on the Shakespeare corpus, then sampling Shakespeare-like text.  
**Key result:** validation loss at **iteration 5000 = 1.7067**, and a generated sample is included in the report. :contentReference[oaicite:2]{index=2}

## 中文
**目标：** 通过在 Shakespeare 语料上进行 GPT 预训练，掌握自回归语言建模与训练/推理流程，并生成 “莎士比亚风格” 文本。  
**关键结果：** 在 **5000 step** 时的验证集 loss 为 **1.7067**，并在报告中展示了生成文本样例。:contentReference[oaicite:3]{index=3}

### Notes / Tips

## English
I ran the experiment on **Kaggle GPU** and encountered a common “kernel title already in use” conflict; switching to interactive notebook execution and following the provided Kaggle guide resolved it. Loss dropped quickly in early iterations (~first 300) and then decreased more steadily. :contentReference[oaicite:4]{index=4}

## 中文
我使用 **Kaggle GPU** 运行实验时遇到过 “kernel 标题冲突（409 - Conflict）” 的问题，通过参考项目提供的 Kaggle 指南并改用交互式运行 notebook 解决。训练过程中 loss 在前期（约前 300 step）下降很快，后期趋于平稳下降。:contentReference[oaicite:5]{index=5}


---

## Part II — Fine-tune Qwen3-0.6B-Base on Math500 (SGD vs Adam vs LoRA)

## English
**Setup:** fine-tune **Qwen3-0.6B-Base** on a filtered **Math500** subset (**458 train / 55 val**) and compare **SGD**, **Adam**, and **LoRA** across learning rates (and LoRA ranks). The report includes training/validation loss curves, memory/time measurements, and final test accuracy. :contentReference[oaicite:6]{index=6}

## 中文
**实验设置：** 以 **Qwen3-0.6B-Base** 为基础模型，在处理后的 **Math500** 子集（**458 训练 / 55 验证**）上进行微调，对比 **SGD / Adam / LoRA** 在不同学习率（以及 LoRA rank）下的训练稳定性、资源消耗与测试表现。报告包含 loss 曲线、显存/时间统计以及最终准确率。:contentReference[oaicite:7]{index=7}

### Main Findings

## English
Fine-tuning significantly improved math accuracy over the base model. The best configuration in our runs was **LoRA (lr=1e-4, rank=8)** with **14.8%** test accuracy, slightly higher than **SGD (14.4%)** and **Adam (11.8%)**, while the **base model scored 4.8%**. :contentReference[oaicite:8]{index=8}

## 中文
微调能显著提升数学题解题准确率：在本组实验中，最佳配置为 **LoRA（lr=1e-4, rank=8）**，测试集准确率达到 **14.8%**，略高于 **SGD（14.4%）** 和 **Adam（11.8%）**；而 **基座模型仅为 4.8%**。:contentReference[oaicite:9]{index=9}

### Result Snapshot (from Part II report)

| Model / Optimizer | Best Setting (reported) | Test Accuracy |
|---|---:|---:|
| Base (Qwen3-0.6B-Base) | — | 4.8% |
| SGD | lr=2e-5 | 14.4% |
| Adam | lr=2e-5 | 11.8% |
| LoRA | lr=1e-4, r=8 | 14.8% |

## English
Resource usage was also logged (GPU max memory and wall-clock training time). LoRA generally offered a strong accuracy–efficiency tradeoff in our setting. :contentReference[oaicite:10]{index=10}

## 中文
我们同时记录了资源消耗（最大显存占用与训练时间）。在本实验设定下，LoRA 通常能在性能与效率之间取得更好的平衡。:contentReference[oaicite:11]{index=11}

## Part III — Test-Time Scaling Results (AIME25 Pass@k)

## English
We evaluate **Pass@k** on **AIME25** with **Qwen2.5-Math-1.5B** under different temperatures, where increasing *k* means sampling more independent solutions and counting success if **any** of the *k* samples is correct.  
Key observations from the results below:
- **Pass@k increases monotonically with k** (more samples → higher chance at least one is correct).
- **Lower temperature (0.6) performs best** across k in these runs.
- The stronger setting (second plot) shows a **large uplift**. For example at **k=16**:
  - Temp=0.6 improves from **21.4% → 31.3%**
  - Temp=1.0 improves from **14.2% → 30.5%**
  - Temp=1.2 improves from **6.4% → ~18%** (approx., as shown in the plot)

## 中文
我们在 **AIME25** 上评估 **Pass@k**：对同一道题独立采样 *k* 个解答，只要其中**任意一个**正确就算成功。  
从下方结果可以观察到：
- **k 越大，Pass@k 越高**（采样次数增加带来更高的命中概率）。
- 本次实验中 **temperature=0.6 整体最好**。
- 第二张图对应的设置整体更强，提升非常明显。例如在 **k=16**：
  - Temp=0.6：**21.4% → 31.3%**
  - Temp=1.0：**14.2% → 30.5%**
  - Temp=1.2：**6.4% → 约 18%**（以图中标注为准，取近似描述）

