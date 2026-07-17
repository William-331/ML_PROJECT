# TP53 基因突变预测 — 基于 METABRIC 乳腺癌 RNA 表达数据

## 项目概述

本项目从零实现机器学习算法（不调用 scikit-learn 等 ML 库），利用 METABRIC 数据库中 **489 个 RNA 基因表达特征** 来预测乳腺癌样本的 **TP53 抑癌基因突变状态**（突变型 / 野生型）。

TP53 是人类癌症中最常发生突变的抑癌基因，传统测序检测成本高、耗时长。本项目的核心动机是探究：**能否仅凭基因表达量，就准确判断 TP53 是否突变？**

## 运行环境

| 依赖 | 用途 |
|---|---|
| **Python 3.8+** | |
| **NumPy** | 所有矩阵运算、线性代数 |
| **Pandas** | CSV 数据读写、DataFrame 预处理 |
| **Matplotlib** | 可视化图表生成 |

安装：

```bash
pip install numpy pandas matplotlib
```

## 项目结构

```
ML_Project/
├── main.py                          # 主入口：加载数据、调度实验、打印结果、调用可视化（~80 行）
├── experiment.py                    # 实验模块：5 组模型训练、正则化路径、学习曲线、RBF 网格搜索
├── preprocessing.py                 # 特征工程：train/test 分割、Z-score 标准化、PCA 降维
├── evaluation.py                    # 评估模块：Accuracy/Precision/Recall/F1、KFold、交叉验证
├── visualization.py                 # 可视化模块：9 张分析图表
├── dataset/
│   ├── data_loader.py               # 数据 I/O：加载 METABRIC CSV、特征提取、标签二值化
│   └── METABRIC_RNA_Mutation.csv    # 数据集（需自行下载放入）
├── models/
│   ├── LogesticRegression.py        # 逻辑回归（L1/L2 正则化，从零实现）
│   └── SVM.py                       # 支持向量机（Linear / RBF 核，从零实现）
└── plots/                           # 输出图表（运行后自动生成）
    └── (9 张 PNG)
```

## 快速开始

### 1. 准备数据

将 `METABRIC_RNA_Mutation.csv` 放入 `dataset/` 目录。

数据格式要求：前 31 列为样本元数据（含 `tp53_mut` 列），第 32 列起为 RNA 表达量（以 `_mut` 结尾的列将被视为突变数据列自动排除）。

### 2. 运行实验

```bash
cd ML_Project
python main.py
```

程序将依次执行 5 组对比实验、9 组正则化路径、3 条学习曲线，最终在 `plots/` 目录输出所有图表。

**预计运行时间**：10–15 分钟（RBF SVM 网格搜索为 20 组 3-fold CV，耗时最久）。

## 代码架构

```
main.py           → 数据加载 + 流程调度（只负责"调谁"，不负责"怎么干"）
experiment.py     → 5 组实验训练 + 正则化路径 + 学习曲线（全部实验逻辑）
models/*.py       → 从零实现的模型类（fit / predict）
preprocessing.py  → 特征工程工具（分割、标准化、PCA）
evaluation.py     → 评估工具（指标、KFold、交叉验证）
visualization.py  → 所有 matplotlib 图表生成
```

分工原则：**main 不知道怎么训练，experiment 不知道数据从哪来**。

## 数据预处理

| 步骤 | 方法 | 实现位置 |
|---|---|---|
| **特征提取** | 提取第 31 列之后的 RNA 表达列，排除 `_mut` 突变列 | `dataset/data_loader.py` |
| **缺失值处理** | 中位数填充（每列独立） | `dataset/data_loader.py` |
| **标签二值化** | `0/NaN/'0'` → 0（野生型），其他 → 1（突变型） | `dataset/data_loader.py` |
| **训练/测试分割** | 80/20 随机分割，固定随机种子可复现 | `preprocessing.py` |
| **特征标准化** | Z-score：`(x - μ) / σ`，测试集使用训练集的 μ 和 σ 防止泄露 | `preprocessing.py` |
| **PCA 降维** | SVD 分解，保留 95% 方差的主成分 | `preprocessing.py` |

## 模型实现

所有模型仅使用 NumPy 从零实现，**不依赖 sklearn / TensorFlow / PyTorch**。

### 逻辑回归（Logistic Regression）

| 特性 | 说明 |
|---|---|
| **模型类型** | 线性二分类器 |
| **激活函数** | Sigmoid：`σ(z) = 1 / (1 + e^(-z))` |
| **损失函数** | 二元交叉熵 + 正则化项 |
| **优化方法** | 批量梯度下降 |
| **L2 正则化** | 梯度项增加 `(λ/m) * w`，将所有权重等比收缩 |
| **L1 正则化** | Proximal Gradient Descent：梯度更新后执行软阈值 `sign(w) * max(0,∣w∣−λ·lr)`，将不重要权重精确置零 |

```python
# 使用示例
lr = LogisticRegression(learning_rate=0.1, num_iterations=5000,
                        lambda_param=0.01, penalty='l1')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
top_indices, top_weights = lr.get_important_features()  # L1 选出的特征
```

### 支持向量机（SVM）

| 特性 | 说明 |
|---|---|
| **模型类型** | 最大间隔二分类器 |
| **目标函数** | Hinge Loss + L2 正则化 |
| **优化方法** | SGD（默认）或 Adam（自适应学习率） |
| **类别不平衡** | `class_weight='balanced'`：按类别反比自动加权 |
| **Linear 核** | 原始空间子梯度下降，权重除以 `√n_features` 稳定梯度 |
| **RBF 核** | 对偶空间 Representer Theorem：`K(x,z) = exp(-γ‖x-z‖²)`，在 `α` 系数上优化（预计算 m×m 核矩阵） |

```python
# 线性 SVM
svm_lin = SVM(learning_rate=0.001, num_iterations=5000,
              lambda_param=0.0001, kernel='linear',
              class_weight='balanced')

# RBF SVM（带超参调优）
svm_rbf = SVM(learning_rate=0.01, num_iterations=3000,
              lambda_param=1e-5, kernel='rbf_exact',
              gamma=0.001, class_weight='balanced',
              optimizer='adam')
```

## 模型评估

### 评估指标

所有指标从零实现（`calculate_metrics()`）：

| 指标 | 公式 | 说明 |
|---|---|---|
| **Accuracy** | `(TP + TN) / Total` | 整体正确率 |
| **Precision** | `TP / (TP + FP)` | 预测为突变中真正突变的比例 |
| **Recall** | `TP / (TP + FN)` | 真实突变中被正确识别的比例 |
| **F1 Score** | `2 × P × R / (P + R)` | Precision 和 Recall 的调和平均值 |

### 5-Fold 交叉验证

`KFold` 类从零实现：

- 将训练集均分成 5 折
- 每折轮流作为验证集，其余 4 折作为训练集
- 每折重新初始化模型、重新训练
- 最终输出 **mean ± std** 的 4 项指标

### RBF SVM 超参调优

在**训练集内部**进行 3-fold CV 网格搜索，选择最佳 `gamma` 和 `lambda` 组合：

- `gamma` ∈ `{0.0005, 0.001, 0.002, 0.005, 0.01}`
- `lambda` ∈ `{1e-6, 1e-5, 5e-5, 1e-4}`

最后在完整训练集上用最佳参数重新训练，在测试集上评估。

## 实验设计

### 5 组对比实验

| # | 模型 | 正则化 | 核心目的 |
|---|---|---|---|
| 1 | Logistic Regression + L2 | Ridge (λ=0.01) | 基准线性模型 |
| 2 | Logistic Regression + L1 | Lasso (λ=0.01) | 自动特征选择 + 可解释性 |
| 3 | PCA (95% var) + LR L2 | 降维后 Ridge | 维数约简对性能的影响 |
| 4 | SVM Linear | L2 正则化 + 类别平衡 | 线性最大间隔分类器 |
| 5 | SVM RBF | L2 正则化 + CV 网格搜索 | 捕捉非线性决策边界 |

### 扩展分析

- **正则化路径**：在 9 个 λ 值（1e-4 ~ 5.0）上分别评估 L1/L2 的 5-fold CV 性能
- **学习曲线**：在 5 种训练集大小（10% ~ 100%）上评估训练/验证准确率，分析偏差-方差权衡

## 分析图表说明

| 图表 | 内容 | 分析意义 |
|---|---|---|
| `01_metrics_comparison.png` | 5 个模型的 Hold-out + CV 对比柱状图 | 直观对比模型性能和泛化差异 |
| `02_cost_curves.png` | LR 和 SVM 各自的损失函数下降曲线 | 判断收敛速度和平稳性 |
| `03_pca_variance.png` | 主成分数量 vs 累计方差解释率 | 数据内在维度的可视化 |
| `04_l1_feature_weights.png` | L1 选出的 Top 30 重要基因及其权重 | 生物可解释性：哪些基因与 TP53 突变最相关 |
| `learning_curve_*.png` (×3) | 训练量 vs 训练/CV 准确率 | 判断过拟合/欠拟合：gap 越大 → 过拟合越严重 |
| `regularization_path_*.png` (×2) | λ vs CV 准确率 | 最优正则化强度，模型对 λ 的敏感度 |

## 实验结论要点

从 `main.py` 运行的输出可以给出以下关键发现：

1. **SVM RBF > SVM Linear**：验证了 TP53 突变在 RNA 表达空间中存在**非线性决策边界**，RBF 核能捕捉到这种结构
2. **SVM RBF > LR + L1**：非线性 SVM 优于稀疏线性模型，说明数据中的模式不能被几个关键基因完全解释
3. **LR + L1 在极低特征维度下表现优异**：仅 106/489 个基因就达到接近 SVM RBF 的准确率，说明存在少數表達差異異常顯著的關鍵基因
4. **PCA 降维未带来性能增益**：312 个主成分才能保留 95% 方差，说明数据方差分布较为分散

## 模块 API 参考

### `utils.py`

| 函数 | 参数 | 返回值 |
|---|---|---|
| `KFold(n_splits, shuffle, random_state)` | 折数、是否打乱、随机种子 | 迭代器：`(train_idx, val_idx)` |
| `cross_validate(model_cls, params, X, y, cv)` | 模型类、参数字典、数据、折数 | `{metric: (mean, std)}` |
| `calculate_metrics(y_true, y_pred)` | 真实/预测标签 | `(acc, prec, rec, f1)` |
| `PCA(n_components, variance_ratio)` | 主成分数(可选)、方差保留率(可选) | PCA 实例 |

### `visualization.py`

| 函数 | 输出文件 |
|---|---|
| `plot_metrics_comparison(results, cv_results)` | `01_metrics_comparison.png` |
| `plot_cost_curves(cost_data)` | `02_cost_curves.png` |
| `plot_pca_variance(explained_variance_ratio_)` | `03_pca_variance.png` |
| `plot_l1_feature_weights(indices, weights, n_features)` | `04_l1_feature_weights.png` |
| `plot_learning_curve(train_sizes, train_scores, val_scores, name)` | `learning_curve_{name}.png` |
| `plot_regularization_path(lambda_values, cv_scores, penalty)` | `regularization_path_{penalty}.png` |

## 定制与扩展

### 添加新模型

1. 在 `models/` 目录创建新文件，实现 `fit(X, y)` 和 `predict(X)` 方法
2. 在 `main.py` 中导入并在 `run_experiments()` 中添加实验
3. 复用 `cross_validate()` 做 5-fold CV

### 修改超参

直接在 `main.py` 的实验中修改 `learning_rate`、`lambda_param`、`num_iterations` 等参数。

### 切换数据集

在 `Data_Preprocessing.py` 中修改 `load_tp53_data` 或新增数据加载函数，确保输出格式为 `(X: ndarray, y: ndarray)`。


## License

For educational use.
