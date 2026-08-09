# 方案2：BLUP 育种值 + 基因型 CNN 预测

## 核心思路

```
训练阶段:
  pheno.csv (60K行, 多环境)
      ↓ MixedLM: y ~ environ(固定) + (1|strain)(随机)
  BLUP[品种] = 每个品种的育种值 (去除了环境主效应)
      ↓
  geno.csv(按品种对齐) + BLUP → CNN训练
      ↓
  模型 (输入=基因型, 输出=育种值)

部署阶段:
  新品种基因型 → CNN → 育种值预测 (无需环境输入)
```

## 文件改动

### 1. 新增 `src/blup.py` — BLUP 提取模块
- `extract_blup(pheno_df, trait, env_col, group_col)` 函数
- 用 statsmodels MixedLM 拟合 `trait ~ C(environ) + (1|strain)`
- 返回: DataFrame[strain, blup, gebv, n_envs], 以及方差分量(遗传力)
- environ 固定效应吸收环境主效应
- strain 随机效应 → BLUP = 品种育种值偏差
- GEBV = 总均值 + BLUP (有实际产量单位)
- 保存结果到 `outputs/blup/blup_{trait}.csv`

### 2. 修改 `configs/config.yaml` — 添加配置
```yaml
blup:
  enabled: true
  env_col: "environ"        # 固定效应列
  group_col: "strain"       # 随机效应列(品种)
  method: "bfgs"            # 优化器
  min_envs_per_strain: 2    # 品种最少环境数(少于则剔除)
  out_dir: outputs/blup
```

### 3. 新增 `train_cnn_blup.py` — 训练脚本
流程:
1. 加载原始 pheno.csv + geno.csv
2. 调用 blup.py 提取 GEBV (全量数据拟合, ~12秒)
3. 按品种对齐基因型与 GEBV
4. 按品种分组拆分 train/val/test (防泄露)
5. 基因型预处理 (imputer + scaler, 全部保存供部署用)
6. build_cnn (单输入, 回归 GEBV)
7. 训练 + ModelCheckpoint + EarlyStopping
8. 评估 R²/RMSE/MAE
9. 保存: 模型 + imputer + scaler + 配置 (用 pickle/joblib)

### 4. 新增 `predict.py` — 部署预测脚本
- 加载模型 + imputer + scaler
- 输入: 基因型 csv (品种×SNP)
- 输出: 每品种预测 GEBV
- 一行命令: `python predict.py --input new_genotypes.csv`

## 关键技术决策

### 为什么用 GEBV 而非 BLUP 偏差
- BLUP = 偏离均值的量 (可为负)
- GEBV = μ + BLUP (绝对产量单位)
- 用 GEBV 训练, 预测值有实际意义 ("预期产量")

### 遗传力 h² 作为质量指标
- h² = σ²_u / (σ²_u + σ²_e/n_per_strain)
- 已验证: 全量数据 h²≈0.75, 合理范围
- 脚本会打印 h², 过低(<0.2)说明数据质量问题

### 评估局限说明
BLUP 是收缩估计, 用BLUP训练+BLUP测试的R²会略偏高。
这是基因组选择MVP的标准做法, 进阶版可做 forward validation
(用部分环境算BLUP, 预测剩余环境)。
```
