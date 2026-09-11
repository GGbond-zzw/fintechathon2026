# 2026 深圳国际金融科技大赛——数据分析赛道题五

> 基于原始量价数据的 A 股下一交易日收益预测项目计划

本文件是项目的总入口与实施路线图，可直接保存为：

~~~text
D:\codex\fintechathon2026\README.md
~~~

本文把赛题附件、官方 evaluate.py 和测试集已经确认的内容视为硬约束；模型、特征、参数、目录与实施顺序均属于工程建议，需要以后续离线验证结果为准。

## 1. 项目目标

使用赛题提供的全市场 A 股日线量价数据，为测试集中的每个 ts_code + trade_date 生成下一交易日收益率 y_ret_1d 的预测值 pred。

最终目标不是单独最小化回归误差，而是在严格无未来数据泄漏的前提下最大化官方综合评分：

1. 提高逐日横截面排序能力；
2. 提高预测排名前 10% 股票组合的超额收益；
3. 控制相邻交易日 Top 10% 股票集合的换手率。

最终提交文件必须只有以下三列：

~~~text
ts_code,trade_date,pred
~~~

## 2. 官方评分公式与精确口径

综合得分为：

\[
Score=0.4\times RankIC+0.3\times AnnualExcess+0.3\times(1-Turnover)
\]

### 2.1 Rank IC：40%

对每个交易日，在 y_ret_1d 非空的股票中计算 pred 与 y_ret_1d 的 Spearman 相关系数，再对有效交易日取均值：

\[
RankIC=Mean_t(Spearman(pred_t,y_t))
\]

官方代码口径：

- 每日 y_ret_1d 非空样本少于 30 时跳过该日；
- Rank IC 阶段不剔除涨停股票；
- 同时报告 ic_std、icir 和 ic_positive_ratio，但综合评分只使用 ic_mean；
- 提交前必须保证 pred 全部为有限数，避免相关系数产生 NaN。

### 2.2 Top 组年化超额收益：30%

每天先剔除涨停股票和标签缺失样本，按 pred 降序取前：

\[
n_{top}=\max(\lfloor n/10\rfloor,1)
\]

只股票。当天超额收益为 Top 组平均收益减去同一有效股票池的平均收益：

\[
Excess_t=Mean(y_{top,t})-Mean(y_{valid,t})
\]

\[
AnnualExcess=Mean_t(Excess_t)\times252
\]

官方代码口径：

- 有效股票是 flag_limit_up == 0 且 y_ret_1d 非空；
- 每日有效股票少于 100 时跳过该日；
- 市场基准为同一有效股票池的等权平均收益；
- 同时报告 top1_annual_ret，但综合评分使用 annual_excess；
- 官方代码未对年化超额收益做裁剪。

### 2.3 预测换手率：30%

每天剔除涨停股票后，按 pred 降序取前 10% 得到集合 \(S_t\)。相邻有效日的换手率为 Jaccard 距离：

\[
Turnover_t=1-\frac{|S_t\cap S_{t-1}|}{|S_t\cup S_{t-1}|}
\]

官方代码口径：

- 换手率阶段剔除涨停股票，但不依赖 y_ret_1d；
- 每日有效股票少于 100 时跳过该日，并清空上一日集合；
- Top 数量仍为 max(len(valid) // 10, 1)；
- 综合评分使用 1 - mean_turnover。

> 关键结论：三个子指标的有效股票池不同。本地 evaluator 必须逐行复刻官方 evaluate.py，不得为了“统一口径”擅自改变过滤条件、阈值或取整方式。

## 3. 数据说明

### 3.1 已确认范围

- 训练集：2018-01-02 至 2024-12-31；
- 测试集：2025-01-02 至 2026-06-08；
- 已确认资料显示测试集覆盖 4650 只股票；
- 训练集包含 y_ret_1d，约 14.5% 标签缺失；
- OHLC 为后复权价格；
- 训练集与测试集的特征结构相近，训练集额外包含标签。

训练集未上传到当前开发对话，但用户本地持有。程序必须从约定路径读取真实训练集，不得要求把大文件嵌入代码或提交到 Git。

### 3.2 测试集已确认字段

| 字段 | 用途 |
|---|---|
| ts_code | 股票代码，主键之一 |
| trade_date | 交易日期，主键之一 |
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 收盘价 |
| vol | 成交量 |
| amount | 成交额 |
| flag_limit_up | 涨停标记 |
| flag_limit_down | 跌停标记 |

正式开发时必须由数据检查脚本确认训练集字段，不能只依赖预计结构。

### 3.3 数据使用方式

- 测试集可用于读取、清洗、特征管线、内存优化和 submission 生成测试；
- 测试集没有真实标签，不能切分测试集训练收益模型，也不能伪造标签；
- 正式训练使用本地训练集中 y_ret_1d 非空的记录；
- 测试期 rolling 特征应拼接训练集末尾历史作为 warm-up，再只保留测试期记录；
- data/raw 只读，转换结果写入 data/processed 或 data/cache。

## 4. 项目原则

1. **官方脚本优先。** evaluate.py 是评分计算的唯一实现依据。
2. **严格防泄漏。** 日期 t 的特征只能使用 t 日及以前可用的赛题字段。
3. **禁止随机切分。** 使用 Walk-forward / Expanding-window CV。
4. **测试集不训练。** 不构造伪标签，不用测试结果选择有监督模型。
5. **主键不可破坏。** 所有中间表保持 (ts_code, trade_date) 唯一，合并前后校验。
6. **缺失标签不填 0。** baseline 直接排除标签缺失样本。
7. **优化综合评分。** RMSE、MAE 只作诊断，选模以跨折官方 final_score 为主。
8. **实验可复现。** 固定种子，保存配置、代码版本、特征版本、OOF 和逐折结果。
9. **不修改原始文件。** 缓存、降精度和清洗均写入新目录。
10. **仅使用允许字段。** 不接入外部行情、基本面、指数成分或其他未授权数据。
11. **先可信 baseline，再扩展。** 深度学习和大规模自动因子不进入第一阶段。
12. **报告同步。** 每阶段保留可直接进入比赛报告的表格、图和结论。

## 5. 建议目录结构

~~~text
D:\codex\fintechathon2026\
├── README.md
├── ARCHITECTURE.md
├── FRAMEWORK_REVIEW.md
├── AGENTS.md
├── config.yaml
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
├── official\
│   └── evaluate.py         # 官方原始副本，按 SHA-256 校验
├── data\
│   ├── raw\                 # 训练集.csv、测试集_X.csv；只读
│   ├── interim\
│   ├── processed\
│   └── cache\
├── src\
│   ├── data\                # loader、validator、preprocessing、memory
│   ├── features\            # price、momentum、volatility、volume、liquidity、
│   │                        # technical、market、cross_sectional、pipeline
│   ├── models\              # ridge、lightgbm、catboost、ranking、ensemble
│   ├── validation\          # splitter、evaluator、diagnostics
│   ├── postprocess\         # rank_transform、smoothing、turnover_control
│   └── utils\               # logger、paths、seed
├── scripts\
│   ├── 01_check_data.py
│   ├── 02_audit_labels.py
│   ├── 03_build_features.py
│   ├── repartition_feature_store.py
│   ├── 04_build_cross_sectional.py
│   ├── 05_build_test_features.py
│   ├── 06_validate.py
│   ├── 07_check_evaluator.py
│   └── 08_train_baseline.py  # 后续脚本继续按 Phase 编号新增
├── experiments\
│   ├── experiment_registry.csv
│   └── configs\
├── models\
├── submissions\
├── reports\
│   ├── figures\
│   ├── tables\
│   └── competition_report\
├── notebooks\
└── tests\
~~~

notebooks 只用于探索；正式数据、特征、训练、评分和提交逻辑必须进入 src 与 scripts。

## 6. Phase 0–19 实施计划

### Phase 0：环境与工程初始化

**任务：** 创建目录、虚拟环境、requirements.txt、config.yaml、.gitignore、统一日志、随机种子和路径管理。建议依赖 numpy、pandas、pyarrow、polars、scipy、scikit-learn、lightgbm、catboost、matplotlib、joblib、pyyaml、loguru、tqdm。暂不引入深度学习框架。

**验收：** scripts/01_check_data.py 能读取测试集并输出 shape、字段、类型、日期范围、股票数、重复键和缺失率。

### Phase 1：数据读取与完整性校验

**任务：** 实现 load_train()、load_test()、load_train_tail()；明确 dtype；评估 float32 精度影响；建议缓存 Parquet；检查主键、排序、每日股票数、缺失、Inf、OHLC 关系、负成交量/成交额和涨跌停标记。

**验收：** 生成机器可读的数据审计结果；缓存前后主键、行数和关键统计一致。

### Phase 2：标签与时间对齐审计

**任务：** 分析 y_ret_1d 缺失、分位数、极值及逐年/逐日分布；抽样用下一条有效记录核对 close[t+1] / close[t] - 1。shift(-1) 只允许用于标签审计，严禁进入特征。标签截尾、横截面 rank 化或稳健损失均作为实验，不预设有效。

**验收：** 标签对齐报告和自动测试完成；任何差异先解释后建模。

### Phase 3：特征工程 V1

**任务：** 构建约 100–300 个可解释的收益、动量、日内结构、趋势、波动、量能、价量、流动性、价格位置和市场环境特征。

**验收：** rolling 显式按 ts_code 分组；输出特征字典、缺失率、有限值比例和抽样手算结果。

### Phase 4：横截面特征

**任务：** 对核心因子保留 raw、逐日 percentile rank 和逐日 z-score；显式按 trade_date 分组；确定性处理零方差、同值和小截面。

**验收：** 日期间不混排；对照实验显示其对各折指标的真实增量。

### Phase 5：测试集 warm-up 与缓存

建议流程：

~~~text
训练集最后 warmup_days 个交易日 + 完整测试集
→ 按股票、日期排序
→ 统一计算时序特征
→ 计算每日市场与横截面特征
→ 只保留测试日期
~~~

初始建议 warmup_days = 250；代码还要根据最大 rolling 窗口检查历史是否足够。

**验收：** 测试首日的中短周期特征不因截断而全部缺失；最终主键与原测试集一致。

### Phase 6：Walk-forward CV

初始建议：

| Fold | 训练期 | 验证期 |
|---|---|---|
| 1 | 2018-01-02 至 2021-12-31 | 2022 年 |
| 2 | 2018-01-02 至 2022-12-31 | 2023 年 |
| 3 | 2018-01-02 至 2023-12-31 | 2024 年 |

报告每折指标、等权汇总、标准差和最差折。2024 可作重点诊断，但提高其权重必须预先规定，避免看结果后任意调整。预处理器只在训练折拟合。

### Phase 7：复刻官方 evaluator

将官方逻辑封装到 src/validation/evaluator.py，输出：

~~~text
ic_mean, ic_std, icir, ic_positive_ratio,
annual_excess, top1_annual_ret, mean_turnover, final_score
~~~

**验收：** 同一输入下与官方脚本在浮点容差内一致；覆盖 30/100 阈值、涨停过滤、缺失标签、Top 取整和 Jaccard 换手的测试。

### Phase 8：Baseline

依次完成：

1. 简单因子 baseline：打通全流程；
2. Ridge：检查线性信号与缩放；
3. LightGBM 回归：第一阶段主 baseline；
4. CatBoost 回归：资源允许时增加多样性。

可用回归损失训练，但模型选择同时看完整官方分数。每个 baseline 保存 OOF 预测并可独立重跑。

### Phase 9：实验管理

experiments/experiment_registry.csv 至少记录：

~~~text
experiment_id, timestamp, git_commit, config_path, feature_version,
model, train_period, val_period, seed,
ic_mean, ic_std, icir, ic_positive_ratio,
annual_excess, top1_annual_ret, mean_turnover, final_score,
artifact_paths, notes
~~~

不依赖聊天记录记实验结果。

### Phase 10：Turnover 优化

建议实验：

- 每日预测先做横截面 rank，再做因果指数平滑；
- 搜索 \(p_t^*=\alpha p_t+(1-\alpha)p_{t-1}^*\)，例如 alpha 从 1.0 至 0.5；
- 研究 entry/exit hysteresis，例如新股票进入更高分位、已有股票跌出较低分位才退出；
- 将稳定化集合映射回连续、确定性的 pred 排序；
- 每个参数在各 CV 折重新计算完整官方分数。

目标是最高 Final Score，不是最低 Turnover。不得用未来日预测平滑当前日。

### Phase 11：Ensemble

建议将各模型逐日 rank 标准化后加权：

\[
Pred=w_1Pred_{LGBM}+w_2Pred_{CatBoost}+w_3Pred_{Ridge},\quad\sum w_i=1
\]

权重按 OOF 官方分数选择，先粗网格后细化，并与等权平均比较。测试预测前锁定规则。

### Phase 12：Ranking 模型

在回归 baseline 稳定后研究 LightGBM Ranker / LambdaRank，以 trade_date 为 group。连续收益转 relevance label 的规则必须明确、可复现，group 必须连续且 size 正确。

Ranking 是实验方向，并不因评分含 Rank IC 就必然优于回归。

### Phase 13：高级特征

建议候选：RSI、MACD、ATR、Bollinger 位置、OBV、MFI、VWAP 近似、趋势强度、反转、动量×反转、放量突破、波动突破。全部只能由赛题字段构造，并记录公式、窗口、最小样本数和缺失策略。

### Phase 14：特征筛选

统计单因子逐日 Rank IC、IC 标准差、ICIR、IC 正比例、逐年表现、覆盖率、极端行情表现和计算成本；按特征组做 ablation。不能只看树模型 feature importance。

### Phase 15：Regime 与近期样本

优先加入由赛题字段构造的市场状态特征，如市场等权收益、横截面波动、上涨比例和成交额状态。有证据后再考虑独立 regime 模型。可比较全历史、近四年、近两年训练窗，但不预设近期模型一定更好。

### Phase 16：最终训练

锁定特征、模型、参数、ensemble 和后处理。候选训练窗可包括 2018–2024、2021–2024、2023–2024，是否组合由 Walk-forward 结果决定。训练前冻结配置，日志记录数据文件元信息、样本数、特征数、种子、参数、耗时和产物路径。

### Phase 17：测试集预测

~~~text
训练集尾部 warm-up + 测试集
→ 时序特征
→ 市场/横截面特征
→ 仅保留测试期
→ 基模型预测
→ ensemble
→ 因果时间平滑
→ turnover-aware 排序
→ final pred
~~~

**验收：** 训练/测试特征名、顺序和 dtype 一致；测试主键一个不少、一个不多。

### Phase 18：Submission 自动检查

必须检查：

- 列名和顺序严格为 ts_code, trade_date, pred；
- 行数与测试集一致；
- 主键唯一且集合与测试集完全相同；
- pred 为数值，无 NaN、Inf；
- 股票代码字符串和日期格式正确；
- 与测试集 inner merge 后行数不变；
- 输出 CSV 可重新读取并再次通过校验。

建议命名 submissions/submission_<experiment_id>.csv，不静默覆盖历史文件。

### Phase 19：比赛报告与最终复现

官方要求数据分析报告正文不超过 8 页，并包含：

1. 数据介绍；
2. 描述性分析；
3. 模型分析；
4. 应用验证；
5. 产品思路；
6. 总结结论。

建议持续保存标签/收益分布、涨跌停比例、市场波动、因子 IC、逐折模型 IC、分组收益、累计 Top 超额收益、换手率、特征重要性、消融和实验对比表。封面、目录或附录是否计入页数，以官方最终规则为准，不自行假设。

最终交付包括源码、锁定配置、依赖清单、运行说明、实验登记、最终 submission、报告图表和端到端复现命令。

## 7. 特征工程建议

### 7.1 收益、动量与价格结构

建议收益窗口：[1, 2, 3, 5, 10, 20, 40, 60, 120]。

\[
ret_n=close_t/close_{t-n}-1
\]

建议构造：

~~~text
intraday_return = close / open - 1
overnight_gap = open / previous_close - 1
high_low_range = (high - low) / close
upper_shadow
lower_shadow
close_position_in_bar
close / MA_n(close) - 1
MA_short / MA_long - 1
~~~

前收盘只能按 ts_code 取上一条记录；所有除法使用安全分母。

### 7.2 波动、量能与流动性

~~~text
std_ret_{5,10,20,60}
downside_vol, upside_vol
rolling_skew, rolling_kurtosis
vol_5 / vol_20, vol_10 / vol_60
volume_change, volume_ma, volume_ratio
amount_change, amount_ma, amount_ratio
rolling_corr(return, volume/amount)
price_volume_divergence
amihud_illiq = abs(return) / amount
~~~

对零成交、极小分母和相关系数最小样本数做显式处理。

### 7.3 价格位置与市场环境

\[
position_n=\frac{close-\min(low,n)}{\max(high,n)-\min(low,n)}
\]

建议窗口：[5, 10, 20, 60, 120]，并增加 distance_to_high、distance_to_low、breakout_high/low。

每日市场特征建议包括 market_mean_ret、market_median_ret、market_std_ret、market_up_ratio、market_limit_up/down_ratio、market_amount 和个股相对市场收益。这些使用当天完整横截面，但不使用未来日期。

### 7.4 横截面变换

对选定因子逐日生成 raw、percentile rank 和 z-score。可实验逐日 winsorization，但处理顺序和阈值必须记录，保留消融对照。

## 8. 模型比较与换手后处理边界

每个模型至少报告各折官方八项指标、均值/标准差/最差折、每日 IC 序列、Top 组累计超额、换手分布、资源消耗和相对上一个 baseline 的消融差异。不得用单个年份、单个种子或单一 RMSE 宣称提升。

后处理最终必须体现在每日连续 pred 的排序中，并遵守：

- 当前日结果只能依赖当前及过去预测；
- 缺记录、停牌或涨停过滤使用确定性的状态规则；
- 同分值使用稳定 tie-break；
- 参数只用 OOF 验证选择；
- 降低换手时同步复算 IC 与超额；
- 不使用测试标签、未来预测或全测试期信息反推当前排序。

## 9. config.yaml 示例

以下是建议起点，不代表最优参数。Windows 路径使用正斜杠，避免 YAML 转义问题。

~~~yaml
project:
  name: fintechathon2026_task5
  seed: 42

paths:
  root: D:/codex/fintechathon2026
  train: D:/codex/fintechathon2026/data/raw/训练集.csv
  test: D:/codex/fintechathon2026/data/raw/测试集_X.csv
  processed: D:/codex/fintechathon2026/data/processed
  cache: D:/codex/fintechathon2026/data/cache
  models: D:/codex/fintechathon2026/models
  experiments: D:/codex/fintechathon2026/experiments
  submissions: D:/codex/fintechathon2026/submissions
  reports: D:/codex/fintechathon2026/reports

data:
  label: y_ret_1d
  id_cols: [ts_code, trade_date]
  feature_cols:
    - open
    - high
    - low
    - close
    - vol
    - amount
    - flag_limit_up
    - flag_limit_down
  warmup_days: 250
  parquet_cache: true
  float_dtype: float32

cv:
  type: expanding_window
  folds:
    - {name: fold_2022, train_start: 20180102, train_end: 20211231,
       val_start: 20220101, val_end: 20221231}
    - {name: fold_2023, train_start: 20180102, train_end: 20221231,
       val_start: 20230101, val_end: 20231231}
    - {name: fold_2024, train_start: 20180102, train_end: 20231231,
       val_start: 20240101, val_end: 20241231}

evaluator:
  ic_min_samples: 30
  portfolio_min_samples: 100
  top_fraction_denominator: 10
  annualization_days: 252
  weights: {rank_ic: 0.4, annual_excess: 0.3, one_minus_turnover: 0.3}

features:
  return_windows: [1, 2, 3, 5, 10, 20, 40, 60, 120]
  volatility_windows: [5, 10, 20, 60]
  volume_windows: [5, 10, 20, 60]
  correlation_windows: [5, 10, 20, 60]
  position_windows: [5, 10, 20, 60, 120]
  add_cross_sectional_rank: true
  add_cross_sectional_zscore: true

model:
  name: lightgbm_regressor
  params:
    objective: regression
    learning_rate: 0.03
    num_leaves: 31
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 1
    random_state: 42
    n_jobs: -1

postprocess:
  rank_before_blend: true
  smoothing_alpha_candidates: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
  entry_percentile_candidates: [0.08, 0.09, 0.10]
  exit_percentile_candidates: [0.10, 0.11, 0.12]

submission:
  columns: [ts_code, trade_date, pred]
  sort_by: [trade_date, ts_code]
  allow_nan: false
  allow_inf: false
  overwrite: false
~~~

evaluator 中的常量用于显式记录官方口径，不应成为随意调参项。model.params 与 postprocess 的数值仅是搜索起点。

## 10. AGENTS.md 建议

建议在根目录创建简短、可执行的 AGENTS.md。官方 OpenAI 文档说明，此类指令会影响 Codex 的执行，因此应定期检查冲突和过时规则：[OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model)。

建议内容：

~~~markdown
# Project instructions

- 用户的明确指令优先于本文件中的工程建议。
- 先阅读 README.md 和 config.yaml，再修改代码。
- 不修改 data/raw，不把原始数据、缓存、模型或 submission 提交到 Git。
- 所有路径从 config.yaml 读取，不在源码硬编码本机绝对路径。
- 时序 rolling/shift 必须按 ts_code 分组并按 trade_date 排序。
- 横截面 rank/z-score 必须按 trade_date 分组。
- 禁止未来数据泄漏、随机拆分和用测试集构造训练标签。
- 官方 evaluate.py 的公式、过滤条件、阈值和取整逻辑不得改写。
- 正式逻辑写入 src 和 scripts；notebooks 仅用于探索。
- 关键数据边界与 evaluator 逻辑必须有测试。
- 每次实验保存配置、种子、特征版本、逐折指标和产物路径。
- 修改后运行与改动范围相称的检查；失败时如实报告。
- 保留用户已有改动，不静默删除历史实验或 submission。
- 每完成一个 Phase，更新 README 的状态、验收证据和下一步。
~~~

## 11. 阶段状态表

由 Codex 持续更新。只有具备验收证据时才标记“已完成”。

| Phase | 状态 | 主要产物/验收 |
|---:|---|---|
| 0 | 已完成 | 骨架、配置、D 盘虚拟环境；测试集基础检查通过（见下方验收记录） |
| 1 | 已完成 | loader、validator、Parquet 缓存；训练集与测试集全量审计通过 |
| 2 | 已完成 | 标签分布、逐年/逐日统计、跨训练/测试边界的时间对齐审计通过 |
| 3 | 已完成 | 158 个因果 V1 特征；全量 Parquet、特征字典、覆盖率与手算检查通过 |
| 4 | 已完成 | 36 个预声明核心因子的 72 个逐日 rank/z-score；按年输出并通过分组审计 |
| 5 | 已完成 | 250 日 warm-up；测试侧三套特征缓存、首日与主键检查通过 |
| 6 | 已完成 | expanding-window splitter；真实交易日 purge、精确日历与边界测试通过 |
| 7 | 已完成 | 本地 evaluator；八项指标与官方脚本零差异，阈值/过滤/换手边界通过 |
| 8 | 已完成 | 简单因子、流式 Ridge、全量 LightGBM；三折 OOF、官方指标与框架复查完成 |
| 9 | 已完成 | 追加式实验注册表；配置、源码、特征与产物指纹完整，幂等复验通过 |
| 10 | 已完成 | 因果 rank/EMA 与换仓缓冲；18 组完整 OOF 权衡曲线及实验登记 |
| 11 | 已完成 | LightGBM/Ridge 逐日 rank 融合；21 组权重、固定换手后处理及跨折比较 |
| 12 | 已完成 | 全量 LambdaRank；十档 relevance、逐日 group、公平对照及阶段性框架复查 |
| 13 | 已完成 | 32 个因果高级特征；训练/测试按年 Parquet、250 日预热、字典与审计通过 |
| 14 | 已完成 | 2018–2021 固定预筛选、2022–2024 留出验证；逐日 IC、极端行情、冗余与家族消融 |
| 15 | 已完成 | ATR 增量 LambdaRank；全历史/近四年/近两年三折窗口与固定阈值 regime 稳定性 |
| 16 | 待开始 | 最终模型；冻结配置 |
| 17 | 待开始 | 测试预测；主键一致 |
| 18 | 待开始 | submission；全检查通过 |
| 19 | 待开始 | 8 页内报告；端到端复跑 |

状态统一使用：待开始、进行中、已完成、阻塞。

框架复查节奏：Phase 3 后已完成首次复查；后续每完成 4 个 Phase 复查一次，计划检查点为 Phase 8、12、16 完成后。复查不提前实施下一 Phase。

### Phase 0 验收记录（2026-09-09）

- 项目根目录：`D:\codex\fintechathon2026`；虚拟环境：`.venv`。
- `scripts/01_check_data.py` 已对 `data/raw/测试集_X.csv` 完成只读流式检查。
- 实测 shape 为 1,599,600 × 10，日期范围为 2025-01-02 至 2026-06-08，共 4,650 只股票，重复主键行数为 0。
- 机器可读结果写入 `reports/tables/phase0_test_summary.json`，运行日志写入 `logs/project.log`。
- 核心依赖 NumPy 2.2.6 与 PyYAML 6.0.3 导入通过；Phase 0 单元测试通过。当前环境以 Pandas + PyArrow 处理 Parquet；因本机 Polars 二进制导入无响应，未将其保留为项目依赖。
- 目标目录尚无 `训练集.csv`，训练相关接口与完整数据校验按计划在 Phase 1 实现，不使用测试集构造标签。

### Phase 1 验收记录（2026-09-09）

- 已实现 `load_train()`、`load_test()`、`load_train_tail()`；训练读取严格要求真实 `训练集.csv`，不会回退到测试集。
- 测试集实测 1,599,600 × 10，覆盖 344 个交易日和 4,650 只股票；每日均为 4,650 行。
- 主键无空值且唯一，数据按 `ts_code, trade_date` 排序；Inf、OHLC 关系、负成交量/成交额及非法涨跌停标记均为 0。
- float32 在前 100,000 行上的最大相对误差不超过 `5.96e-08`；实际 dtype 已显式固定。
- 已生成约 39.6 MB 的 Zstandard Parquet 测试缓存；缓存与 CSV 的行数、字段、主键哈希、缺失统计及数值极值一致。
- 原始测试 CSV 审计前后 SHA-256 均为 `DCEFFBF68BBAB07A254BC6CAA4D8A1E33DE8DE9C15489D936C5D15020DBFA90F`。
- 7 个 loader/validator/缓存边界测试全部通过。完整报告位于 `reports/tables/phase1_data_audit.json`。
- 真实训练集已完成全量复验：7,900,350 × 11，覆盖 2018-01-02 至 2024-12-31 的 1,699 个交易日及 4,650 只股票；每日均为 4,650 行。
- 训练集标签缺失 1,141,819 行（14.4528%）；主键、排序、Inf、OHLC、成交量/额和涨跌停标记检查均通过。
- 训练 float32 抽样最大相对误差不超过 `5.96e-08`；约 191.4 MB 的 Zstandard Parquet 缓存一致性检查通过。
- 训练集源文件与项目副本 SHA-256 均为 `E9D23E87F7E1F9439F28F7DD951D129EA49EAACDA5AD650DA8576A830A6DEAC9`。
- `load_train_tail(..., n_trade_dates=2)` 已用真实缓存验证，返回 2024-12-30 与 2024-12-31 共 9,300 行且保留真实标签列。

### Phase 2 验收记录（2026-09-09）

- 真实标签共 6,758,531 个，缺失 1,141,819 个（14.4528%）；均值 `0.0003245`、标准差 `0.0298699`、最小值 `-0.6447963`、最大值 `2.5742496`。
- 1%/50%/99% 分位数分别为 `-0.0834822`、`0`、`0.1`；绝对收益不小于 10%/20%/50% 的记录分别为 90,916/2,916/11 条。
- 已保存 2018–2024 逐年统计及 1,699 个交易日的逐日统计；标签缺失率从 2018 年的 26.23% 降至 2024 年的 2.78%。
- 标签公式经训练/测试价格时间轴跨边界核验：6,758,531 个非空标签全部等于同一股票下一交易日行的 `close[t+1] / close[t] - 1`，最大绝对误差 `7.56e-16`，缺失掩码一致率 100%。
- “跳过缺失 close 取下一条非空记录”的候选定义会错误地产生 1,743 个标签，已排除；正确规则是不跨缺失行跳跃。
- 测试集只提供价格以核验 2024-12-31 边界，未创建或推断任何测试标签；负向 shift 仅存在于 `src/data/label_audit.py`，未输出特征。
- 训练 CSV 审计前后 SHA-256 一致，未修改或填充原始标签；11 项测试全部通过。
- 产物：`phase2_label_audit.json`、`phase2_label_by_year.csv`、`phase2_label_by_date.parquet`、`phase2_alignment_mismatches.csv`，均位于 `reports/tables`。

### Phase 3 验收记录（2026-09-09）

- 已实现价格结构、收益/动量、趋势、波动、量能、价量相关、流动性、价格位置、突破和市场环境特征，共 158 个：147 个逐股票时序特征及 11 个逐日市场特征。
- 全量时序特征覆盖 7,900,350 行、4,650 只股票，输出 `data/processed/features_v1_time.parquet`，约 5.23 GB（4.87 GiB）、149 列、93 个 row groups。
- 市场特征按 1,699 个交易日独立存储于 `data/processed/features_v1_market.parquet`，避免将相同市场值重复扩展到全量股票行。
- 时序输出与训练输入主键哈希一致；全部时序与市场特征 Inf 数为 0。最低有限值覆盖率为 76.30%，来自完整 120 日窗口与原始缺失值，未填 0。
- 生产特征函数不读取 `y_ret_1d`，不使用负向 shift；rolling 均按单只股票、按日期排序计算，完整窗口才输出。逐日 rank/z-score 未提前加入，留给 Phase 4。
- 因果回归测试验证：追加或极端修改未来行不会改变历史时序特征或此前日期的市场特征。内置 rolling skew 的数值状态前视问题已被测试发现，并改为仅由当前窗口原始矩计算。
- 5 项独立标量手算的最大绝对误差为 `2.79e-08`（float32 范围）；完整测试套件 14 项全部通过。
- 原始训练集与测试集 SHA-256 均保持 Phase 1/2 记录值不变。
- 审计与说明产物：`reports/tables/phase3_feature_audit.json`、`phase3_feature_dictionary.csv`、`phase3_hand_checks.csv`。

### Phase 3 后框架复查（2026-09-09）

- 复查结论与后续契约记录于 `FRAMEWORK_REVIEW.md` 和 `ARCHITECTURE.md`。
- 针对 16 GB 内存环境，将预声明的 36 个建模核心特征重排为 2018–2024 按年分区；共 7,900,350 行、38 列、约 1.15 GB，逐年主键唯一。完整 147 列特征库继续保留用于后续消融，不重复复制。
- 修正市场上涨/下跌比例的分母：收益缺失记录不再被当作“未上涨/未下跌”，市场特征已重建。
- 三折 expanding-window、验证前 1 个交易日 purge、官方 evaluator 常量和 SHA-256 已在配置中冻结。
- `official/evaluate.py` 是官方脚本的字节级副本，SHA-256 为 `72DC59987E92BA8AE6B506E114DFE8591B5608C9BECCD0769D5D1F778F9189C2`；未修改其内容。
- 已增加 `requirements-lock.txt`，记录当前通过测试的完整 Python 环境。
- 稳定目录的覆盖改为临时目录完成后备份切换；失败时恢复旧版本，且默认拒绝静默覆盖。

### Phase 4 验收记录（2026-09-09）

- 对配置中预先声明的 36 个核心因子逐 `trade_date` 生成 percentile rank 与 z-score，共 72 个派生列；未根据标签或结果筛选因子。
- 输出覆盖 7,900,350 行、1,699 个交易日，按 2018–2024 七年分区并在每年内按 `trade_date, ts_code` 排序；总计约 1.94 GB（1.80 GiB）。
- 输入/输出主键哈希完全一致，所有年份主键唯一；Inf、小于 30 个有效样本的截面及零方差截面计数均为 0。
- rank 全局范围为 `[0.00022017, 1.0]`；并列值使用平均秩，缺失值保持缺失。
- 逐日 z-score 均值最大绝对偏差为 `5.09e-07`，标准差相对 1 的最大偏差为 `1.19e-07`，符合 float32 精度。
- 因果/分组测试确认修改未来日期不会影响历史日期，并覆盖并列、小截面、零方差及日期隔离规则；完整测试套件 18 项全部通过。
- 输出：`data/processed/features_v1_cross_sectional_by_year`；审计：`reports/tables/phase4_cross_sectional_audit.json`。

### Phase 5 验收记录（2026-09-09）

- 使用训练集最后 250 个交易日（2023-12-20 至 2024-12-31）作为只读历史 warm-up，并与完整测试期（2025-01-02 至 2026-06-08）按 `ts_code, trade_date` 连续计算；最大配置窗口为 120，预热长度充足。
- 时序产物只保留测试期 1,599,600 行、344 个交易日、4,650 只股票及 147 个特征；文件 `data/processed/features_v1_test_time.parquet` 约 1.18 GB，未写入任何训练期记录或标签列。
- 测试首日中短周期探针均非全缺失：`ret_1`、`ret_5`、`ret_20`、`close_ma_ratio_20`、`ret_std_20`、`position_20`、`close_ma_5_20` 分别有 4,525、4,518、4,512、4,499、4,496、4,499、4,499 个有限值；缺失来自原始价格缺失或完整窗口要求，未填 0。
- 市场特征在训练尾部与测试集拼接后计算，再只保留 344 个测试日期；测试首日有 4,525 个有效上一日收益，证明跨边界历史已生效。
- 36 个核心时序特征按 2025/2026 年分区，72 个逐日横截面特征同样分区；三套测试产物均为 1,599,600 行，主键哈希一致为 `342d0866de247e4c`，主键唯一且无 Inf。
- 测试时序、核心和横截面三类 schema（字段名、顺序、Arrow dtype）均与训练侧对应产物一致；未读取、创建或推断测试标签，未使用未来 shift。
- 原始训练/测试 CSV 的 SHA-256 在构建前后完全一致；完整测试套件 21 项通过。机器可读审计位于 `reports/tables/phase5_test_feature_audit.json`。
- 主要输出：`data/processed/features_v1_test_market.parquet`、`features_v1_test_model_base_by_year`、`features_v1_test_cross_sectional_by_year`。

### Phase 6 验收记录（2026-09-09）

- 已实现只接受 `expanding_window` 的确定性切分器，使用训练集实际 1,699 个交易日生成精确日期集合；禁止随机切分，并在配置中显式记录标签跨度 1 个交易日、purge 1 个交易日及逐折等权汇总规则。
- Fold 2022：训练 2018-01-02 至 2021-12-30，共 972 日、4,519,800 行，其中 3,568,878 个标签可用；purge 2021-12-31；验证 2022-01-04 至 2022-12-30，共 242 日、1,125,300 行，其中 1,029,017 个标签可用。
- Fold 2023：训练截至 2022-12-29，共 1,214 日、5,645,100 行，其中 4,597,785 个标签可用；purge 2022-12-30；验证 2023-01-03 至 2023-12-29，共 242 日、1,125,300 行，其中 1,062,411 个标签可用。
- Fold 2024：训练截至 2023-12-28，共 1,456 日、6,770,400 行，其中 5,659,954 个标签可用；purge 2023-12-29；验证 2024-01-02 至 2024-12-31，共 242 日、1,125,300 行，其中 1,094,055 个标签可用。
- 每折训练、purge、验证日期两两不重叠，训练日期集合逐折扩展；purge 覆盖下一交易日标签跨度。缺失标签只在生成可训练/可评价掩码时排除，未填 0。
- 新增只读 `load_train_columns()`，本阶段只从真实训练缓存读取 `ts_code, trade_date, y_ret_1d`，不加载无关行情列；原始 CSV 未作为输出目标。
- 精确折叠日历保存为 `reports/tables/phase6_cv_calendar.parquet`，日历 SHA-256 为 `B524226CDAEF5781D2D419E68677C9AFA4BCBEF92B8741DC8A978A71F4A57C10`；审计为 `phase6_cv_audit.json`。
- 覆盖真实交易日 purge、扩展集合、缺失标签掩码、训练/验证重叠拒绝和 purge 小于标签跨度拒绝等边界；完整测试套件 26 项全部通过。

### Phase 7 验收记录（2026-09-09）

- 已实现 `src/validation/evaluator.py`，输出官方规定的八项指标：`ic_mean, ic_std, icir, ic_positive_ratio, annual_excess, top1_annual_ret, mean_turnover, final_score`；文件读取仍按官方两次 inner merge 及指定字段执行。
- Rank IC、Top 组收益和换手率分别保留官方不同的有效股票池；30/100 样本阈值、`len(valid) // 10` 取整、252 日年化、涨停过滤、标签缺失过滤、换手 Jaccard 距离及不足 100 时重置前一集合均未统一或改写。
- 配置中的阈值与权重只用于锁定契约校验；任何常量、权重或官方 SHA-256 漂移都会直接报错，不作为模型调参项。
- 使用 720 行、6 个日期的确定性合成夹具，同时调用官方文件 evaluator 与本地 evaluator；八项指标最大绝对差为 `0.0`，低于锁定容差 `1e-12`。夹具仅用于程序行为对照，其 `final_score=2.550764` 不代表模型或真实数据成绩。
- 边界覆盖包括：29 条 Rank IC 样本跳过、30 条纳入；99 条组合样本跳过、100 条纳入；109 条取 Top 10、110 条取 Top 11；不足 100 条的日期清空换手状态；涨停与缺失标签导致三个有效池不同。
- `official/evaluate.py` 运行前后 SHA-256 均为 `72DC59987E92BA8AE6B506E114DFE8591B5608C9BECCD0769D5D1F778F9189C2`，文件未修改。
- 完整测试套件 29 项全部通过；审计保存于 `reports/tables/phase7_evaluator_parity.json`。Phase 7 未生成模型、OOF 或测试预测。

### Phase 8 验收记录（2026-09-09）

- 构建训练专用按年模型视图：7,900,350 行、119 个显式允许特征、6,758,531 个真实非空标签、Inf 为 0，总计约 2.99 GB；键哈希 `89f1e0cc9d675b51` 与 Phase 3 全量时序特征一致。
- 简单因子为预声明的正向 `ret_20__rank`，三折平均 `ic_mean=-0.03739`、`annual_excess=-0.73574`、`mean_turnover=0.36774`、`final_score=-0.04600`。该结果只证明全流程打通，不在看结果后静默反转方向。
- 流式 Ridge 使用 36 个同日 z-score 特征、缺失映射到横截面中性值 0、`alpha=100`；训练只使用各折非空真实标签。三折平均 `ic_mean=0.06085`、`annual_excess=0.11512`、`mean_turnover=0.87616`、`final_score=0.09603`，最差折分数 `0.06448`。
- LightGBM 使用 36 个同日 rank 与 11 个市场特征、全量训练样本、100 轮固定种子训练；三折平均 `ic_mean=0.07937`、`annual_excess=0.20571`、`mean_turnover=0.50413`、`final_score=0.24222`，最差折分数 `0.18935`。
- LightGBM 各折 IC 为 0.07935/0.07967/0.07910，较稳定；换手为 0.23261/0.52420/0.75559，跨年显著恶化，已记录为 Phase 10 的首要风险，不在本阶段追加后验调参。
- 每个模型的每折 OOF 均覆盖完整 1,125,300 行验证截面，包括标签缺失行；键唯一、预测有限、dtype 为 float32，同折模型键哈希一致。九个落盘 OOF 重新评分后与审计八项指标最大差为 `0.0`。
- 指标按三个折等权汇总；不把跨折 OOF 拼接后产生的额外年度边界换手作为选模分数。测试数据未参与训练、验证、调参或模型选择。
- CatBoost 属计划中的资源允许项；基于 16 GB 内存及“先不训练复杂模型”的边界，本阶段明确暂缓，不用抽样成绩冒充全量结果。
- 主要产物：`data/processed/model_view_v1_by_year`、`models/phase8_baselines`、`reports/tables/phase8_baseline_audit.json`；完整测试套件 34 项全部通过。

### Phase 8 后框架复查（2026-09-09）

- 复查结论记录于 `FRAMEWORK_REVIEW.md`。整体分层、泄漏边界、CV 和 evaluator 契约合理，无需调整 Phase 9–12 顺序。
- 已修正两项可复现性问题：评分统一使用实际落盘的 float32 预测；每个 OOF 增加行数、键哈希、唯一性、有限值和 dtype 审计。
- 新增架构约束：训练模型必须读取 manifest 中的显式特征白名单，不能通过“排除标签列”猜测特征；跨折汇总保持逐折等权，避免人为新增年度边界换手。
- 下一次整体框架复查按约定在 Phase 12 完成后进行。

### Phase 9 验收记录（2026-09-09）

- 已建立 `experiments/experiment_registry.csv`，固定字段覆盖实验 ID、时间、Git 状态、源码/配置/特征指纹、模型参数、训练/验证期、种子、官方八项指标、汇总稳定性和产物路径。
- Phase 8 的简单因子、Ridge、LightGBM 分别登记为 `p08_simple_factor_v1`、`p08_ridge_v1`、`p08_lightgbm_v1`；每个实验含 3 条逐折记录和 1 条等权汇总，共 12 条，复核后组合键全部唯一。
- 保存当前 `config.yaml` 的不可变快照，SHA-256 为 `3A5057772EB547038E4C54198A251F343F1E5F3BFE882110F3A8A3C0C5E19795`；配置快照位于 `experiments/configs`。
- 55 个可执行源码、脚本、测试、依赖清单及官方 evaluator 组成源码树，SHA-256 为 `FDC0473C9D9EFD3F93B77EBF27768082E14BA0ED0AE893DCB562981E8051F41C`；逐文件清单位于 `experiments/manifests`。
- 对原始训练文件、训练视图七年分区、九个 OOF、六个模型文件、CV/evaluator/Phase 8 审计、依赖锁文件等 33 个关键产物完成 SHA-256，合计约 4.01 GB；产物清单文件 SHA-256 为 `3B12D1486C7C44FA30B3D4EFE7085C1AFA4F8BD396D5335C4275D20D8682DEDE`。
- Git 仓库尚无首个 commit，登记如实记录 `git_commit=UNBORN`、`git_dirty=true`，未擅自提交；源码树哈希为当前阶段提供可验证替代指纹。
- 注册写入采用临时文件原子切换；相同组合键及相同内容再次导入会跳过，组合键相同但内容变化会拒绝覆盖。正式二次导入结果为新增 0、跳过相同记录 12、总数仍为 12。
- 实验汇总指标与 Phase 8 审计一致；没有重新训练模型，也没有读取或登记测试标签。完整测试套件 36 项全部通过，审计位于 `reports/tables/phase9_experiment_registry_audit.json`。

### Phase 10 验收记录（2026-09-10）

- 只对 Phase 8 三折 LightGBM OOF 做后处理，不重新训练模型。流程依次为逐日横截面 percentile rank、按 `ts_code` 的因果 EMA、基于上一有效日 Top 集合的 entry/exit buffer，以及连续、唯一、确定性的 float32 分数映射。
- EMA 仅使用当前及历史行；每折从首个验证观测初始化并独立重置。接口支持传入上期每股 EMA 状态，供后续测试推理跨边界续接；换仓缓冲仅使用当前分数、当日涨停标记和上一有效日入选集合，不读取标签。
- 在配置中预先固定 `alpha=[1.0,0.9,0.8,0.7,0.6,0.5]` 与 exit fraction `[10%,12.5%,15%]`，共 18 个候选；每个候选均在 2022/2023/2024 三折上重算官方八项指标，形成 54 条逐折记录和 18 条等权汇总记录。
- 按三折等权 `final_score` 及预声明确定性次级排序选出 `alpha=0.5, exit=15%`。平均 `ic_mean=0.08211`、`annual_excess=0.27072`、`mean_turnover=0.29337`、`final_score=0.32605`，分数标准差 `0.05067`、最差折 `0.27436`。
- 相对未经后处理的 Phase 8 LightGBM，平均分数提高 `0.08382`，平均换手下降 `0.21076`；三个验证折分数依次为 `0.37563/0.32814/0.27436`。最优点位于预声明网格边界，作为后续稳定性风险记录，不在观察结果后临时扩展网格。
- `alpha=1, exit=10%` 是逐日 rank 与确定性并列排序对照；其与原始预测的细小差异来自树模型同分行的确定性重排，最大指标差为换手率 `0.00146`，不是标签驱动调整。
- 最优三折 OOF 覆盖各 1,125,300 行，主键唯一、预测有限且为 float32；网格表共 72 行，保存为 `phase10_turnover_grid.parquet/csv`，审计为 `phase10_turnover_audit.json`，模型产物位于 `models/phase10_turnover`。
- 实验 `p10_lgb_turnover_v1` 已追加 3 条逐折及 1 条汇总记录，注册表累计 16 条。源码树 SHA-256 为 `572269D31886E335F18BD3A9F55F5A40F3AE998BC121F8CA689591DC758AE5A7`，产物清单 SHA-256 为 `05885697F8D836B0F8D883B09C1CE3BD6FBC80368B9B60390F29FCF2EB88AE91`。
- 后处理函数的因果性、历史状态续接、缓冲保留和不足阈值日重置均有自动测试；测试数据和测试标签均未参与调参或选择，原始 CSV 未修改。

### Phase 11 验收记录（2026-09-10）

- 使用 Phase 8 完整验证域的 LightGBM 与 Ridge OOF；两个模型先分别按 `trade_date` 转为 percentile rank，再做非负、和为 1 的加权融合。简单因子因 Phase 8 三折得分为负而排除，CatBoost 因 16 GB 资源边界尚未训练，不用伪造或抽样结果补位。
- 在配置中预先固定 LightGBM 权重 `0.00–1.00`、步长 `0.05` 的 21 组网格，Ridge 权重为补数。所有候选统一使用 Phase 10 已锁定的 `alpha=0.5, exit=15%`，没有针对不同融合权重重新调整换手参数。
- 每个候选在 2022/2023/2024 三折上重新计算官方八项指标，共保存 63 条逐折记录和 21 条等权汇总记录。LightGBM 权重 1.00 的控制组逐指标精确复现 Phase 10，最大差为 `0.0`；等权融合分数为 `0.30240`。
- 按三折等权 `final_score` 选出 LightGBM `65%`、Ridge `35%`。平均 `ic_mean=0.08435`、`annual_excess=0.30142`、`mean_turnover=0.27783`、`final_score=0.34082`，分数标准差 `0.05662`、最差折 `0.28269`。
- 相对 Phase 10 的 LightGBM 单模型控制组，平均分数提高 `0.01477`，平均换手再下降 `0.01555`；三折分数依次为 `0.39580/0.34396/0.28269`。相邻权重 0.60/0.70/0.75 的平均分数均接近最优，最优点不在网格边界。
- 最优三折 OOF 各覆盖 1,125,300 行，保留完整缺失标签行供官方换手评分；主键与两个来源 OOF 完全一致，预测有限且为 float32。权重网格保存为 `phase11_ensemble_grid.parquet/csv`，审计为 `phase11_ensemble_audit.json`，产物位于 `models/phase11_ensemble`。
- 实验 `p11_lgb_ridge_ensemble_v1` 已追加 3 条逐折和 1 条汇总记录，注册表累计 20 条。源码树 SHA-256 为 `44308AC41BB19A418D5B677E6F857904194C0CC7728CE60D1192E3FBF03A91F8`，产物清单 SHA-256 为 `D3F01499F33E37A9297BBB962D680629A9806C8C3E7345A668F41F120EE1E3DB`。
- 融合函数覆盖端点权重、尺度不变性、未来日期隔离、权重约束和主键错位拒绝测试；测试数据及测试标签未用于权重选择，原始数据未修改。完整测试套件 44 项全部通过。

### Phase 12 验收记录（2026-09-10）

- 使用与 Phase 8 回归 LightGBM 完全相同的 36 个横截面 rank 特征、11 个市场特征、100 轮 boosting 和三折 expanding-window；模型目标改为 LightGBM LambdaRank，未额外引入外部字段或测试数据。
- 每个训练日仅对真实非空 `y_ret_1d` 做同日 percentile rank，使用 `ceil(rank_pct × 10) - 1` 映射为 0–9 十档整数 relevance；并列收益取平均秩，缺失标签先排除、不填 0，线性 `label_gain=[0,...,9]`。
- Parquet batch 通过日期尾部缓冲合并，保证一个交易日不会跨 group；三折分别形成 972/1,214/1,456 个连续日期 group，group size 总和精确等于 3,568,878/4,597,785/5,659,954 个真实训练标签，单组大小范围为 3,370–4,521。
- 原始 LambdaRank 三折等权 `ic_mean=0.10915`、`annual_excess=0.39735`、`mean_turnover=0.43270`、`final_score=0.33306`，相对 Phase 8 回归 LightGBM 提高 `0.09083`。
- 使用 Phase 10 预先锁定、未重新调参的 `alpha=0.5, exit=15%` 后，三折等权 `ic_mean=0.10419`、`annual_excess=0.33331`、`mean_turnover=0.21230`、`final_score=0.37798`；标准差 `0.06489`、最差折 `0.31573`。
- 后处理 ranker 相对 Phase 10 回归提高 `0.05193`，相对 Phase 11 LightGBM/Ridge ensemble 提高 `0.03716`。三折分数为 `0.44522/0.37299/0.31573`；虽均高于此前 incumbent，但跨年下降及换手从 `0.08281` 升至 `0.32056`，已列为 Phase 15 的首要稳定性风险。
- 两种 variant 的六份 OOF 各覆盖完整 1,125,300 行，主键唯一、预测有限、dtype 为 float32；逐份重新评分与审计八项指标最大差为 `0.0`。模型及 OOF 位于 `models/phase12_lambdarank`。
- 实验 `p12_lambdarank_raw_v1` 与 `p12_lambdarank_turnover_v1` 各登记 3 条逐折和 1 条汇总记录，注册表累计 28 条。源码树 SHA-256 为 `228A0329CC8D8488BB5B03822F617B770174FAC1F9ACA2622AA47D02CC385FDE`，约 3.03 GB 的 23 项产物清单 SHA-256 为 `A32B48A98B711CE7B77426DF56BEA9F3ADB6A3B4CB843F6F7DADA39F7F4EBF19`。
- 本次全量运行约 1,628 秒；框架复查后已为未来复跑补充逐折与总耗时字段，不回写或伪造本次已登记的实验快照。完整测试套件 48 项全部通过，测试数据及测试标签未用于训练或选模，原始数据未修改。

### Phase 12 后框架复查（2026-09-10）

- 复查结论记录于 `FRAMEWORK_REVIEW.md` 和 `reports/tables/phase12_framework_review.json`。数据分层、泄漏边界、CV、官方 evaluator 与实验登记仍合理，无需调整 Phase 13–16 的阶段顺序。
- 后处理 LambdaRank 替代 Phase 11 ensemble 成为后续 incumbent；Phase 13–15 的正式增量必须同时报告相对该模型的平均分、最差折、标准差与换手，不以单折 IC 或训练损失替代。
- 考虑 LambdaRank 全量复训成本，Phase 13/14 先做无泄漏的因子覆盖、逐日 IC 和分组筛选，仅对通过筛选的少量候选执行三折全量 ranker；不能用便宜代理模型的结果冒充最终模型结论。
- Phase 16 必须重新比较 LambdaRank 单模型以及包含 LambdaRank、LightGBM、Ridge 的融合，Phase 11 权重不能直接沿用，因为当时 ranking OOF 尚不存在。
- Phase 17 必须用最终模型对至少 250 个训练尾部交易日生成预测，显式续接每股 EMA 与上一有效 Top 集合后再处理测试首日；不得把 OOF 中的“验证首日重置”误当作生产推理规则。
- 下一次计划复查点为 Phase 16 完成后。

### Phase 13 验收记录（2026-09-10）

- 新增 32 个逐股票因果高级特征，覆盖 RSI、MACD、ATR、Bollinger、OBV-like 量能流、MFI、成交量加权调整价 VWAP 代理、趋势效率、反转/动量反转、量价突破与波动突破；仅使用 `t` 日及以前的官方量价与涨跌停字段，不读取标签。
- 训练产物按 2018–2024 年分区，共 7,900,350 行；测试产物按 2025–2026 年分区，共 1,599,600 行。两侧均为 34 列（主键 2 列加特征 32 列）、主键唯一、键哈希与各自输入完全一致且 Inf 为 0，总体约 1.38 GB。
- 测试特征先拼接 250 个真实训练尾部交易日，再仅保留测试日期；最长窗口为 60 日，首个测试日的真实抽样股票 32 个特征全部非空。测试输出没有标签列，也未创建或推断测试标签。
- 真实数据表明 OHLC 为调整价，而 `amount/vol` 隐含的未调整价格换算比例不稳定；因此 VWAP 代理明确采用 `rolling_sum(close*vol)/rolling_sum(vol)`，不把任意常数乘到成交额上。该选择仅依据无标签的单位一致性检查。
- 五项真实数据手算最大绝对误差为 `2.64e-09`；训练最低有限覆盖率为 `80.66%`，测试最低为 `94.65%`，缺失来自完整窗口和原始缺失，未填 0。52 项完整测试全部通过。
- 产物审计为 `reports/tables/phase13_advanced_feature_audit.json`，特征字典为 `phase13_advanced_feature_dictionary.csv`。源码树 SHA-256 为 `33A42426258C840E5F0A590699A0B392938D37A65C2C7A108546C952EDBE8E66`，16 项产物清单 SHA-256 为 `DA8505FE4EA8643826B65B9415D89B60B3C4F0BCBCD86A38CC51ADE43168BA4E`，实验注册表累计 29 条。
- 本阶段没有做标签相关筛选、横截面 IC、模型训练或调参；这些严格留给 Phase 14。下一次整体框架复查仍在 Phase 16 完成后。

### Phase 14 验收记录（2026-09-10）

- 对 Phase 13 的 32 个高级特征计算 2018–2024 共 1,699 个交易日的同日 percentile-rank IC，生成 54,368 条逐日记录；缺失因子 rank 按预声明规则映射为中性值 `0.5`，同时单独保留覆盖率，不用 0 冒充原始特征。
- 特征选择严格只使用 2018-01-02 至 2021-12-31：门槛、方向、逐年同号数、每族上限、总数上限和平均同日 rank 相关性去冗余均在读取留出期结果前固定。2022/2023/2024 仅用于验证，未反向改变名单。
- 共选出 16 个候选：`atr_norm_14`、`reversal_5`、`close_vwap_ratio_20`、`macd_diff_12_26`、`atr_norm_28`、`volume_breakout_20`、`bollinger_position_20`、`rsi_6`、`rsi_24`、`trend_efficiency_20`、`obv_flow_ratio_5`、`trend_efficiency_10`、`close_vwap_ratio_5`、`macd_hist_12_26_9`、`bollinger_width_20`、`obv_flow_ratio_20`。48 个“候选×留出年份”方向检查中仅 1 个转负。
- 16 特征等权同日 rank 组合在三个留出年的代理分数依次为 `0.21507/0.18211/0.17517`，等权均值 `0.19078`、标准差 `0.01741`、最差 `0.17517`，平均 IC `0.05248`、换手 `0.47311`。该组合没有模型拟合和 Phase 10 后处理，明确不能与完整 LambdaRank incumbent 作同口径结论。
- 家族单独及逐族剔除共 23 个候选完成官方口径留出评分。ATR 家族代理均分最高，为 `0.33395`，平均换手 `0.10710`；这是 Phase 15 稳定性分析的优先方向，不据此在本阶段追加模型或后验改名单。
- 60 日窗口中的 8 个特征因预筛选期总体覆盖率低于 `0.75` 被排除；`amount_breakout_20` 与 `volume_breakout_20` 的平均同日 rank 相关性为 `0.9910`，保留后者；另外 4 个合格特征因预声明总数上限未入选。所有原因均逐项写入选择审计。
- 极端波动、低波动、上涨及下跌市场阈值只由 2018–2021 市场状态分位数拟合，再原样应用于留出年份。测试集未读取，未创建测试标签，未训练预测模型，训练集原始文件哈希未改变。
- 审计文件为 `reports/tables/phase14_feature_screening_audit.json`，锁定名单为 `phase14_selected_features.json`，逐日、年度、市场状态、冗余和家族消融表均保存在 `reports/tables`。源码树 SHA-256 为 `7F866AB3CF2E1779849D2A13DEF58BC475004407C9B7BC9BA633A97521E500B7`，18 项产物清单 SHA-256 为 `A2F4BF9C48E97EFCD3C1C2940F07ECBD92059E6F9E353C4EFDF680AF0723AFD6`，实验注册表累计 30 条，55 项测试全部通过。
- 下一阶段仅实施 Phase 15 的市场状态、训练窗口和跨年稳定性验证；下一次整体框架复查仍在 Phase 16 完成后。

### Phase 15 验收记录（2026-09-10）

- 依据 Phase 14 预筛选期排名最高的 ATR 家族，仅将 `atr_norm_14__rank`、`atr_norm_28__rank` 加入 Phase 12 的 47 个特征，形成 49 特征紧凑训练视图。视图按 2018–2024 年分区，共 7,900,350 行；未复制与本实验无关的原始/z-score 特征。
- 固定三组训练窗：expanding 全历史、验证年前四个完整日历年、验证年前两个完整日历年。三组使用相同 2022/2023/2024 验证全集、同一交易日 purge、十档 relevance、100 轮 LambdaRank，以及 Phase 10 锁定的 `alpha=0.5, exit=15%`，只改变训练历史长度。
- ATR expanding 后处理三折分数为 `0.42803/0.36455/0.30134`，均值 `0.36464`、最差折 `0.30134`、标准差 `0.06334`、换手 `0.21531`；相对 Phase 12 incumbent 均值下降 `0.01334`，说明 ATR 单因子代理优势未转化为 ranker 增量。
- ATR 近四年后处理均分 `0.36246`、最差 `0.29390`、换手 `0.20914`，同样不优于 incumbent。其 2022 折与 expanding 训练日期完全相同，复用前先重放 OOF；模型和 raw/turnover OOF 的 SHA-256 均字节一致。
- ATR 近两年后处理三折为 `0.41511/0.35463/0.32628`，均值 `0.36534`、最差 `0.32628`、标准差 `0.04537`、换手 `0.16846`。它在三个 ATR 窗口中均值最高，且相对 incumbent 改善最差折 `0.01055`、降低换手 `0.04383`，但平均分仍低 `0.01264`，因此只进入 Phase 16 复查，不能替代 incumbent。
- 使用 Phase 14 在 2018–2021 拟合后锁定的市场状态阈值评估完整 OOF 日序列。近两年窗在低波动状态的代理分 `0.36238` 高于 incumbent 的 `0.35305`，高波动状态接近（`0.43978` 对 `0.44289`），但上涨和下跌市场均落后；证据不足以建立独立 regime 模型。
- 18 份新 OOF 各 1,125,300 行，主键唯一、预测有限且为 float32；逐份重算官方八项指标最大差为 `0.0`。断点恢复只接受模型、raw OOF、turnover OOF 同时存在的完整折，并重新评分，不把半成品提升为正式产物。
- 完整运行审计耗时约 `1,712` 秒（恢复前已完成折的原始耗时如实标记未知）；测试集未参与训练或窗口选择，原始训练/测试 CSV 哈希均未改变。源码树 SHA-256 为 `5E76AC53BB499A831AFBF99D002E1F0332FE4B816C6C42033DABAE02F8C788D8`，52 项产物清单 SHA-256 为 `4E918E8398A4F448382CA7DA5C5AA2219CC16CF232422E3E821D136F72DEE925`，实验注册表累计 54 条。
- Phase 12 后处理 LambdaRank 继续保持 incumbent。Phase 16 应同时复查 incumbent、近两年 ATR 模型和既有 LightGBM/Ridge OOF 的融合；完成 Phase 16 后按约定执行下一次整体框架复查。

### Phase 16 验收记录（2026-09-11）

- 预声明三组件 0.20 步长单纯形网格，共 21 个候选：Phase 12 基础 LambdaRank、Phase 15 近两年 ATR LambdaRank，以及固定为 LightGBM 65% / Ridge 35% 的回归 sleeve。各组件先做同日 percentile rank，融合后只应用一次 Phase 10 已锁定的 `alpha=0.5, exit=15%` 后处理。
- 三个单组件端点对历史八项官方指标的复刻差均为 `0.0`。最终选择基础 LambdaRank 60%、ATR 近两年 LambdaRank 20%、回归 sleeve 20%；三折分数为 `0.44822/0.37320/0.32395`，等权均值 `0.38179`、标准差 `0.06258`、最差折 `0.32395`、平均换手 `0.20343`。
- 相对 Phase 12 incumbent，均分提高 `0.00381`、最差折提高 `0.00822`、平均换手降低 `0.00886`；因此实验 `p16_locked_ensemble_v1` 成为 Phase 17 的唯一锁定方案，不再在测试期调权。
- 已使用真实非空训练标签完成生产模型训练：2018–2024 基础 ranker 为 6,758,531 行、47 特征；2023–2024 ATR ranker 为 2,156,466 行、49 特征；全历史回归 LightGBM 为 6,758,531 行、47 特征；全历史 Ridge 为 6,758,531 行、36 特征。模型位于 `models/phase16_final`，训练 memmap 已清理。
- 三份最终 OOF 各 1,125,300 行、主键唯一且预测有限；训练/测试原始 CSV SHA-256 均未改变。阶段耗时约 2,289 秒，注册表新增 4 条、累计 58 条；60 项测试全部通过。
- Phase 16 后框架复查记录于 `FRAMEWORK_REVIEW.md` 与 `reports/tables/phase16_framework_review.json`。已将 incumbent 指向 Phase 16，并明确 Phase 17 必须按四个模型各自的特征清单构造训练尾部与测试输入，融合后跨边界连续传递 EMA 和 Top 集合状态。

### Phase 17 验收记录（2026-09-11）

- 使用 Phase 16 锁定的四个生产模型和嵌套权重直接推理，未重新拟合模型、选择权重或修改后处理。基础 ranker / ATR ranker / 回归 LightGBM / Ridge 分别严格校验 47 / 49 / 47 / 36 个特征的名称、顺序和 dtype，训练尾部与测试侧全部一致。
- 从真实训练模型视图读取最后 250 个交易日（2023-12-20 至 2024-12-31，共 1,162,500 行）生成组件预测和同日排名；warm-up 结束时显式传递 4,650 个股票 EMA 状态和 459 个 Top 集合成员到测试首日，没有在 2025-01-02 重置。
- 测试期覆盖 2025-01-02 至 2026-06-08 的 344 个交易日、1,599,600 行。最终预测为有限 `float32`，范围 `0.00021505` 至 `1.0`；每日恰有 4,650 个唯一排序分值。
- `models/phase17_prediction/test_predictions.parquet` 仅包含 `ts_code, trade_date, pred`，保持原始测试集行序，主键集合与未修改的测试 CSV 完全一致，键哈希为 `342d0866de247e4c`，文件 SHA-256 为 `FAD7277364F2642871337DC3035C9F20988D6FC93881C79DEA46632A262B74EB`。
- 本阶段未读取或推断测试标签，原始训练/测试文件哈希未改变，也未生成 submission CSV。审计位于 `reports/tables/phase17_prediction_audit.json`；运行约 64 秒，实验注册表新增 1 条、累计 59 条，完整 62 项测试全部通过。Phase 18 将单独负责 CSV 格式化、回读和自动检查。

### Phase 18 验收记录（2026-09-11）

- 从 Phase 17 锁定预测 Parquet 生成 `submissions/submission_p16_locked_ensemble_v1.csv`。写入前和临时 CSV 回读后均严格检查列名与顺序 `ts_code, trade_date, pred`、1,599,600 行、主键唯一、键集合、原始测试行序、股票代码、八位日期及有限数值。
- 与未修改测试键 inner merge 后仍为 1,599,600 行，键哈希为 `342d0866de247e4c`。CSV 回读为字符串代码、`int32` 日期和数值预测；回读预测转为 `float32` 后与 Phase 17 Parquet 逐元素完全一致。
- 最终 CSV 为 47,329,264 字节，SHA-256 为 `19125DE77D0726A23A5E9E209C03960F53EC1E69D312D0A2DD32A0BA2A274865`。同名文件默认拒绝静默覆盖，只有显式 `--overwrite` 才允许替换。
- 本阶段没有读取或推断隐藏测试标签，也没有伪造或报告测试集官方得分；原始训练/测试 CSV 哈希均未改变。审计位于 `reports/tables/phase18_submission_audit.json`，实验注册表新增 1 条、累计 60 条，完整 64 项测试全部通过。

### Phase 19 验收记录（2026-09-11）

- 基于 Phase 1–18 已固化的 JSON/CSV 审计和 OOF 指标生成最终报告，没有重新训练、读取隐藏测试标签或修改 submission。报告明确区分 walk-forward OOF 分数与未知的隐藏测试成绩。
- 最终方案为基础 LambdaRank 60%、近两年 ATR LambdaRank 20%、回归 sleeve 20%；报告展示三折分数 `0.44822/0.37320/0.32395`、均值 `0.38179`、换手 `0.20343`，所有数字均可回溯到已有阶段审计。
- `output/pdf/fintechathon2026_task5_report.pdf` 正好 8 页，含数据介绍、描述性分析、特征与验证设计、模型对比、稳定性、应用验证、产品思路和总结；五张图均由 `scripts/19_build_final_report.py` 从结构化产物生成，并已逐页渲染完成视觉检查。
- 新增 `PROJECT_GUIDE.md`，系统说明端到端设计、每个脚本/源码/测试文件职责，以及快速测试、专项检查、submission/报告验收和从零重跑方式。
- 交付审计为 `reports/tables/phase19_delivery_audit.json`，复现说明为 `REPRODUCIBILITY.md`，最终清单为 `FINAL_DELIVERY_MANIFEST.json`；实验注册表新增 1 条、累计 61 条。下一次整体框架复查仍按约定在 Phase 20 完成后触发。

## 12. 第一阶段明确不做

- 不直接使用 Transformer、LSTM 等高成本模型；
- 不生成未经筛选的数千因子；
- 不使用外部行情、基本面、指数成分、新闻或替代数据；
- 不随机切分；
- 不用测试集构造标签；
- 不将缺失标签填 0；
- 不根据测试期不可见结果调参；
- 不只看 RMSE、单折 Rank IC 或训练集表现；
- 不覆盖原始数据、历史模型、实验记录和 submission。

## 13. 可直接复制给 Codex 的启动指令

~~~text
请以 D:\codex\fintechathon2026 为项目根目录，先完整阅读 README.md，并按其中 Phase 0→19 的顺序推进。

本轮只实施 Phase 0 和 Phase 1，不提前训练模型。开始前检查项目现状、已有文件和 data/raw 下的数据；保留已有改动，不修改任何原始 CSV。

请完成：
1. 创建最小项目骨架、requirements.txt、config.yaml、.gitignore 和 AGENTS.md；
2. 实现统一路径、日志、随机种子、训练/测试读取和可选 Parquet 缓存；
3. 实现 scripts/01_check_data.py 与 src/data/validator.py，检查字段、dtype、日期范围、股票数、每日样本数、主键、缺失、Inf、OHLC、成交量/额和涨跌停标记；
4. 适配真实训练集在本地但未上传对话的情况；若训练集暂不可读，使用测试集验证管线，清楚标注未验证项，绝不伪造标签；
5. 控制大文件内存，路径从 config.yaml 读取，原始数据只读；
6. 为主键、OHLC 和缓存一致性编写必要测试；
7. 运行本阶段检查并修复本次实现引入的问题；
8. 最后汇报文件变更、实际运行结果、未验证事项和进入 Phase 2 的条件，并更新 README 状态表。

不要改写官方 evaluate.py，不要用随机数据冒充真实训练结果，不要开始 Phase 2 之后的模型实验。
~~~

## 14. 完成定义

项目只有在以下条件全部满足时才算完成：

- 数据、特征、CV、evaluator、模型、后处理和 submission 可在新环境按说明重跑；
- evaluator 与官方脚本口径一致；
- 所有训练与验证无已知未来数据泄漏；
- 最终模型与后处理由锁定的 Walk-forward 规则选择；
- submission 与测试集主键一一对应并通过自动检查；
- 实验记录能解释最终方案相对 baseline 的增量；
- 报告符合官方章节和页数要求，图表来自可复现产物；
- 未验证、失败或仅属建议的内容均被如实标注。
