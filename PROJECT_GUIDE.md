# 题五项目设计、文件职责与测试指南

项目根目录固定为 `D:\codex\fintechathon2026`。以下命令均在该目录执行；原始 CSV 只读，缓存、模型和报告均保存在 D 盘。

## 1. 端到端设计流程

```text
官方训练/测试 CSV（只读）
  -> 01 数据契约、检查与 Parquet 缓存
  -> 02 标签定义、缺失和时间对齐审计
  -> 03 因果时序/市场特征
  -> 04 同日横截面 rank 与 z-score
  -> 05 测试特征（真实训练尾部 warm-up，无测试标签）
  -> 06 expanding walk-forward + 1 交易日 purge
  -> 07 官方评价指标独立复刻和一致性验证
  -> 08 Ridge / LightGBM baseline
  -> 09 配置、源码、产物哈希和实验注册
  -> 10 EMA + Top 集合滞回降低换手
  -> 11 LightGBM/Ridge 回归融合
  -> 12 LambdaRank 排序模型
  -> 13 高级因果候选特征
  -> 14 仅用 2018–2021 标签筛选特征
  -> 15 训练窗口与市场状态稳定性验证
  -> 16 锁定 60/20/20 融合并训练生产模型
  -> 17 跨训练/测试边界连续状态推理
  -> 18 测试键对齐、CSV 回读和 submission 固化
  -> 19 8 页报告、复现说明和交付清单
```

核心边界：特征只读同一股票 `t` 日及以前；标签只在训练/验证装配时连接；测试侧不创建或推断标签；2022/2023/2024 是三个独立验证折；模型选择看三折官方分数均值、最差折、标准差和换手；测试首日继承训练尾部 EMA/Top 状态；Phase 18 的结构验收不等于隐藏测试成绩。

## 2. 根目录文件

| 文件 | 作用 |
|---|---|
| `README.md` | Phase 0–19 总计划、状态和逐阶段验收事实。 |
| `PROJECT_GUIDE.md` | 本文：流程、文件职责、测试与重跑入口。 |
| `ARCHITECTURE.md` | 数据层、泄漏边界、下游锁定契约和产物安全规则。 |
| `FRAMEWORK_REVIEW.md` | 每四阶段一次的框架复查；最近一次在 Phase 16。 |
| `REPRODUCIBILITY.md` | 环境、输入哈希、运行顺序和最终产物。 |
| `FINAL_DELIVERY_MANIFEST.json` | 最终交付文件路径、大小与 SHA-256。 |
| `config.yaml` | D 盘路径、dtype、特征、CV、模型、后处理、预测、提交和报告的统一配置。 |
| `requirements.txt` / `requirements-lock.txt` | 可维护依赖范围 / 已验证精确版本。 |
| `AGENTS.md` | 自动化修改本仓库时必须遵守的安全与阶段约束。 |
| `.gitignore` | 排除原始数据、缓存、模型、日志和大文件产物。 |

## 3. `scripts/`：Phase 入口

| 文件 | 作用 |
|---|---|
| `01_check_data.py` | 只读检查 schema、日期、主键、缺失、Inf、OHLC、成交量和涨跌停，并建立身份保护缓存。 |
| `02_audit_labels.py` | 核验 `y_ret_1d`、股票边界、末行缺失和年度分布。 |
| `03_build_features.py` | 生成训练侧因果时序、市场特征和分年核心存储。 |
| `04_build_cross_sectional.py` | 生成同日 rank/z-score 和显式 allowlist 模型视图。 |
| `05_build_test_features.py` | 用真实训练尾部 warm-up 生成无标签测试特征。 |
| `06_validate.py` | 建立 2022/2023/2024 expanding folds 并检查 purge。 |
| `07_check_evaluator.py` | 本地评价器与 `official/evaluate.py` 逐项/边界/压力一致性验证。 |
| `08_train_baseline.py` | 训练 Ridge、LightGBM baseline 并保存完整验证全集 OOF。 |
| `09_register_experiments.py` | 固化配置、源码树、产物清单和实验记录。 |
| `10_optimize_turnover.py` | 选择因果 EMA 与 Top 集合滞回参数。 |
| `11_build_ensemble.py` | 精确对齐 OOF，日内排名后搜索 LGB/Ridge 非负权重。 |
| `12_train_lambdarank.py` | 按交易日 group 训练 LambdaRank。 |
| `13_build_advanced_features.py` | 构造 RSI、MACD、ATR、Bollinger、量价与趋势等 32 个候选。 |
| `14_screen_features.py` | 只用 2018–2021 标签做方向、覆盖、冗余和家族筛选。 |
| `15_analyze_stability.py` | 比较 expanding、近四年、近两年训练窗与市场状态。 |
| `16_finalize_model.py` | 搜索三组件单纯形，锁定 60% 基础 ranker、20% ATR ranker、20% 回归 sleeve，训练四个生产模型。 |
| `17_predict_test.py` | 校验四套 schema，以训练尾部连续状态推理并写预测 Parquet。 |
| `18_build_submission.py` | 按原测试行序生成三列 CSV，写前/回读后双重校验。 |
| `19_build_final_report.py` | 从审计产物生成五张图、8 页 PDF、Markdown、复现说明和交付审计，不重训。 |
| `repartition_feature_store.py` | 历史大特征文件的分年重分区维护工具，非正常执行链。 |

## 4. `src/`：实现文件

### 数据、特征与验证

| 文件 | 作用 |
|---|---|
| `src/data/loader.py` | CSV/Parquet 分块读取、dtype、缓存身份校验和只读入口。 |
| `src/data/validator.py` | schema、主键、日期、行情逻辑、缺失和有限值检查。 |
| `src/data/label_audit.py` | 标签重算与边界审计；负向 shift 只存在于该审计域。 |
| `src/features/registry.py` | 特征定义、类别、方向、窗口和 allowlist。 |
| `src/features/time_series.py` | 逐股票基础因果时序特征。 |
| `src/features/market.py` | 每交易日一行的市场聚合特征。 |
| `src/features/cross_sectional.py` | 同日 rank/z-score、缺失和常数截面处理。 |
| `src/features/pipeline.py` | Phase 3 基础特征分块编排与审计。 |
| `src/features/storage.py` | 分年 Parquet、manifest、schema 与键哈希。 |
| `src/features/advanced.py` | 高级技术/量价特征纯计算函数。 |
| `src/features/phase13.py` | 高级特征训练/测试构建和 warm-up 编排。 |
| `src/features/screening.py` | 同日 IC、时间稳定性、相关性去冗余和家族消融。 |
| `src/features/inference_pipeline.py` | 四个生产模型的测试特征装配和 schema 校验。 |
| `src/validation/splitter.py` | expanding walk-forward 日历与 purge。 |
| `src/validation/evaluator.py` | Rank IC、Top10% 年化超额、换手和 0.4/0.3/0.3 综合分复刻。 |
| `src/validation/parity.py` | 本地/官方评价器输入标准化和压力场景比较。 |

### 模型、后处理、实验与工具

| 文件 | 作用 |
|---|---|
| `src/models/dataset.py` | 分年读取、真实标签连接、fold 内预处理和矩阵装配。 |
| `src/models/baselines.py` | Ridge/LightGBM 训练、预测、保存和 OOF。 |
| `src/models/ensemble.py` | OOF 键对齐、日内排名、加权和融合评分。 |
| `src/models/ranking.py` | relevance、日期 group、LambdaRank 与 OOF。 |
| `src/models/stability.py` | ATR 紧凑视图、训练窗口和状态比较。 |
| `src/models/finalization.py` | 三组件网格、端点复刻、最终选择和全量生产训练。 |
| `src/models/inference.py` | 模型载入、分批预测、嵌套融合和跨边界状态。 |
| `src/postprocess/turnover_control.py` | 因果 EMA、资格过滤和精确 Top 集合滞回。 |
| `src/postprocess/phase10.py` | Phase 10 网格、逐折评分和产物编排。 |
| `src/experiments/registry.py` | 源码/配置/产物哈希、不可变快照和 append-only 注册表。 |
| `src/submission/validator.py` | 列、dtype、代码/日期、键、行序、有限值和回读校验。 |
| `src/utils/config.py` / `paths.py` | YAML 配置验证 / 根目录与 D 盘路径解析。 |
| `src/utils/logging.py` / `seed.py` | 统一日志 / 随机种子。 |
| 各目录 `__init__.py` | 声明 Python 包边界。 |

## 5. `tests/`：测试文件职责

| 文件 | 覆盖内容 |
|---|---|
| `test_config_and_paths.py` | 配置和 D 盘路径约束。 |
| `test_loader.py` / `test_validator.py` | 加载、缓存身份 / 数据契约。 |
| `test_label_audit.py` | 标签定义、边界和末行缺失。 |
| `test_features.py` / `test_advanced_features.py` | 基础 / 高级特征公式与因果性。 |
| `test_feature_storage.py` | 分区、manifest、键和列契约。 |
| `test_cross_sectional.py` | 同日 rank/z-score 和异常截面。 |
| `test_feature_screening.py` | 预筛选边界、方向、覆盖和去冗余。 |
| `test_model_dataset.py` | fold 装配、标签连接、allowlist 和预处理隔离。 |
| `test_splitter.py` | expanding folds、purge 和不重叠。 |
| `test_evaluator.py` | 官方八项指标、边界输入和 parity。 |
| `test_baselines.py` / `test_ranking.py` | baseline / LambdaRank 接口和数据契约。 |
| `test_turnover_control.py` | EMA、Top 大小、涨停排除、状态连续性与因果性。 |
| `test_ensemble.py` | OOF 对齐、日内排名、权重和端点。 |
| `test_experiment_registry.py` | 哈希、快照、幂等追加和冲突拒绝。 |
| `test_stability.py` | 训练窗口、ATR 视图、状态阈值和恢复。 |
| `test_finalization.py` | 单纯形、端点、最终选择和生产 manifest。 |
| `test_inference_features.py` / `test_inference.py` | 测试特征、四模型推理和状态传递。 |
| `test_submission.py` | 三列格式、原始键/行序、回读和不覆盖。 |

## 6. 数据与产物目录

- `data/raw/`：官方训练/测试 CSV，只读；`data/cache/`：身份保护的 Parquet 缓存；`data/processed/`：按年分区的因果特征和模型视图。
- `models/phase8_baselines`、`phase10_turnover`、`phase11_ensemble`、`phase12_lambdarank`、`phase15_stability`：各阶段模型和 OOF。
- `models/phase16_final/`：最终四模型、融合 manifest 和 OOF；`models/phase17_prediction/`：无标签测试预测。
- `reports/tables/`：阶段审计与小型比较表；`reports/figures/phase19/`：报告图；`reports/final_report.md`：文字版。
- `output/pdf/`：8 页最终报告；`submissions/`：最终 CSV，默认拒绝同名覆盖。
- `experiments/configs/`：不可变配置快照；`experiments/manifests/`：源码/产物哈希；`experiment_registry.csv`：联合键 append-only 注册表。
- 分年目录内的大量 Parquet 具有相同职责，无需逐个操作；完整列表、行数和哈希由相邻 manifest 管理。

## 7. 如何进行测试

### 7.1 快速、只读、建议先跑

```powershell
Set-Location D:\codex\fintechathon2026
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

该命令不重训模型。当前基线为 64 项通过；以后若增加测试，以零失败为准。

### 7.2 数据、时间切分和评价器专项检查

```powershell
.\.venv\Scripts\python.exe scripts\01_check_data.py
.\.venv\Scripts\python.exe scripts\02_audit_labels.py
.\.venv\Scripts\python.exe scripts\06_validate.py
.\.venv\Scripts\python.exe scripts\07_check_evaluator.py
```

应确认原始哈希不变、主键唯一、标签来自真实训练集、每折有一个交易日 purge、本地/官方指标最大差不超过 `1e-12`。

### 7.3 推理与 submission 验收

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_inference tests.test_submission -v
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('reports/tables/phase18_submission_audit.json').read_text(encoding='utf-8')); print(json.dumps(a['contracts'],ensure_ascii=False,indent=2)); print(a['submission'])"
```

检查三列 schema、1,599,600 行、原始键/行序、有限预测和 float32 回读一致。不要用 Excel 保存最终 CSV，以免改写股票代码、日期或浮点格式。

### 7.4 报告验收

```powershell
.\.venv\Scripts\python.exe -c "from pypdf import PdfReader; p=PdfReader('output/pdf/fintechathon2026_task5_report.pdf'); print('pages=',len(p.pages)); print('text_chars=',sum(len(x.extract_text() or '') for x in p.pages))"
```

页数必须为 8；再人工检查字体、图表、页码、表格和免责声明。报告中的 `0.3818` 是 walk-forward OOF 均分，不是隐藏测试分。

### 7.5 从零重跑

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

确认两个官方 CSV 位于 `data/raw/` 后，按编号依次执行 `01` 至 `19`。Phase 8、12、15、16 耗时和空间较大；普通验收不要删除已有模型/Parquet。稳定产物默认拒绝覆盖，只在确认重建该 phase 时使用脚本支持的显式 `--overwrite`；Phase 15 完整折恢复可用 `--resume`。

失败时依次检查：当前目录和解释器；原始 SHA-256；对应 `phase*_audit.json`；输入 manifest/键哈希/allowlist；只从最早失效 phase 重建下游。始终禁止改原始 CSV、跳过审计或用测试集造标签。

## 8. 最终交付

- `submissions/submission_p16_locked_ensemble_v1.csv`
- `output/pdf/fintechathon2026_task5_report.pdf`
- `models/phase17_prediction/test_predictions.parquet`
- `models/phase16_final/`
- `REPRODUCIBILITY.md`
- `FINAL_DELIVERY_MANIFEST.json`
- `reports/tables/phase19_delivery_audit.json`

