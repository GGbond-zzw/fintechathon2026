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
- 一次只实施一个 Phase；完成后记录验收证据再进入下一 Phase。

