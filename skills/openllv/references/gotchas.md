# 已核实的边界行为与参数陷阱

每条都来自源码阅读。依据里的源码路径相对仓库根的 `openLLV/` 包目录；仓库级文件（`README.md`、`docs/`、`.agents/`）按仓库根路径书写。遇到与本文冲突的行为，先复查源码。

## 训练与通用参数

### 1. `best.pt` 不是“最佳 epoch”

每个 epoch 结束时 `last.pt` 和 `best.pt` 都会无条件重写；`best_val_loss` 只记录在返回值、history 和 checkpoint 字段里。`best.pt` 实际是最近一个 epoch 的那份。

处理：需要最佳权重时从 `history` 找最小 `val_loss` 对应的 epoch，或用 `epoch_<n>.pt`（需 `save_every>0`），或自己按 `best_val_loss` 另存。依据：`deepLearning/trainer.py:1378-1387`。

### 2. 训练目录里的 `<Model>.yaml` 不能直接喂回 `llv.train`

它是运行记录，结构是 `{model_name, dataset_name, training_started_at, training_ended_at, config: {...}}`，不是训练配置本身。直接加载会因缺少顶层 `model.name` 报错。

处理：续训用 `resume=.../last.pt`；要复用配置就自己保存 `config` 字段或原始 YAML。依据：`deepLearning/trainer.py:1679-1707`。

### 3. 有验证集才会验证；没有就设 `val_split=None`

`Trainer` 默认 `val_split="val"`，内置配置各不相同（有 `val`、`_test`、`eval`、`null`）。数据集目录不存在时直接 `FileNotFoundError`，不会自动跳过。

处理：只有训练目录时传 `val_split=None`；或提供 `val_input_dir` / `val_target_dir`。依据：`deepLearning/trainer.py:769-816`、`deepLearning/config/*.yaml`。

### 4. 公开 worker 参数名是 `workers`，不是 `num_workers`

`Predictor`、深度后端 `Predictor`、`Trainer` 平铺键、`Evaluator` 的公开参数都叫 `workers`；`num_workers` 只是内部构造 DataLoader 时的实参名。

写错位置的后果不同：`llv.train(..., num_workers=4)` 抛 `TypeError`；`llv.predict(..., num_workers=4)` 在深度后端会被并入模型 `config`，静默无效。调用时统一写 `workers=...`。

依据：`predictor.py:58`、`deepLearning/predictor.py:186`、`deepLearning/trainer.py:265`、`evaluation/evaluator.py:28`。

### 5. 内置配置的 `root_dir` 是占位符

`openLLV/deepLearning/config/*.yaml` 里写的是 `path/to/your/dataset`。不覆盖就会在数据集校验时报路径不存在。

处理：始终传 `root_dir=...`。依据：`deepLearning/config/ZeroDCE.yaml:6`。

### 6. `workers>0` 时要保护训练入口

DataLoader 在 spawn 平台会重新导入主模块；没有 `if __name__ == "__main__":` 会递归启动。依据：`README.md:490-503` 与 Python DataLoader 行为。

## 预测

### 7. 注册名预测 != 预训练权重

仓库不附带预训练权重。`llv.predict("ZeroDCE", ...)` 用随机初始化网络，结果没有视觉意义。

处理：先训练，或换成 `llv.predict(".../best.pt", ...)`。依据：`README.md:113-116`、`examples/03_deep_enhance.py:26`。

### 8. `ZeroDCE++` 是配置名，不是模型名

模型注册名是 `ZeroDCEPlusPlus`（无别名 `ZeroDCE++`）。`llv.train("ZeroDCE++")` 能用，是配置解析器做了标点归一化；`llv.predict("ZeroDCE++")` 会报未注册。

依据：`deepLearning/models/LLIE/ZeroDCEPlusPlus.py:16-18`、`deepLearning/config/__init__.py:204-209`。

### 9. 传统后端的未知参数只警告

`LLVEnhancer.create_enhancer` 会过滤不支持的构造参数，在 stderr 打印“不会使用”和拼写建议，然后继续运行。参数拼错不会失败。

处理：看 stderr 的 `Parameters:` 输出或 `Predictor.get_params()["enhancer"]` 核对实际值。依据：`tradition/algorithms/BaseModel.py:168-195`。

### 10. 深度后端的未知 kwargs 静默进入模型 config

`Predictor.__init__` 把剩余 kwargs 合并进 `config`，`LLVModel` 再合并进 `self.config`；模型不读的键不会报错。

处理：参数名从模型源码或 `docs/models/...` 核对；`forward` 的键必须放 `model_kwargs`。依据：`predictor.py:98-115`、`deepLearning/models/BaseModel.py:157-172`。

### 11. `device` 对传统后端无效，`resize` 会报错

传统后端没有设备概念，`device` 会被参数过滤器丢掉并告警；`resize` 在 `Predictor` 层直接对传统后端抛 `ValueError`。

依据：`predictor.py:116-131`、`tradition/predictor.py:272-307`。

### 12. 目录输入的 `output` 一律当目录

即使写成 `output="results/out.png"`，目录预测也会创建名为 `out.png` 的目录。`output_name` 对目录输入直接抛 `ValueError`。

依据：`deepLearning/predictor.py:364-372`、`tradition/predictor.py:178-185`。

### 13. `save=False` 不创建任何目录

目录预测 `save=False` 时返回图像列表，输入/输出提示中的输出路径显示 `Not saved (save=False)`，磁盘不变。依据：`deepLearning/predictor.py:392-397,461-470`。

### 14. 单图显式文件路径的 `output_ext` 行为不一致

深度后端会把 `output_ext` 应用到显式文件路径；传统后端对显式文件路径原样使用（只改目录模式下的文件名）。依据：`deepLearning/predictor.py:762-786`、`tradition/predictor.py:341-375`。

### 15. `image_range` 会改变预测前后处理

`image_range="minus_one_one"` 时预测器先把输入映射到 `[-1,1]`，输出再映回 `[0,1]`。`LEDNet` 模型默认是 `zero_one`，但内置 YAML 设成 `minus_one_one`；按名字预测和按配置训练可能走不同值域。

依据：`deepLearning/predictor.py:610-622`、`deepLearning/models/LLIE/LEDNet.py:426`、`deepLearning/config/LEDNet.yaml:8`。

## 评估

### 16. 不传 `save_path` 也会写文件

`evaluate` 始终调用 `save_results`；`save_path=None` 时写到 `./results/eval.json`。只读评估也会产生磁盘副作用。依据：`evaluation/evaluator.py:374-402`。

### 17. 给了参考目录就会丢掉未配对图像

`EvaluateDataset` 在提供参考图时只保留 stem 能配对的增强图；未配对的只发警告并从 `dataset` 中移除。之后**所有**指标（包括无参考指标）都只在这批子集上计算。

处理：先核对 `len(dataset)` / `results["filenames"]` 与输出文件数是否一致。依据：`data/coreDataset.py:153-176`。

### 18. 参考图尺寸会改变评估输入

全参考指标计算前，`BaseMetric._prepare_inputs` 会把增强图双线性缩放到参考图尺寸；两者尺寸不同时，指标不是按原尺寸算的。

处理：对齐输出与参考图尺寸，或确认这种缩放可接受。依据：`evaluation/baseMetric.py:80-100`。

### 19. 指标 kwargs 会发给所有指标

`Evaluator` 把 `**kwargs` 传给每个 metric 构造器。给某个指标传专属参数，其他指标也可能收到；构造失败会被捕获并跳过（只警告）。

依据：`evaluation/evaluator.py:61-80`。

## 图像 I/O

### 20. `imwrite` 写入 `[0,1]` 浮点 NumPy 会变黑

NumPy 路径走 `ImageReader` → `_ensure_uint8`，只做 `clip(0,255).astype(uint8)`，**不会**乘 255。`np.float32` 的 0.5 会变成 0。

处理：先转成 uint8（乘 255）或传 torch tensor（tensor 路径会按 `[0,1]` 缩放）。依据：`data/image_io.py:448-455,519-527,759-795`。

### 21. NumPy 输入按 RGB 解释

`ImageReader` 不做 BGR→RGB；OpenCV 读出的 BGR 数组直接传入会红蓝互换。依据：`data/image_io.py:499-513`。

### 22. 单图 tensor 只接受 batch=1

`[1,C,H,W]` 中 batch 必须为 1；更大 batch 抛 `ValueError`。依据：`data/image_io.py:471-481`。

## 组件查询

### 23. `list_available` 与 `list_*` 的粒度不同

`list_available()` 每个类只出现一次并带 `aliases`；`list_models()` / `list_algorithms()` 等返回**所有**查询键（含别名）。做存在性判断用 `list_*`，展示组件用 `list_available`。依据：`api.py:239-285`。
