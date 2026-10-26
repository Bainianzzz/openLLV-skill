# openLLV 公开契约与 kwargs 路由

依据：openLLV 0.1.0 源码（`openLLV/api.py`、`predictor.py`、`deepLearning/predictor.py`、`tradition/predictor.py`、`deepLearning/trainer.py`、`evaluation/evaluator.py`、`data/image_io.py`）。源码变动后以源码为准。

## 顶层 API

```text
predict(method, source, output=None, **kwargs)
enhance(method, source, output=None, **kwargs)          # predict 别名
train(config=None, **kwargs) -> dict
evaluate(en_img_dir=_MISSING, ref_img_dir=_MISSING, metrics=None,
         save_path=None, return_evaluator=False, *,
         en=_MISSING, ref=_MISSING, **kwargs)
eval(*args, **kwargs)                                   # evaluate 别名
imread(source, output_format="pil", **kwargs)
imwrite(image, output=None, *, save_format=None, output_name=None, **kwargs) -> Path
read_image / write_image                                # imread / imwrite 别名
list_models() / list_algorithms() / list_metrics()
list_losses() / list_datasets() -> list[str]
list_available() -> dict[str, list[{"name": str, "aliases": list[str]}]]
```

`list_*` 返回**每个可接受查询键**（含别名，已转小写并排序）；`list_available` 每个实现类只出现一次。

## 预测返回契约

| 输入 | 后端 | 返回值 |
| --- | --- | --- |
| 单图 | 深度 | `(PIL.Image RGB, Path 或 None)` |
| 单图 | 传统 | `(np.ndarray RGB, Path 或 None)` |
| 目录 | 深度 | `save=True`：`list[Path]`；`save=False`：`list[PIL.Image]` |
| 目录 | 传统 | `save=True`：`list[Path]`；`save=False`：`list[np.ndarray RGB]` |
| 空目录 | 任意 | `[]` |

单图 `save=False` 时路径为 `None`。目录 `save=False` 不创建任何输出目录，列表顺序按源文件路径小写排序。

## `predict` 的 kwargs 路由

`openLLV/api.py` 的 `_PREDICT_CALL_KWARGS`：

```text
progress_bar, output_name, output_ext, save, model_kwargs,
ext, timeout, headers, verify_ssl, cancel
```

这些键进入后端 `predict_single` / `predict_batch`；其余键全部进入 `Predictor(...)`：

```text
Predictor(target=None, *, model=None, method=None, backend="auto",
          output_dir=None, config=None, device=None, transform=None,
          resize=None, batch_size=1, workers=0, **kwargs)
```

- `target` 与 `model` / `method` 互斥；`model` 与 `method` 也互斥。
- `backend`：`"auto"`、深度别名（`deep`、`deeplearning`、`deep_learning`、`dl`、`model`）、传统别名（`tradition`、`traditional`、`traditionalalgorithm`、`traditional_algorithm`、`ta`、`method`、`algorithm`）。大小写和首尾空白不敏感。
- `auto` 推断：`.pt` / `.pth` → 深度；`LLVModel` 实例 → 深度；`LLVEnhancer` 实例 → 传统；注册名只在一边 → 对应后端；两边都有 → `ValueError`。
- `config` 必须是 `dict` 或 `None`。
- `resize` 只属于深度后端；传统后端收到非 `None` 值抛 `ValueError`。
- `**kwargs`：深度后端合并进模型配置（kwargs 覆盖 `config`）；传统后端合并进算法构造参数。

### 深度后端

```python
DeepPredictor(model, output_dir=None, config=None, device=None,
              transform=None, resize=None, batch_size=1, workers=0)
```

- `model`：注册名、`.pt` / `.pth` checkpoint、`LLVModel` 实例。
- `device=None` → CUDA 可用则 CUDA，否则 CPU（不尝试 MPS）。
- `transform`：可调用对象或可调用对象列表；默认 `ToImage + ToFloat`，不做缩放。`resize` 在 transform 之前执行。
- `resize`：`None` 保持原尺寸；正整数为正方形；二元 tuple/list 为 `(height, width)`。
- `batch_size`：目录输入按图像尺寸分组；只有完整同尺寸组一次前向，余数逐图前向，不 padding。
- `workers`：DataLoader 读取/预处理进程数；spawn 平台下 `workers > 0` 时自定义 transform 必须可 pickle。
- `config` 对 checkpoint 是配置覆盖；checkpoint 以 `strict=True` 重建。
- 输出约定：模型返回 tensor，或含 `pred` 的 dict，或 tuple/list（取最后一个 tensor）；单图输出 1/3/4 通道，值域 `[0,1]`。配置 `image_range="minus_one_one"` 时预测器自动把输入映射到 `[-1,1]`、输出映射回 `[0,1]`。

### 传统后端

```python
TraditionalPredictor(method="he", output_dir=None, config=None, **kwargs)
```

- `method`：注册名或 `LLVEnhancer` 实例；实例会被强制 `output_type="numpy"`。
- `config` + `**kwargs` 合并后传给 `LLVEnhancer.create_enhancer`。
- 算法基类每次预测都会收到 `output_type="numpy"`。
- 不支持的参数被过滤，并在 stderr 打印“不会使用 + 拼写建议”；不会抛错。
- `value_range`：`"auto"`（默认）、`"unit"`、`"byte"` 或 `(min, max)`。`auto` 对 uint8 用 `[0,255]`，对浮点用 `[0,1]`（最大值 ≤ 1）或 `[0,255]`（最大值 ≤ 255）。

### 单图输出路径规则

- `output` 不存在且带后缀 → 文件路径。
- `output` 存在且是文件 → 文件路径。
- 其他情况（存在目录、不存在且无后缀）→ 目录，文件名取 `output_name` 或源文件名。
- `output=None` → `results/<模型类名>/` 或 `results/<算法 name 小写>/`。
- 深度后端会把 `output_ext` 应用到显式文件路径；传统后端对显式文件路径原样使用。

### 目录输出规则

- `output` 一律当输出根目录。
- 保留源目录相对结构和文件名；`output_ext` 给所有输出换后缀；`output_name` 对目录输入抛 `ValueError`。
- 支持后缀：`.jpg .jpeg .png .bmp .tiff .tif .webp .gif .ppm .pgm .pbm .sr .ras`。

### 取消

```python
from openLLV import CancelSignal
signal = CancelSignal()
# 另一个线程调用 signal.cancel()
llv.predict(..., cancel=signal)
```

目录预测在下一次图像/批次边界抛 `TaskCancelled`；已保存文件保留。

## 训练

```python
Trainer(config=None, **kwargs)
```

`config`：`None`、内置配置名、YAML 路径、嵌套 dict。内置名匹配忽略大小写和标点（`"ZeroDCEPlusPlus"` 命中 `ZeroDCE++.yaml`）。

`**kwargs` 走 `Trainer._kwargs_to_config`，未知键抛 `TypeError`。平铺键：

| 分组 | 键 |
| --- | --- |
| 模型 | `model`/`model_name`、`model_params`、`model_config` |
| 数据 | `dataset`/`dataset_name`、`root_dir`、`batch_size`、`workers`、`pin_memory`、`shuffle`、`drop_last`、`train_split`、`val_split`、`return_filename`、`resize`、`train_input_dir`、`train_target_dir`、`val_input_dir`、`val_target_dir`、`data_params`、`train_params`、`val_params` |
| 损失 | `loss`/`loss_name`、`loss_params`、`output_index`、`output_key` |
| 优化器 | `optimizer`/`optimizer_name`、`lr`、`optimizer_params` |
| 调度器 | `scheduler`/`scheduler_name`、`scheduler_params` |
| 训练 | `epochs`、`output_dir`、`save_every`、`validate_every`、`log_every`、`grad_clip`、`amp`、`resume`/`resume_path`、`strict_resume`、`seed`、`cudnn_deterministic`、`cudnn_benchmark`、`device`、`device_ids`、`progress_bar` |
| 整段覆盖 | `data`、`model`、`loss`、`optimizer`、`scheduler`、`train`（传 dict 时递归合并） |

注意：公开键是 `workers`，不是 `num_workers`。

默认值（通用 Trainer，未叠加内置 YAML）：`batch_size=4`、`workers=0`、`epochs=100`、`lr=1e-4`、`optimizer=adam`、`scheduler=None`、`seed=42`、`amp=False`、`validate_every=1`、`log_every=10`、`save_every=0`。

默认输出：`checkpoints/<Model>_<Dataset>/`，其中 `checkpoints/{last.pt,best.pt}`、`logs/history.json`、`<Model>.yaml`。

`train()` 返回：

```python
{"history": [...], "best_val_loss": float, "checkpoint_dir": str}
```

被取消时额外含 `"stopped": True`、`"stop_epoch": int`。

## 评估

```python
Evaluator(en_img_dir, ref_img_dir=None, save_path=None, metrics=None,
          device=None, batch_size=1, workers=8, **kwargs)
```

- 构造时立即执行评估，结果存 `evaluator.results`。
- `metrics=None` → `["PSNR", "SSIM"]`；指标名大小写不敏感，未知指标警告后跳过。
- `**kwargs` 会传给**每个**指标构造器（如 `data_range`、`window_size`）。
- 缺参考图时，全参考指标每个文件记 `NaN`。
- `save_path=None` → `./results/eval.json`。
- `en` / `ref` 是 `en_img_dir` / `ref_img_dir` 的别名；同一参数两种写法同时出现抛 `TypeError`。
- `return_evaluator=True` 返回已跑完评估的 `Evaluator`，否则返回结果字典。

结果：

```python
{
  "filenames": ["0001.png"],
  "metrics": {"PSNR": {"0001.png": 21.3}},
  "statistics": {"PSNR": {
      "mean": 21.3, "std": 0.4, "min": 20.1, "max": 22.0,
      "valid_count": 1, "total_count": 1, "better": "↑",
  }},
}
```

被取消时结果额外含 `"cancelled": True` 和已完成部分。

## 图像 I/O

```text
ImageReader()(input_data, output_format="pil", **reader_kwargs)
# reader_kwargs: ext, timeout, headers(dict), verify_ssl
ImageWriter(output_dir="results")(image, output=None, *,
                                  save_format=None, output_name=None,
                                  **reader_kwargs)
```

- 输出格式：`pil`（RGB）、`numpy`（RGB uint8）、`bytes`、`base64`、`file`（临时文件路径字符串）。
- 输入：路径、URL、base64、bytes/bytearray、NumPy、PIL、单图 tensor（`[H,W]`、`[C,H,W]`、`[1,C,H,W]`）。
- NumPy 输入按 **RGB** 解释，不做 BGR→RGB 转换。
- `imwrite` 返回最终 `Path`；JPG 输出遇到非 RGB/L 模式会先转 RGB。
- bytes/base64 输出在源无后缀时用 `ext` 指定格式。
