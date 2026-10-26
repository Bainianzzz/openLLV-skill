---
name: openllv
description: 使用或扩展 openLLV（Python/PyTorch 低层视觉工具包）的 Python SDK：图像增强与复原的预测、训练、评估、图像 I/O，新增 LLVModel / LLVEnhancer / BaseLoss / BaseDataset / BaseMetric 组件，以及排查组件注册、输出路径、值域与 checkpoint 问题。任务涉及 openLLV 仓库，或代码中出现 `import openLLV` 时使用。
license: MIT
metadata:
  version: "0.1.0"
---

# openLLV 使用手册

**仓库根目录** = 含 `pyproject.toml` 和 `openLLV/` 包目录的目录。本文命令都在根目录执行。

权威顺序：`openLLV/**/*.py` > `docs/`、`README.md`、`.agents/` 里的旧说明。已知边界行为见 `references/gotchas.md`。

## 0. 先确认能导入

```bash
python -c "import openLLV as llv; print(len(llv.list_models()), len(llv.list_algorithms()))"
```

- 打印两个数字 → 环境可用，继续。
- `ModuleNotFoundError: openLLV` → `python -m pip install -e ".[dev]"`（要求 Python ≥ 3.8、torch ≥ 2.0）。
- `ModuleNotFoundError: torch` → 先装 PyTorch；不要在缺 torch 的环境里判断代码有 bug。

## 1. 选路线

| 要做的事 | 入口 | 读哪里 |
| --- | --- | --- |
| 单图或目录增强/复原 | `llv.predict`、`llv.enhance` | §2；参数路由见 `references/api-contracts.md` |
| 训练模型 | `llv.train` | §3 |
| 评估输出目录 | `llv.evaluate` | §4 |
| 图像读取、保存、格式转换 | `llv.imread`、`llv.imwrite` | §5 |
| 新增模型、算法、损失、数据集、指标 | 对应基类 | `references/extending.md` |
| 查已注册名称与别名 | `llv.list_available()` | `references/catalog.md` |
| 行为与预期不符 | — | `references/gotchas.md` |

## 2. 预测

1. 确认目标在注册表里：

```python
import openLLV as llv
print(llv.list_models())       # 深度模型
print(llv.list_algorithms())   # 传统算法
```

2. 单图。返回值是 `(图像, 保存路径)`；`save=False` 时路径为 `None`。

```python
import openLLV as llv

# 传统算法：参数直接平铺
enhanced, path = llv.predict(
    "Gamma", "Inputs/input.jpg",
    output="results/gamma.png", gamma=0.8,
)

# 深度模型：构造参数平铺或放 config，前向参数放 model_kwargs
enhanced, path = llv.predict(
    "ZeroDCE", "Inputs/input.jpg",
    output="results/zero_dce.png", device="cpu",
)
```

3. 目录。递归处理，保留相对子目录和文件名；返回列表，顺序固定。

```python
saved_paths = llv.predict(
    "CLAHE", "Inputs/", output="results/clahe",
    clip_limit=2.0, tile_grid_size=(8, 8), progress_bar=False,
)
```

4. 用训练 checkpoint 代替模型名：

```python
enhanced, path = llv.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/last.pt",
    "Inputs/input.jpg", output="results/from_ckpt.png", device="cpu",
)
```

**完成判据**：单图返回二元组，`save=True` 时路径指向已存在文件、`save=False` 时路径为 `None`；目录返回路径数量等于输入支持图片数量，或 `save=False` 时返回等长图像列表且没有写盘。

### 参数放哪（最常错）

`llv.predict(name, source, output=None, **kwargs)` 先拆 kwargs：

- 调用参数（其余全进 `Predictor`）：`progress_bar`、`output_name`、`output_ext`、`save`、`model_kwargs`、`ext`、`timeout`、`headers`、`verify_ssl`、`cancel`。
- `Predictor` 参数：`backend`、`device`、`output_dir`、`config`、`transform`、`resize`、`batch_size`、`workers`。
- 剩下的键：深度后端进模型 `config`；传统后端进算法构造器。

硬约束：

- 名字同时存在于两个注册表 → `ValueError`，显式写 `backend="deep"` 或 `backend="traditional"`。
- 传统后端不接受 `resize`（抛 `ValueError`），也不使用 `device`（只打印警告）。
- 传统后端收到不存在的参数只警告、不报错；参数拼错会静默失效。
- 深度后端的未知平铺参数会进入模型 `config`；模型不读取就没有效果。

## 3. 训练

1. 准备配对数据（`CommonDataset` 首选布局）：

```text
dataset_root/
  train/input/*.png
  train/target/*.png
  val/input/*.png
  val/target/*.png
```

2. 选配置名（内置 17 个，见 `references/catalog.md`）：

```bash
python -c "from openLLV.deepLearning.config import list_available_configs; print(list_available_configs())"
```

3. 启动训练：

```python
import openLLV as llv

if __name__ == "__main__":   # workers > 0 时必须保留这层保护
    result = llv.train(
        "ZeroDCE",
        root_dir="datasets/my_dataset",
        epochs=10, batch_size=4, device="cpu",
    )
    print(result["history"], result["checkpoint_dir"])
```

4. 检查产物，默认写到 `checkpoints/<Model>_<Dataset>/`：

```text
checkpoints/ZeroDCE_CommonDataset/
  checkpoints/last.pt
  checkpoints/best.pt
  logs/history.json
  ZeroDCE.yaml
```

5. 续训：`resume=".../last.pt"`（或 `resume_path=`）。

**完成判据**：`result["history"]` 非空，且 `result["checkpoint_dir"]/last.pt` 存在。

关键点：

- `root_dir` 必填；内置 YAML 里的 `path/to/your/dataset` 是占位符。
- 没有验证集时显式传 `val_split=None`，否则会去解析验证目录并报路径错误。
- 未知平铺键抛 `TypeError`，不会静默忽略；完整键表见 `references/api-contracts.md`。
- 需微调的层用 `model_params` 传模型构造键；设备只由 `device` 控制，不要塞进 `model_params`。
- 监督损失需要 `target`；无参考损失需设 `requires_target = False`（见 `references/extending.md`）。

## 4. 评估

```python
import openLLV as llv

results = llv.evaluate(
    en_img_dir="results/zero_dce",
    ref_img_dir="datasets/my_dataset/val/target",
    metrics=["PSNR", "SSIM"],
    save_path="results/eval.json",
    device="cpu", batch_size=1, workers=0,
)
print(results["statistics"]["PSNR"]["mean"])
```

- 有参考图 → 全参考指标；无 `ref_img_dir` → 全参考指标记 `NaN` 并跳过。
- 按**文件名 stem（不含后缀）小写**配对；扩展名可以不同。
- 结果结构：`{"filenames", "metrics": {指标: {文件名: 值}}, "statistics": {指标: {mean,std,min,max,valid_count,total_count,better}}}`。
- 不传 `save_path` 也会写 `./results/eval.json`。

**完成判据**：`results["statistics"]` 含请求的指标，且全参考指标的 `valid_count > 0`。

指标方向：PSNR、SSIM、MUSIQ 越高越好；MSE、MAE、LPIPS、LOE、NIQE、PI 越低越好。LOE 的参考图应是原始暗光图。LPIPS、NIQE、MUSIQ、PI 依赖 `pyiqa`。

## 5. 图像 I/O

```python
import openLLV as llv

pil = llv.imread("Inputs/input.jpg")                  # RGB PIL
arr = llv.imread("Inputs/input.jpg", output_format="numpy")  # uint8 RGB
path = llv.imwrite(arr, "results/copy.png")           # -> Path
```

- 输入：本地路径、URL、base64、bytes、PIL、NumPy、单图 tensor。
- `output_format`：`pil`、`numpy`、`bytes`、`base64`、`file`。
- 公共三通道数据统一 RGB；OpenCV 的 BGR 只存在于传统算法内部。
- `imwrite` 的 `output` 带后缀按文件处理，不带后缀按目录处理；省略时写 `results/`。

## 6. 扩展组件（最小规则）

- **模型**：继承 `LLVModel`，声明 `task`、`aliases`，实现 `_init_model()` 与 `forward()`；预测输出必须在 `[0,1]`，或配置 `image_range: "minus_one_one"`。在模型所在 task 包的 `__init__.py` 导入后才会注册。
- **传统算法**：继承 `LLVEnhancer`，声明 `name`、`aliases`、`working_range`，实现 `_enhance()`。构造参数必须写进显式签名；`_enhance` 收到 BGR，返回同值域 BGR。
- **损失 / 数据集 / 指标**：同样“导入即注册”。数据集至少实现 `_resolve_pair_dirs()`；指标类名以 `Metric` 结尾；损失按有/无 target 选 `forward` 签名。

完整改动清单、代码模板和测试要求 → `references/extending.md`。

## 7. 排错顺序

1. 跑 §0 的导入检查。
2. 用 `llv.list_*()` 确认名字确实注册。
3. `grep -rn "<类名或参数名>" openLLV/` 定位真实实现。
4. 对照 `references/gotchas.md` 的已知边界。
5. 跑最小复现：`python -m pytest -q test/<相关文件>`。

## 参考文件

- `references/api-contracts.md` — 公开签名、返回契约、kwargs 路由、训练平铺键、结果结构。
- `references/catalog.md` — 已注册模型/算法/指标/损失/数据集、别名、源码与参数文档入口。
- `references/extending.md` — 新增五类组件的改动清单、模板与验证方式。
- `references/gotchas.md` — 已核实的边界行为、默认副作用与参数命名陷阱。
