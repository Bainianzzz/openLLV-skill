---
name: openllv
description: 使用 openLLV（Python/PyTorch 低层视觉工具包）的 Python SDK：图像增强与复原的预测、训练、评估与图像 I/O，以及排查参数、输出路径、值域与 checkpoint 问题。任务涉及 openLLV 仓库，或代码中出现 `import openLLV` 时使用。
license: MIT
metadata:
  version: "0.1.0"
---

# openLLV 使用手册

**仓库根目录** = 含 `pyproject.toml` 和 `openLLV/` 包目录的目录。本文命令都在根目录执行。

权威顺序：`openLLV/**/*.py` > `docs/`、`README.md`、`.agents/` 里的旧说明。

内容分工：契约与签名在 `references/api-contracts.md`，名称与指标特性在 `references/catalog.md`，边界行为与陷阱在 `references/gotchas.md`。每条内容只写在其中一处。

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
| 查已注册名称与别名 | `llv.list_available()` | `references/catalog.md` |
| 行为与预期不符 | — | `references/gotchas.md` |

## 2. 预测

1. 确认目标在注册表里：

```python
import openLLV as llv
print(llv.list_models())       # 深度模型
print(llv.list_algorithms())   # 传统算法
```

2. 单图：

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

3. 目录。递归处理，返回列表，顺序固定。

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

5. 参数放哪：`llv.predict` 的键分三层——调用键、`Predictor` 键、其余按后端进模型 `config` 或算法构造器。完整键表与 backend 推断规则见 `references/api-contracts.md` §`predict` 的 kwargs 路由；放错层的后果见 `references/gotchas.md` #9—#11。

**完成判据**：单图返回二元组，`save=True` 时路径指向已存在文件、`save=False` 时路径为 `None`；目录返回路径数量等于输入支持图片数量，或 `save=False` 时返回等长图像列表且没有写盘。

## 3. 训练

1. 准备配对数据（`CommonDataset` 首选布局）：

```text
dataset_root/
  train/input/*.png
  train/target/*.png
  val/input/*.png
  val/target/*.png
```

2. 查内置配置名（清单见 `references/catalog.md`）：

```bash
python -c "from openLLV.deepLearning.config import list_available_configs; print(list_available_configs())"
```

3. 启动训练：

```python
import openLLV as llv

if __name__ == "__main__":
    result = llv.train(
        "ZeroDCE",
        root_dir="datasets/my_dataset",
        epochs=10, batch_size=4, device="cpu",
    )
    print(result["history"], result["checkpoint_dir"])
```

4. 续训：`resume=".../last.pt"`（或 `resume_path=`）。产物目录与文件清单见 `references/api-contracts.md` §训练。

**完成判据**：`result["history"]` 非空，且 `result["checkpoint_dir"]/last.pt` 存在。

关键点：

- 需微调的层用 `model_params` 传模型构造键；设备只由 `device` 控制，不要塞进 `model_params`。
- 监督损失需要成对的 `target` 图；数据集中缺 target 时会报错。
- 三个训练期高频坑：验证集、数据集根、入口保护。见 `references/gotchas.md` #3、#5、#6。

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

- 指标方向、是否需要参考图、外部依赖见 `references/catalog.md` §评估指标。
- 配对规则、结果结构见 `references/api-contracts.md` §评估。
- 评估的已知副作用见 `references/gotchas.md` #16—#19。

**完成判据**：`results["statistics"]` 含请求的指标，且全参考指标的 `valid_count > 0`。

## 5. 图像 I/O

```python
import openLLV as llv

pil = llv.imread("Inputs/input.jpg")                  # RGB PIL
arr = llv.imread("Inputs/input.jpg", output_format="numpy")  # uint8 RGB
path = llv.imwrite(arr, "results/copy.png")           # -> Path
```

- 输入类型、`output_format` 取值、`imwrite` 路径规则见 `references/api-contracts.md` §图像 I/O。
- 值域与通道顺序的陷阱见 `references/gotchas.md` #20—#22。

## 6. 排错顺序

1. 跑 §0 的导入检查。
2. 用 `llv.list_*()` 确认名字确实注册。
3. 对照 `references/gotchas.md` 的已知边界。
4. 跑最小复现：`python -m pytest -q test/<相关文件>`。

## 参考文件

- `references/api-contracts.md` — 公开签名、默认值、返回契约、kwargs 路由、训练平铺键与产物、评估配对与结果结构、图像 I/O 格式与路径规则。
- `references/catalog.md` — 已注册模型/算法/指标/损失/数据集、别名与参数文档入口、指标方向与依赖、配置名匹配规则。
- `references/gotchas.md` — 23 条已核实的边界行为、默认副作用与参数命名陷阱；同类内容只在此处展开。
