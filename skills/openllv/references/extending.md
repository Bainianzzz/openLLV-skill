# 扩展 openLLV：五类组件改动清单

所有组件都靠 **`__init_subclass__` + 模块导入** 注册。类定义不会自动生效，必须让模块被导入；内置实现由 `openLLV/__init__.py` 的导入链加载。

注册规则差异（新增名前先查 `references/catalog.md`）：

| 组件 | 注册键 | 重名行为 |
| --- | --- | --- |
| `LLVModel` | 类名 + `aliases` | 抛 `ValueError` |
| `LLVEnhancer` | 类名 + `name` + `aliases` | 静默覆盖 |
| `BaseLoss` | 类名 + `name` + `aliases` | 静默覆盖 |
| `BaseDataset` | 类名 + `name` + `aliases` | 静默覆盖 |
| `BaseMetric` | 仅类名 | 静默覆盖 |

## 1. 新增深度模型（LLVModel）

### 改动

1. 新建 `openLLV/deepLearning/models/<Task>/MyModel.py`，继承 `LLVModel`。
2. 在该 task 包的 `openLLV/deepLearning/models/<Task>/__init__.py` 导入并加入 `__all__`。
3. 如果是新 task 包，再在 `openLLV/deepLearning/models/__init__.py` 导入。
4. 补 `test/deepLearning/test_<model>.py`，至少覆盖构造、前向形状、注册名。
5. 按需补 `docs/models/<task>/<name>.md` 与 `docs-zh-CN/models/<task>/<name>.md`。

### 最小实现

```python
import torch.nn as nn

from openLLV.deepLearning.models import LLVModel


class MyModel(LLVModel):
    task = "llie"
    aliases = ["my-model"]

    def _get_default_config(self):
        config = super()._get_default_config()
        config.update({"hidden_channels": 32, "image_range": "zero_one"})
        return config

    def _validate_config(self):
        super()._validate_config()
        if int(self.config["hidden_channels"]) <= 0:
            raise ValueError("hidden_channels must be positive.")

    def _init_model(self):
        hidden = int(self.config["hidden_channels"])
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, image, **kwargs):
        return self.net(image).clamp(0.0, 1.0)
```

### 契约

- `task` 必须声明；默认配置用它生成 `save_dir`。
- 输入：`[N, C, H, W]` 浮点张量；预测默认值域 `[0,1]`，训练由数据集决定。
- 输出：tensor，或含 `"pred"` 的 dict。预测器对 dict 优先取 `"pred"`，对 tuple/list 取**最后一个** tensor；不要用 `(pred, aux)` 这种把辅助项放最后的顺序。
- 输出通道必须是 1、3、4；`_tensor_to_pil` 会 clamp 到 `[0,1]`。
- 值域为 `[-1,1]` 时在默认配置里写 `"image_range": "minus_one_one"`；预测器会自动做输入/输出映射。
- 设备由 `Predictor` / `Trainer` 管理；不要在模型里保存 device 或 `.to()`。
- 需要配对输入（例如 PairLIE）时声明类属性 `requires_paired_forward = True`，训练器会调用 `model(input, paired_image=target)`。
- 训练期 `config["mode"]` 会在 `"train"` / `"inference"` 间切换；模型可读取它。
- checkpoint 由 `save_model()` 写入，包含 `model_class`、`config`、`model_state_dict`；`llv.predict("x.pt", ...)` 会用 `strict=True` 重建。

### 验证

```bash
python -c "import openLLV as llv; print('MyModel' in llv.list_models(), llv.list_models()[-5:])"
python -c "import openLLV as llv; m=llv.LLVModel.create_model('MyModel'); print(m)"
python -m pytest -q test/deepLearning
```

完成判据：`list_models()` 能查到注册名，构造成功，`forward(torch.rand(1,3,32,32))` 返回 `[1,3,32,32]`。

## 2. 新增传统算法（LLVEnhancer）

### 改动

1. 新建 `openLLV/tradition/algorithms/<Group>/MyAlgo.py`。
2. 在组包 `__init__.py` 导入；新组还要在 `openLLV/tradition/algorithms/__init__.py` 导入。
3. 补 `test/tradition/` 用例，覆盖默认参数、`value_range`、BGR 约定和注册名。
4. 补中英文算法文档（参数逐个从源码核对）。

### 最小实现

```python
import numpy as np

from openLLV.tradition.algorithms import LLVEnhancer


class MyGamma(LLVEnhancer):
    name = "mygamma"
    aliases = ["my-gamma"]
    working_range = "byte"       # 或 "unit"

    def __init__(self, gamma: float = 0.8, **kwargs):
        super().__init__(**kwargs)   # 接收 output_type/keep_dtype/clip_output/value_range
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be a finite positive number.")
        self.gamma = float(gamma)

    def _enhance(self, image, **kwargs):
        table = (
            (np.arange(256, dtype=np.float32) / 255.0) ** self.gamma * 255.0
        ).round().astype(np.uint8)
        return table[image]
```

### 契约

- `_enhance(image, **kwargs)` 收到的是 **BGR** 数组：`working_range="byte"` 为 uint8 `[0,255]`，`"unit"` 为 float32 `[0,1]`；返回同通道顺序、同值域。
- 基类负责：读图、RGB→BGR、值域解析/校验、裁剪、dtype 还原、BGR→RGB、输出格式转换。
- 构造参数必须出现在显式签名里。工厂 `create_enhancer` 只保留显式参数；`**kwargs` 里的未知键会被过滤并在 stderr 告警。
- 基类参数：`output_type`、`keep_dtype`、`clip_output`、`value_range`。经 `Predictor` 调用时 `output_type` 固定为 `"numpy"`。
- 运行期改参数只能用 `set_params(**params)`，且属性必须已存在。
- 算法实例每次 `_enhance` 只处理一张图；目录预测由预测器循环。

### 验证

```bash
python -c "import openLLV as llv; print('mygamma' in llv.list_algorithms())"
python -c "import openLLV as llv; print(llv.predict('mygamma','Inputs/input.jpg',output='results/mygamma.png',save=True))"
python -m pytest -q test/tradition
```

完成判据：注册名可查，单图输出为 `(uint8 RGB array, Path)`，保存文件存在。

## 3. 新增损失（BaseLoss）

### 改动

1. 新建 `openLLV/deepLearning/loss/LLIELoss/MyLoss.py`（其他任务建对应子包）。
2. 在 `LLIELoss/__init__.py` 导入并加入 `__all__`；新任务子包要在 `loss/__init__.py` 导入。
3. 补用例，覆盖有/无 target 两条路径与注册名。

### 有监督损失

```python
import torch

from openLLV.deepLearning.loss import BaseLoss


class MyCharbonnierLoss(BaseLoss):
    name = "mycharbonnier"
    aliases = ["my_charbonnier"]
    requires_target = True

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, prediction, target):
        diff = prediction - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()
```

### 无参考损失

```python
class MyZeroRefLoss(BaseLoss):
    name = "myzeroref"
    requires_target = False

    def forward(self, input_tensor, model_output):
        # 自己从 model_output 取需要的张量；返回标量 tensor
        ...
```

### 契约

- `requires_target=True`（默认）：`forward(prediction, target)`。Trainer 负责从模型输出中抽取 prediction 并对齐尺寸。
- `requires_target=False`：`forward(input_tensor, model_output)`；Trainer 不要求 target。
- 返回标量 tensor；非标量会被 `.mean()`，非有限值抛 `FloatingPointError`。
- 构造参数走 `loss_params`；`output_key` / `output_index` 用于非常规结构输出。
- 未配置损失时按模型类名推断，失败回退 `charbonnier`。

### 验证

```bash
python -c "import openLLV as llv; print('mycharbonnier' in llv.list_losses())"
python -m pytest -q test/deepLearning/test_llie_loss.py test/deepLearning/test_trainer.py
```

完成判据：注册名可查，最小训练 1 个 epoch 不报 loss 错误。

## 4. 新增数据集（BaseDataset）

### 改动

1. 新建 `openLLV/data/datasets/MyDataset.py`。
2. 在 `openLLV/data/datasets/__init__.py` 导入并加入 `__all__`。
3. 补 `test/data/test_datasets.py` 风格用例：配对、缺失 target、`return_filename`。

### 最小实现

```python
from openLLV.data.datasets import BaseDataset


class LowHighDataset(BaseDataset):
    name = "lowhigh"
    aliases = ["low-high"]

    def _resolve_pair_dirs(self, input_dir, target_dir):
        if input_dir is not None:
            return input_dir, target_dir
        split = self.split
        return (
            self.root_dir / split / "low",
            self.root_dir / split / "high",
        )
```

### 契约

- 至少实现 `_resolve_pair_dirs(input_dir, target_dir)`，返回 `(input_dir, target_dir 或 None)`；基类会检查目录存在并配对。
- 默认按文件名 stem（大小写不敏感）配对；target 可以不存在（无参考训练）。
- `__getitem__` 由基类实现，返回 `(input, target, filename)` 或 `(input, target)`。
- 一般不要重写 `__init__`：基类已接受 `root_dir`、`split`、`input_dir`、`target_dir`、`transform_input`、`transform_target`、`common_transform`、`resize`、`return_filename`、`strict_pairing`、`image_extensions`、`mean`、`std`、`crop_size`、`use_flip`、`use_rot`。
- Trainer 会额外传入 `dataset_params` / `train_params` / `val_params` 里的键；这些键必须能被构造器接受。
- 数据集实例只能用于训练 split；用实例时设 `val_split=None`。

### 验证

```bash
python -c "import openLLV as llv; print('lowhigh' in llv.list_datasets())"
python -m pytest -q test/data
```

完成判据：能在 `TinyCommonDataset` 风格目录上取到 `len(ds) > 0` 且 batch 可被 Trainer 消费。

## 5. 新增指标（BaseMetric）

### 改动

1. 在 `openLLV/evaluation/metrics.py` 增加类，或新建模块并在 `openLLV/evaluation/__init__.py` 导入。
2. 补 `test/eval/test_metrics.py` 风格用例，覆盖参考/无参考路径和方向。

### 最小实现

```python
import torch

from openLLV.evaluation import BaseMetric


class SharpnessMetric(BaseMetric):
    @property
    def requires_reference(self):
        return False

    @property
    def higher_is_better(self):
        return True

    def _compute_impl(self, en_image, ref_image=None):
        return float(torch.var(en_image))
```

### 契约

- 类名必须以 `Metric` 结尾。`create_metric` 会自动补后缀；评估结果里的显示名是去掉 `Metric` 的部分（`SharpnessMetric` → `Sharpness`）。
- `_compute_impl` 收到已搬到 metric device、尺寸已对齐（有参考时）的 tensor；返回可转 `float` 的值。
- 不实现别名：`BaseMetric` 的 `aliases` 属性当前不参与注册，只按类名查。
- 评估器把所有 metric kwargs 转发给**每个**指标构造器，签名里保留 `**kwargs` 更安全。
- 参考图缺失时，`requires_reference=True` 的指标记 `NaN`。

### 验证

```bash
python -c "import openLLV as llv; print('Sharpness' in llv.list_metrics())"
python -c "import openLLV as llv; print(llv.evaluate('results','results',metrics=['Sharpness'],save_path='results/eval.json'))"
python -m pytest -q test/eval
```

完成判据：`evaluate(metrics=[...])` 的 `statistics` 含该指标且 `total_count` 正确。

## 6. 收尾检查

- [ ] 新模块确实被导入链加载；`llv.list_*()` 能查到。
- [ ] 名称没有静默覆盖已有组件（尤其算法/损失/数据集/指标）。
- [ ] 设备只在 `Predictor` / `Trainer` 管理。
- [ ] 深度模型输出 `[0,1]`（或声明 `image_range`），传统算法保持 BGR 与值域。
- [ ] `python -m pytest -q test` 通过。
- [ ] 用户可见功能补了 `docs/` 与 `docs-zh-CN/`（遵循 `.agents/skills/openllv-doc-writer/SKILL.md`）。
- [ ] 没有把新实现说成官方复现；架构、权重、数值未对齐上游时明确说明。
