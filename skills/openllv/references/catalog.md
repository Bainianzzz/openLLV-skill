# openLLV 组件目录（0.1.0 源码快照）

查询键忽略大小写和首尾空白。新增组件前先在这里查重，再用命令刷新：

```bash
python -c "import openLLV as llv; import pprint; pprint.pp(llv.list_available())"
```

别名列写的是声明值；实际查询用小写化后的键（例如 `Kind++` 用 `kind++` 查询）。

## 深度模型（17）

| 注册名（类名） | 别名 | 任务 | 源码 | 参数文档 |
| --- | --- | --- | --- | --- |
| `LLNet` | — | llie | `openLLV/deepLearning/models/LLIE/LLNet.py` | `docs/models/llie/llnet.md` |
| `KinD` | — | llie | `openLLV/deepLearning/models/LLIE/KinD.py` | `docs/models/llie/kind.md` |
| `ZeroDCE` | — | llie | `openLLV/deepLearning/models/LLIE/ZeroDCE.py` | `docs/models/llie/zero-dce.md` |
| `ZeroDCEPlusPlus` | — | llie | `openLLV/deepLearning/models/LLIE/ZeroDCEPlusPlus.py` | `docs/models/llie/zero-dce++.md` |
| `RUAS` | — | llie | `openLLV/deepLearning/models/LLIE/RUAS.py` | `docs/models/llie/ruas.md` |
| `KinDPlusPlus` | `Kind++` | llie | `openLLV/deepLearning/models/LLIE/KinDPlusPlus.py` | `docs/models/llie/kind++.md` |
| `EnlightenGAN` | `EnlightenGAN` | llie | `openLLV/deepLearning/models/LLIE/EnlightenGAN.py` | `docs/models/llie/enlightengan.md` |
| `SCI` | — | llie | `openLLV/deepLearning/models/LLIE/SCI.py` | `docs/models/llie/sci.md` |
| `URetinexNet` | — | llie | `openLLV/deepLearning/models/LLIE/URetinex.py` | `docs/models/llie/uretinex-net.md` |
| `LEDNet` | — | llie | `openLLV/deepLearning/models/LLIE/LEDNet.py` | `docs/models/llie/lednet.md` |
| `LLFlow` | — | llie | `openLLV/deepLearning/models/LLIE/LLFlow.py` | `docs/models/llie/llflow.md` |
| `RetinexFormer` | — | llie | `openLLV/deepLearning/models/LLIE/RetinexFormer.py` | `docs/models/llie/retinexformer.md` |
| `PairLIE` | `Pair-LIE` | llie | `openLLV/deepLearning/models/LLIE/PairLIE.py` | `docs/models/llie/pairlie.md` |
| `LLFormer` | `LL-Former` | llie | `openLLV/deepLearning/models/LLIE/LLFormer.py` | `docs/models/llie/llformer.md` |
| `ZeroIG` | — | llie | `openLLV/deepLearning/models/LLIE/ZeroIG.py` | `docs/models/llie/zero-ig.md` |
| `DarkIR` | — | llie | `openLLV/deepLearning/models/LLIE/DarkIR.py` | `docs/models/llie/darkir.md` |
| `CIDNet` | `HVI-CIDNet` | llie | `openLLV/deepLearning/models/LLIE/CIDNet.py` | `docs/models/llie/cidnet.md` |

**模型名不是配置名**：`ZeroDCE++` 是配置名，不是模型查询名；模型要写 `ZeroDCEPlusPlus`（或直接用配置名调 `llv.train`）。

## 传统算法（13）

| 注册名 | 其他查询键 | 源码 | 参数文档 |
| --- | --- | --- | --- |
| `HE` | `he` | `openLLV/tradition/algorithms/BaseMethods/HE.py` | `docs/algorithms/base_methods/he.md` |
| `AHE` | `ahe` | `openLLV/tradition/algorithms/BaseMethods/AHE.py` | `docs/algorithms/base_methods/ahe.md` |
| `CLAHE` | `clahe` | `openLLV/tradition/algorithms/BaseMethods/CLAHE.py` | `docs/algorithms/base_methods/clahe.md` |
| `RCLAHE` | `rclahe` | `openLLV/tradition/algorithms/BaseMethods/RCLAHE.py` | `docs/algorithms/base_methods/rclahe.md` |
| `SSR` | `ssr` | `openLLV/tradition/algorithms/LLIE/Retinex.py` | `docs/algorithms/llie/retinex.md` |
| `MSR` | `msr` | 同上 | 同上 |
| `MSRCR` | `msrcr` | 同上 | 同上 |
| `NPE` | `npe` | `openLLV/tradition/algorithms/LLIE/NPE.py` | `docs/algorithms/llie/npe.md` |
| `BIMEF` | `bimef` | `openLLV/tradition/algorithms/LLIE/BIMEF.py` | `docs/algorithms/llie/bimef.md` |
| `LIME` | `lime` | `openLLV/tradition/algorithms/LLIE/LIME.py` | `docs/algorithms/llie/lime.md` |
| `GCP` | `gcp`、`gcp-ms` | `openLLV/tradition/algorithms/LLIE/GCP.py` | `docs/algorithms/llie/gcp.md` |
| `Gamma` | `gamma` | `openLLV/tradition/algorithms/LLIE/Gamma.py` | `docs/algorithms/llie/gamma.md` |
| `DarkChannel` | `darkchannel`、`dcp` | `openLLV/tradition/algorithms/Dehazing/DCP.py` | `docs/algorithms/dehazing/dcp.md` |

算法名来自类名、`name` 和 `aliases`，全部小写化。所有算法都继承基类参数：`output_type`、`keep_dtype`、`clip_output`、`value_range`。

## 评估指标（9）

| 显示名 | 参考图 | 方向 | 源码 |
| --- | --- | --- | --- |
| `PSNR` | 需要 | 越高越好 | `openLLV/evaluation/metrics.py` |
| `SSIM` | 需要 | 越高越好 | 同上 |
| `MSE` | 需要 | 越低越好 | 同上 |
| `MAE` | 需要 | 越低越好 | 同上 |
| `LPIPS` | 需要 | 越低越好 | 同上 |
| `LOE` | 需要（原始暗光图） | 越低越好 | 同上 |
| `NIQE` | 不需要 | 越低越好 | 同上 |
| `MUSIQ` | 不需要 | 越高越好 | 同上 |
| `PI` | 不需要 | 越低越好 | 同上 |

注册键是类名 `PSNRMetric` 等；`evaluate(metrics=...)` 接受显示名，大小写不敏感。`LPIPS`、`NIQE`、`MUSIQ`、`PI` 需要 `pyiqa` 和对应权重。

## 损失（按规范名查询）

类名、`name`、`aliases` 都可查询。常用规范名：

| 规范名 | 类 | 源码 |
| --- | --- | --- |
| `l1` | `L1Loss` | `openLLV/deepLearning/loss/LLIELoss/CommonLoss.py` |
| `mse` | `MSELoss` | 同上 |
| `smooth_l1` | `SmoothL1Loss` | 同上 |
| `charbonnier` | `CharbonnierLoss` | 同上 |
| `zerodce` | `ZeroDCE_Loss` | `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` |
| `zerodce_extension` | `ZeroDCE_extension_Loss` | 同上 |
| `sci` | `Sci_Loss` | `openLLV/deepLearning/loss/LLIELoss/Sci_Loss.py` |
| `llnet` | `LLNet_Loss` | `openLLV/deepLearning/loss/LLIELoss/LLNet_Loss.py` |
| `cidnet` | `CIDNet_Loss` | `openLLV/deepLearning/loss/LLIELoss/CIDNet_Loss.py` |
| `retinexformer` | `RetinexFormer_Loss` | `openLLV/deepLearning/loss/LLIELoss/RetinexFormer_Loss.py` |
| `kind` | `KinD_Loss` | `openLLV/deepLearning/loss/LLIELoss/KinD_Loss.py` |
| `kindplusplus` | `KinDPlusPlus_Loss` | `openLLV/deepLearning/loss/LLIELoss/KinDPlusPlus_Loss.py` |
| `llflow` | `LLFlow_Loss` | `openLLV/deepLearning/loss/LLIELoss/LLFlow_Loss.py` |
| `llformer` | `LLFormer_Loss` | `openLLV/deepLearning/loss/LLIELoss/LLFormer_Loss.py` |
| `uretinex` | `URetinex_Loss` | `openLLV/deepLearning/loss/LLIELoss/URetinex_Loss.py` |
| `ruas` | `RUAS_Loss` | `openLLV/deepLearning/loss/LLIELoss/RUAS_Loss.py` |
| `pairlie` | `PairLIE_Loss` | `openLLV/deepLearning/loss/LLIELoss/PairLIE_Loss.py` |
| `lednet` | `LEDNet_Loss` | `openLLV/deepLearning/loss/LLIELoss/LEDNet_Loss.py` |
| `darkir` | `DarkIR_Loss` | `openLLV/deepLearning/loss/LLIELoss/DarkIR_Loss.py` |
| `zeroig` | `ZeroIG_Loss` | `openLLV/deepLearning/loss/LLIELoss/ZeroIG_Loss.py` |
| `enlightengan` | `EnlightenGAN_Loss` | `openLLV/deepLearning/loss/LLIELoss/EnlightenGAN_Loss.py` |

`Trainer` 未指定 `loss` 时，先按模型类名匹配同名损失，匹配不到用 `charbonnier`。

## 数据集（2）

| 注册名 | 别名 | 源码 | 说明 |
| --- | --- | --- | --- |
| `CommonDataset` | `PairedDataset` | `openLLV/data/datasets/CommonDataset.py` | 通用配对数据集；支持 `root/split/input` 与 `root/split/target`，也支持根级 `input`/`target` |
| `LLFlowDataset` | — | `openLLV/data/datasets/LLFlowDataset.py` | LLFlow 官方 our485/eval15 布局；固定预处理 |

`list_datasets()` 还会出现 `basedataset`：`CommonDataset` 和 `LLFlowDataset` 都继承了 `BaseDataset.name = "BaseDataset"`，该键最终指向后注册的 `LLFlowDataset`。不要把它当成独立实现。

`Trainer` 的 `dataset` 也接受点号路径（`my_pkg.MyDataset`）、数据集类、数据集实例。

## 内置训练配置（17）

| 配置名 | 模型 |
| --- | --- |
| `LLNet` | `LLNet` |
| `KinD` | `KinD` |
| `KinD++` | `KinDPlusPlus` |
| `ZeroDCE` | `ZeroDCE` |
| `ZeroDCE++` | `ZeroDCEPlusPlus` |
| `RUAS` | `RUAS` |
| `EnlightenGAN` | `EnlightenGAN` |
| `SCI` | `SCI` |
| `URetinexNet` | `URetinexNet` |
| `LEDNet` | `LEDNet` |
| `LLFlow` | `LLFlow` |
| `RetinexFormer` | `RetinexFormer` |
| `PairLIE` | `PairLIE` |
| `LLFormer` | `LLFormer` |
| `ZeroIG` | `ZeroIG` |
| `DarkIR` | `DarkIR` |
| `CIDNet` | `CIDNet` |

配置文件在 `openLLV/deepLearning/config/*.yaml`。配置名匹配忽略大小写和标点，也匹配 YAML 内的 `model.name`；重名歧义会抛 `ValueError`。

## 预留任务目录

`openLLV/deepLearning/models/` 与 `openLLV/tradition/algorithms/` 下已有空包：`AiOIR`、`Deblurring`、`Dehazing`、`Denosing`（目录名拼写如此）、`Deraining`、`IC`、`IRR`、`ISR`、`JAR`、`SR`、`UIE`。新增对应任务实现时放进这些目录，并补 import。
