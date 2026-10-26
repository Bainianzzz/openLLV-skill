# openLLV 可用组件速查（0.1.0）

这里是 `llv.predict`、`llv.train`、`llv.evaluate` 能接受的名称与别名。

查询忽略大小写和首尾空白；用命令刷新完整列表：

```bash
python -c "import openLLV as llv; import pprint; pprint.pp(llv.list_available())"
```

表中别名按声明原样列出，查询时统一忽略大小写（例如 `Kind++` 用 `kind++`）。

## 深度模型（17）

| 注册名（类名） | 别名 | 任务 | 参数文档 |
| --- | --- | --- | --- |
| `LLNet` | — | llie | `docs/models/llie/llnet.md` |
| `KinD` | — | llie | `docs/models/llie/kind.md` |
| `ZeroDCE` | — | llie | `docs/models/llie/zero-dce.md` |
| `ZeroDCEPlusPlus` | — | llie | `docs/models/llie/zero-dce++.md` |
| `RUAS` | — | llie | `docs/models/llie/ruas.md` |
| `KinDPlusPlus` | `Kind++` | llie | `docs/models/llie/kind++.md` |
| `EnlightenGAN` | — | llie | `docs/models/llie/enlightengan.md` |
| `SCI` | — | llie | `docs/models/llie/sci.md` |
| `URetinexNet` | — | llie | `docs/models/llie/uretinex-net.md` |
| `LEDNet` | — | llie | `docs/models/llie/lednet.md` |
| `LLFlow` | — | llie | `docs/models/llie/llflow.md` |
| `RetinexFormer` | — | llie | `docs/models/llie/retinexformer.md` |
| `PairLIE` | `Pair-LIE` | llie | `docs/models/llie/pairlie.md` |
| `LLFormer` | `LL-Former` | llie | `docs/models/llie/llformer.md` |
| `ZeroIG` | — | llie | `docs/models/llie/zero-ig.md` |
| `DarkIR` | — | llie | `docs/models/llie/darkir.md` |
| `CIDNet` | `HVI-CIDNet` | llie | `docs/models/llie/cidnet.md` |

**模型名不是配置名**：`ZeroDCE++` 是配置名，不是模型查询名；模型要写 `ZeroDCEPlusPlus`（或直接用配置名调 `llv.train`）。

## 传统算法（13）

| 注册名 | 其他查询键 | 参数文档 |
| --- | --- | --- |
| `HE` | `he` | `docs/algorithms/base_methods/he.md` |
| `AHE` | `ahe` | `docs/algorithms/base_methods/ahe.md` |
| `CLAHE` | `clahe` | `docs/algorithms/base_methods/clahe.md` |
| `RCLAHE` | `rclahe` | `docs/algorithms/base_methods/rclahe.md` |
| `SSR` | `ssr` | `docs/algorithms/llie/retinex.md` |
| `MSR` | `msr` | 同上 |
| `MSRCR` | `msrcr` | 同上 |
| `NPE` | `npe` | `docs/algorithms/llie/npe.md` |
| `BIMEF` | `bimef` | `docs/algorithms/llie/bimef.md` |
| `LIME` | `lime` | `docs/algorithms/llie/lime.md` |
| `GCP` | `gcp`、`gcp-ms` | `docs/algorithms/llie/gcp.md` |
| `Gamma` | `gamma` | `docs/algorithms/llie/gamma.md` |
| `DarkChannel` | `darkchannel`、`dcp` | `docs/algorithms/dehazing/dcp.md` |

算法支持类名和别名查询，大小写不敏感。所有算法都继承基类参数：`output_type`、`keep_dtype`、`clip_output`、`value_range`。

## 评估指标（9）

| 显示名 | 参考图 | 方向 |
| --- | --- | --- |
| `PSNR` | 需要 | 越高越好 |
| `SSIM` | 需要 | 越高越好 |
| `MSE` | 需要 | 越低越好 |
| `MAE` | 需要 | 越低越好 |
| `LPIPS` | 需要 | 越低越好 |
| `LOE` | 需要（原始暗光图） | 越低越好 |
| `NIQE` | 不需要 | 越低越好 |
| `MUSIQ` | 不需要 | 越高越好 |
| `PI` | 不需要 | 越低越好 |

`evaluate(metrics=...)` 接受显示名，大小写不敏感。`LPIPS`、`NIQE`、`MUSIQ`、`PI` 需要 `pyiqa` 和对应权重。

## 损失（按规范名查询）

规范名和类名都可查询，大小写不敏感。常用规范名：

| 规范名 | 类 |
| --- | --- |
| `l1` | `L1Loss` |
| `mse` | `MSELoss` |
| `smooth_l1` | `SmoothL1Loss` |
| `charbonnier` | `CharbonnierLoss` |
| `zerodce` | `ZeroDCE_Loss` |
| `zerodce_extension` | `ZeroDCE_extension_Loss` |
| `sci` | `Sci_Loss` |
| `llnet` | `LLNet_Loss` |
| `cidnet` | `CIDNet_Loss` |
| `retinexformer` | `RetinexFormer_Loss` |
| `kind` | `KinD_Loss` |
| `kindplusplus` | `KinDPlusPlus_Loss` |
| `llflow` | `LLFlow_Loss` |
| `llformer` | `LLFormer_Loss` |
| `uretinex` | `URetinex_Loss` |
| `ruas` | `RUAS_Loss` |
| `pairlie` | `PairLIE_Loss` |
| `lednet` | `LEDNet_Loss` |
| `darkir` | `DarkIR_Loss` |
| `zeroig` | `ZeroIG_Loss` |
| `enlightengan` | `EnlightenGAN_Loss` |

`Trainer` 未指定 `loss` 时，先按模型类名匹配同名损失，匹配不到用 `charbonnier`。

## 数据集（2）

| 注册名 | 别名 | 说明 |
| --- | --- | --- |
| `CommonDataset` | `PairedDataset` | 通用配对数据集；支持 `root/split/input` 与 `root/split/target`，也支持根级 `input`/`target` |
| `LLFlowDataset` | — | LLFlow 官方 our485/eval15 布局；固定预处理 |

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

配置名匹配忽略大小写和标点，也匹配 YAML 内的 `model.name`；重名歧义会抛 `ValueError`。
