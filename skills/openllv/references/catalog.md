# openLLV 可用组件速查（0.1.0）

这里是 `llv.predict`、`llv.train`、`llv.evaluate` 能接受的名称与别名，以及每个组件的构造参数。

查询忽略大小写和首尾空白；用命令刷新完整列表：

```bash
python -c "import openLLV as llv; import pprint; pprint.pp(llv.list_available())"
```

表中别名按声明原样列出，查询时统一忽略大小写（例如 `Kind++` 用 `kind++`）。

## 深度模型（17）

深度模型的构造参数即 `config` 里的键：`llv.predict` 可把 `config={...}` 传，也可平铺传；`llv.train` 用内置配置的默认值，`model_params={...}` 覆盖。`input_channels=3` 和 `mode` 由框架管理，不用手填。

| 注册名（类名） | 别名 | 任务 | 构造参数（默认） |
| --- | --- | --- | --- |
| `LLNet` | — | llie | `patch_size=17`、`patch_stride=3`、`hidden_dims=[2000,1600,1200]`、`activation=sigmoid`、`output_activation=sigmoid` |
| `KinD` | — | llie | `decomposition_channels=64`、`decomposition_layers=5`、`restoration_channels=32`、`adjustment_channels=32`、`adjustment_layers=3`、`illumination_ratio=5.0` |
| `ZeroDCE` | — | llie | `number_f=32` |
| `ZeroDCEPlusPlus` | — | llie | `number_f=32`、`scale_factor=12`、`initialize_weights=false` |
| `RUAS` | — | llie | `iem_nums=3`、`nrm_nums=3`、`enhance_channel=3`、`denoise_channel=6`、`pretrained_denoise_path=None` |
| `KinDPlusPlus` | `Kind++` | llie | `decomposition_channels=32`、`restoration_channels=32`、`adjustment_channels=32`、`illumination_ratio=5.0` |
| `EnlightenGAN` | — | llie | `generator_channels=32`、`discriminator_channels=32`、`discriminator_layers=3`、`use_attention=true`、`local_patch_ratio=0.5` |
| `SCI` | — | llie | `stage=3`、`enhance_layers=1`、`enhance_channels=3`、`calibrate_layers=3`、`calibrate_channels=16` |
| `URetinexNet` | — | llie | `unfolding_rounds=3`、`gamma=0.01`、`lambda=0.01`、`gamma_offset=0.01`、`lambda_offset=0.01`、`use_concat_l=true`、`adjustment_ratio=5.0`、`adaptive_ratio=false` |
| `LEDNet` | — | llie | `channels=[32,64,128,128]`、`connection=false`、`use_side_loss=false`、`kernel_size=5`、`curve_n=3`、`ppm_bins=(1,2,3,6)`、`image_range=zero_one` |
| `LLFlow` | — | llie | `nf=64`、`nb=24`、`K=12`、`L=3`、`train_gt_ratio=0.2`、`quant=32`、`inference_padding=reflect16` |
| `RetinexFormer` | — | llie | `output_channels=3`、`feature_channels=32`、`stage=1`、`levels=2`、`num_blocks=[1,1,1]`、`clamp_output=true` |
| `PairLIE` | `Pair-LIE` | llie | `feature_channels=64`、`enhancement_gamma=0.2`、`clamp_output=true` |
| `LLFormer` | `LL-Former` | llie | `output_channels=3`、`dim=16`、`num_blocks=[2,4,8,16]`、`num_refinement_blocks=2`、`heads=[1,2,4,8]`、`ffn_expansion_factor=2.66`、`bias=false`、`layernorm_type=WithBias`、`attention=true`、`skip=false`、`pad_input=true`、`clamp_output=true`、`tile_size=None`、`tile_overlap=0` |
| `ZeroIG` | — | llie | `enhance_layers=3`、`enhance_channels=64`、`denoise1_channels=48`、`denoise2_channels=48`、`pretrained_weights=None` |
| `DarkIR` | — | llie | `width=32`、`middle_blk_num_enc=2`、`middle_blk_num_dec=2`、`enc_blk_nums=[1,2,3]`、`dec_blk_nums=[3,1,1]`、`dilations=[1,4,9]`、`extra_depth_wise=true`、`side_loss=false` |
| `CIDNet` | `HVI-CIDNet` | llie | `channels=[36,36,72,144]`、`heads=[1,2,4,8]`、`norm=false`、`density_k=0.2`、`input_gamma=1.0`、`saturation_scale=1.0`、`intensity_scale=1.0`、`clamp_output=true` |

**模型名不是配置名**：`llv.predict` 只认模型注册名，配置名只用于 `llv.train`。典型误用见 `gotchas.md` #8。

## 传统算法（13）

所有算法都继承基类构造参数：`output_type="numpy"`、`keep_dtype=true`、`clip_output=true`、`value_range="auto"`。算法参数在 `llv.predict` 里直接平铺传。

| 注册名 | 其他查询键 | 构造参数（默认） |
| --- | --- | --- |
| `HE` | `he` | `color_space=rgb` |
| `AHE` | `ahe` | `color_space=yuv`、`tile_grid_size=(8,8)` |
| `CLAHE` | `clahe` | `color_space=yuv`、`clip_limit=2.0`、`tile_grid_size=(8,8)` |
| `RCLAHE` | `rclahe` | `color_space=yuv`、`clip_limit=2.0`、`tile_grid_size=(8,8)`、`iterations=3` |
| `SSR` | `ssr` | `sigma=80.0`、`low_clip=1.0`、`high_clip=99.0`、`eps=1e-6` |
| `MSR` | `msr` | `scales=(15.0,80.0,250.0)`、`low_clip=1.0`、`high_clip=99.0`、`eps=1e-6` |
| `MSRCR` | `msrcr` | `scales=(15.0,80.0,250.0)`、`alpha=125.0`、`beta=46.0`、`gain=1.0`、`offset=0.0`、`low_clip=1.0`、`high_clip=99.0`、`eps=1e-6` |
| `NPE` | `npe` | `sigma=15.0`、`illumination_floor=0.05`、`enhancement_strength=4.0`、`naturalness=0.35`、`detail_weight=1.0` |
| `BIMEF` | `bimef` | `exposure_ratio=None`、`target_mean=0.55`、`max_ratio=5.0`、`well_exposed_sigma=0.2`、`contrast_weight=1.0`、`saturation_weight=1.0`、`well_exposed_weight=1.0` |
| `LIME` | `lime` | `gamma=0.8`、`guided_radius=15`、`guided_eps=0.001`、`illumination_floor=0.05`、`exposure=1.0` |
| `GCP` | `gcp`、`gcp-ms` | `gamma_max=6.0`、`erosion_window=15`、`atmospheric_bins=200`、`atmospheric_percentile=0.99`、`t_min=0.1`、`blur_ksize=7`、`high_percentile=99.5`、`low_percentile=0.5`、`eps=1e-6` |
| `Gamma` | `gamma` | `gamma=0.6` |
| `DarkChannel` | `darkchannel`、`dcp` | `size=15`、`omega=0.95`、`t_min=0.1`、`guided_radius=60`、`guided_eps=0.0001` |

算法支持类名和别名查询，大小写不敏感。

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

配置名匹配忽略大小写和标点（`ZeroDCEPlusPlus` 命中 `ZeroDCE++` 配置），也匹配配置里的 `model.name`；重名歧义会抛 `ValueError`。
