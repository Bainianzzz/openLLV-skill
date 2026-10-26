# openLLV skill

[openLLV](https://github.com/glory-wan/openLLV) 是一个面向低层视觉（low-level vision）的 Python 工具包，统一了传统算法、深度学习模型、训练、推理、图像 I/O 与图像质量评估。
本 Skill 只覆盖 Python SDK：`predict` / `train` / `evaluate` / `imread` / `imwrite` 的正确用法，以及常见参数、输出路径与值域问题。内容全部对照 openLLV 0.1.0 源码核实，Agent 会先按任务路由读取对应说明，再生成代码，避免误用接口与参数。

## 安装

如果你正在使用 Claude Code、Codex 等 coding agent，直接将下面这段话发给它即可自动完成安装：

```text
Fetch the installation guide and follow it: https://raw.githubusercontent.com/Bainianzzz/openLLV-skill/main/README.md
```

手动安装推荐使用全局安装方式：

```bash
npx skills add Bainianzzz/openLLV-skill -g -y
```

然后准备 openLLV 运行环境（Skill 只有在目标仓库里才能执行）：

```bash
git clone https://github.com/glory-wan/openLLV.git
cd openLLV
python -m pip install -e .          # 需要 Python 3.8+ 与 PyTorch 2.0+
python -c "import openLLV as llv; print(len(llv.list_models()))"   # 自检
```

后续如果本 Skill 有更新，执行：

```bash
npx skills update openllv -g -y
```

## 内容

```text
skills/
└── openllv/
    ├── SKILL.md
    └── references/
        ├── api-contracts.md
        ├── catalog.md
        └── gotchas.md
```

| 文件                          | 作用                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| `SKILL.md`                    | 轻量入口：导入自检、任务路由，以及预测 / 训练 / 评估 / 图像 I/O 四段带完成判据的流程  |
| `references/api-contracts.md` | 公开签名、`predict` 的 kwargs 路由、返回契约、`Trainer` 平铺键、评估结果结构          |
| `references/catalog.md`       | 已注册的 17 个模型、13 个传统算法、9 个指标、21 个损失、2 个数据集、17 个内置训练配置 |
| `references/gotchas.md`       | 23 条已核实的边界行为、默认副作用与参数命名陷阱                                       |

`SKILL.md` 只保留每次都会用到的流程；细节按需从 `references/` 读取，不需要一次全加载。

## 能力

| 能力                                                  | 方式                                    | 入口                          |
| ----------------------------------------------------- | --------------------------------------- | ----------------------------- |
| 单图或目录做增强 / 复原（传统算法与深度模型）         | `llv.predict` / `llv.enhance`           | `SKILL.md` §2                 |
| 用训练产出的 checkpoint 推理                          | `llv.predict(".../last.pt", ...)`       | `references/api-contracts.md` |
| 训练模型（内置配置、YAML、字典、平铺参数）            | `llv.train` / `Trainer`                 | `SKILL.md` §3                 |
| 评估输出目录（全参考与无参考指标）                    | `llv.evaluate`                          | `SKILL.md` §4                 |
| 读取、保存、转换图像格式                              | `llv.imread` / `llv.imwrite`            | `SKILL.md` §5                 |
| 查询可用的模型、算法、指标、损失、数据集与别名        | `llv.list_available()` / `llv.list_*()` | `references/catalog.md`       |
| 排查参数、输出路径、值域、checkpoint 问题             | —                                       | `references/gotchas.md`       |

## 许可证

[MIT License](LICENSE).

Copyright (c) 2026 glory wan
