# Pi Plus Pro 当前实机 AMP 快照

2026-11-10 从用户真机只读取得，源目录 `/home/nvidia/action_library/policy/amp/`。
配置策略标识为 `walk/pi_plus_amp_1110_policy_1.rknn`，Jetson 实际运行 `amp.trt`。
原文件与导出文件的 SHA-256 见 `SOURCE.json`，仿真加载时检查这些哈希。

- `amp.yaml`、`amp.trt`、`param.yaml`：原厂配置及引擎原样保存。
- `pd.yaml`：身体/头部名称、硬件方向及零点参考；仿真采用 AMP 配置中的身体 PD 增益。
- `actor.npz`：从该引擎提取的四层 FP32 权重，供本机 CPU/PyTorch 使用。
- `engine_reference.npz`、`engine_probe.json`：独立 Jetson TensorRT 推理的 129 组输入输出及引擎元信息。
- `runtime_reference.npz`：正在运行的控制器连续 50 帧调试数据，按消息序号同步，包含观测、历史输入、动作及 PD 顺序的目标位移。
- `actor.verification.json`：导出时的数值验证结果。

重新导出运行 `.venv/bin/python scripts/export_device_amp.py`，其张量布局仅对固定引擎哈希有效。
第三层 GEMM 在引擎内按转置布局存储；导出恢复原矩阵，而非估计权重。
这是用户设备的本地仿真副本，不赋予原厂资产新的再分发许可。

控制器二进制审计、只读推理脚本和逐帧对照保存在
`output/deployment/loco_sync_20261110/`；运行与验证边界见根目录 `SIMULATION.md`。
