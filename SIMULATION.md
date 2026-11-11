# Mini Pi Plus：仿真导航＋基础 locomotion

> 2026-11-11：当前仿真与机载入口已共用 `navigation/`，AMP 固定为真机 Pro 版本。
> 当前代码、测试结果和未通过项以 [共享导航状态](SHARED_NAVIGATION_STATUS.md) 为准；下文旧策略成绩保留作历史记录。


本文记录原有真值导航基线。新增的相机驱动版本使用独立入口 `scripts/run_visual.py`，
传感器边界、运行方式与验证记录见 [VISUAL_NAVIGATION.md](VISUAL_NAVIGATION.md)。

## 2026-11-10：真机基础运控同步

原仿真与真机不一致，现已将 `sim.robot.Robot` 的默认策略切换到真机当前版本。
`scripts/run_sim.py`、`scripts/run_visual.py` 共用此入口。同步源为真机
`/home/nvidia/action_library/policy/amp/`，原始文件、导出的 CPU 权重及哈希保存在
`vendor/pi_plus_pro/`；审计证据在 `output/deployment/loco_sync_20261110/`。

| 项目 | 原仿真 | 当前默认仿真 / 真机配置 |
|---|---|---|
| 权重 | 上游 `model_40000.pt` | 当前 `amp.trt` 中的同一组 FP32 权重，策略标识 `pi_plus_amp_1110_policy_1.rknn` |
| 观测 / 历史 | 69 × 5，观测限幅 ±100 | 69 × 5，观测限幅 ±18，旧帧到新帧，初始历史为零 |
| 动作 | 限幅 ±100，缩放 0.25 | 限幅 ±10，缩放 0.25，各关节额外缩放为 1 |
| 速度入口 | 直接送入策略 | Soccer / normal，限幅后 0.3 低通，再执行死区 |
| 死区 | 无控制器死区 | 平移 0.08 m/s、转向 0.17 rad/s，死区处理结果参与下一帧滤波 |
| PD / 物理频率 | 500 Hz | 1000 Hz；策略频率仍为 50 Hz |
| PD 增益 | 上游较小阻尼 | 使用实机 AMP 的 `kp`、`kd`，按关节名映射 |

原厂运行的是 Jetson TensorRT 引擎，配置中的 `.rknn` 文件名不是实际推理后端。
设备上没有对应原始 ONNX/PT，故从已校验 SHA-256 的 TensorRT 文件提取原有 FP32 张量，
使用相同的 `345→512→256→128→20` 网络与 ELU 激活在 CPU 推理，没有训练或拟合新策略。
`scripts/export_device_amp.py` 只接受这一个已审计引擎的哈希，拒绝对其他版本套用字节布局。

验证结果：129 组独立 Jetson 推理输入的最大绝对误差为 `1.3352e-5`；另从正在运行的
原厂控制器只读订阅连续 50 帧调试数据，按序号核对历史堆叠、上一帧动作、20 关节目标映射，
本地策略输出最大绝对误差为 `2.3842e-7`。本次没有向真机发送速度、电机或模式切换指令。
站立、0.4 m/s 直行及双向带前进转弯各进行了 2 秒稳定 + 5 秒指令仿真，均未摔倒。
详细结果在 `output/sim/device_amp_sync_20261110/motion_checks.json`。

**旧导航参数需要重新标定。** 原默认 0.22–0.25 m/s 从零开始时，经 0.3 滤波得到
0.066–0.075 m/s，被 0.08 m/s 死区置零，并且不会逐帧积累。因此旧的默认导航目前不能起步；
原 0.12 m/s 侧移下限及较小转向指令也不能继续沿用。新增 `policy_cmd_vx/vy/wz` 遥测列记录
实际送入网络的速度，`summary.json` / 视觉 `run_manifest.json` 记录策略、配置哈希与频率。
完成基础运控同步不代表当前策略已重新通过整圈导航。

```sh
# 当前真机运控的短时真值导航检查；到达时限会报告 duration_limit
.venv/bin/python scripts/run_sim.py --speed .4 --duration 10 --output output/sim/pro_smoke
# 明确指定旧权重，复查历史基线（不是当前真机版本）
.venv/bin/python scripts/run_sim.py --policy vendor/mini_pi_plus_amp/Exported_policy/model_40000.pt
.venv/bin/python -m unittest scripts.test_device_amp scripts.test_sim scripts.test_visual scripts.test_camera_profile
```

同步范围是 AMP 身体运控，使用原厂默认 Soccer / normal 速度模式。手柄模式按钮、
真机启动状态机、看门狗、头部控制不属于此仿真适配器。旧 MJCF 与真机的关节零点通过
相对默认姿态的关节位移转换；没有把真机肩部编码器偏置直接填进旧模型的关节角。
动力学、力矩限制、碰撞几何、相机外参仍保留原仿真假设，不能据此声称整机物理一致。

## 同步前的历史完成范围

已在现有田径场地上运行真实 MuJoCo 关节动力学：导航输出身体坐标系 `vx, vy, wz`，
高擎 Mini Pi Plus AMP 输出 20 关节目标，PD 控制驱动电机。
三个起点均完成逆时针一圈、绕过三个箱体，并在终点后继续仿真 3 秒检查减速停车。
没有设置基座轨迹、外力托举、修改机器人质量、缩小碰撞模型或取消障碍碰撞。

**这是仿真真值导航基线。** 规划读取完整赛道和障碍坐标，控制读取 MuJoCo 基座位姿；
相机已安装并可渲染 RGB／深度，但图像尚未用于导航。当前没有 MPPI、SLAM、ROS2 或视觉终点识别。
此阶段的作用是验证现有 loco 能否执行赛道要求，并给后续视觉算法提供可对照的基线。

**原视频在最后一个箱体旁存在严重的步态采样混叠。** 01 起点仿真 93–95 s，
同一只脚的触地周期约 0.31 s，而原视频以 3 倍速、10 fps 导出，每帧跨过 0.30 s，
几乎只展示相同的步态相位。两脚最低点的最大离地高度分别仅约 1.23、1.31 cm，
进一步强化了滑行观感。原速 50 fps 回放可见交替迈步。

同时存在短暂落脚滑移：每步接触点切向速度积分约 5 mm，大部分发生在触地后 50 ms 内；
稳定支撑段（同一足连续接触超过 50 ms）的按法向力加权滑移速度约 0.0022 m/s。
含触地瞬间的全接触均值约 0.057 m/s，不能将其解释成整个支撑期持续滑行。
只改变 NoSlip 迭代数 0→3 的短对照没有降低全接触均值，不能据此归因于软接触参数。
以下完赛结果仅表示通过当前路径判罚；落脚质量、扰动鲁棒性尚未完成验收。

## 运行

本机环境已经安装好，可直接运行：

```sh
# 默认：01 起点，包含三个障碍，完整身体包络规划，无图形窗口
.venv/bin/python scripts/run_sim.py

# macOS 实时观察；关闭窗口结束。Linux 也可用 python 加 --viewer
.venv/bin/mjpython scripts/run_sim.py --viewer --start 01

# 换起点、指定结果目录
.venv/bin/python scripts/run_sim.py --start 02 --output output/sim/start_02

# 从已经记录的物理状态生成原速 50 fps 视频，右下角显示机载相机
.venv/bin/python scripts/render_replay.py output/sim/start_01

# 重放相同指令检查实际接触滑移，并以原记录频率导出最后一个箱子片段
.venv/bin/python scripts/diagnose_slip.py
.venv/bin/python scripts/render_slip_audit.py

# 去掉障碍的巡线对照，以及错误的小包络对照
.venv/bin/python scripts/run_sim.py --no-obstacles --output output/sim/clear_track
.venv/bin/python scripts/run_sim.py --planner feet-only --output output/sim/feet_only

# 功能检查与三个官方策略的速度响应测量
.venv/bin/python scripts/test_sim.py
.venv/bin/python scripts/benchmark_loco.py
```

完赛退出码为 `0`；碰撞、摔倒、出界、时限结束或手动关窗返回 `2`。
默认时限为 240 秒，`--duration` 最大 300 秒。`--speed` 仅控制普通路段，障碍区由控制器降低速度。
交互窗口尚未进行人工操作验收；已经实际验证的是无窗口物理仿真和离屏渲染。

新环境：

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-sim.txt
.venv/bin/python scripts/fetch_pi_plus.py
.venv/bin/python scripts/generate_field.py
.venv/bin/python scripts/test_sim.py
.venv/bin/python scripts/run_sim.py
```

`requirements-sim-tested.txt` 记录本次测试的完整依赖版本。
`upstream.lock.json` 固定官方提交和每个文件的 Git blob SHA；下载器逐文件校验。
下载内容位于忽略目录 `vendor/mini_pi_plus_amp/`，包括模型、三个权重、原部署脚本、训练配置及许可证。
不需要安装 Isaac Lab，不在本机训练策略。

## 已验证结果

在固定地图、无额外扰动、无感知误差条件下，三个起点各测试一轮。
以下结果来自最终的连续线段碰撞检查版本，不是早期调参记录。

| 起点 | 一圈用时 | 过线后检查 | 停车期间位移 | 判罚 |
|---|---:|---:|---:|---|
| 01 | 120.62 s | 3 s | 0.111 m | 无 |
| 02 | 120.04 s | 3 s | 0.111 m | 无 |
| 03 | 120.66 s | 3 s | 0.109 m | 无 |

模型车道中心长度约 21.5635 m；其内圈与规则文字 19 m 的差异仍保留，见原 README。
三个起点的详细结果分别位于 `output/sim/start_01/`、`start_02/`、`start_03/`。
5 项回归检查覆盖真实行走、关节映射、单脚／双脚越界区别、障碍碰撞、摔倒及不可容纳包络。

**允许部分脚掌越线是这版成功结果的重要条件。** 判罚使用两只脚的实际碰撞网格投影：
两只脚均与跑道无交集才终止。足部投影不完全在线内的累计时长分别为 17.44、17.58、18.34 秒。
这不是“双脚始终完全在跑道内”的验证结果；若现场要求更严格，需要改变步态／手臂动作或路径约束。

基于本地场地和原始权重的失败对照也保留：小包络路径会造成手腕碰箱；
只把前进指令降低到 0.04–0.10 m/s 会出现明显死区。`model_40000.pt` 的 5 秒直行试验中，
0.08 m/s 指令只产生约 0.00063 m/s 的平均前向位移；0.12 m/s 指令约为 0.094 m/s。
0.12 m/s 纯侧移试验还带有约 0.019 m/s 的前向漂移。
这些是包含启动过程的平均值，不能视为实机稳态模型。完整测量在 `output/sim/loco_benchmark.json`。

## 实现和文件

| 文件 | 职责 |
|---|---|
| `sim/robot.py` | 合并机器人／场地，加载 AMP，构造观测、映射关节、执行 PD |
| `sim/navigation.py` | 已知地图路径格点搜索、身体包络检查、位置反馈和速度平滑 |
| `sim/scoring.py` | 接触、姿态、脚掌投影和越界检查 |
| `scripts/run_sim.py` | 时序调度、起点、终点停车、日志与完整状态记录 |
| `scripts/render_replay.py` | 已记录物理状态回放、机载画面、深度样本、轨迹图 |
| `scripts/benchmark_loco.py` | 对三个官方权重进行速度响应测量 |
| `scripts/test_sim.py` | 有物理执行的功能回归 |

底层使用 2 ms 物理步长、10 倍降采样，即 50 Hz 策略更新；每帧观测 69 维，5 帧历史，
20 维动作，动作缩放 0.25，观测顺序、关节顺序、默认姿态和 PD 增益与上游部署契约对齐。
策略用 CPU 推理。训练 checkpoint 通过 `weights_only=True` 加载；训练用 Normalizer 对象
以不执行代码的空状态类接收，actor 不使用其状态。未导入上游游戏手柄、键盘监听或 Isaac 环境。

上层当前也以 50 Hz 更新。普通路段跟踪中心线；障碍区保持身体朝向跑道切向，
用侧移绕行。规划采用 26 × 42 cm 的近似矩形身体包络，纵向余量 5 mm、横向余量 15 mm，
基座允许偏离中心线最多 23 cm。连续线段／扩张矩形的相交检查保证该近似路径不切过箱体；
实际是否碰撞仍以机器人原始三维碰撞网格为准。此矩形规划只支持当前水平直道上的轴对齐箱体，
改变该场景假设会明确报错。它不是适用于任意地形的通用全身规划器。

障碍区低速反馈加入了约 0.12 m/s 的平移指令下限来跨过当前策略的死区，仍保留加速度限制。
这个参数依赖当前权重；替换策略后应重跑速度测量和整圈测试。

每次运行输出：

- `summary.json`：成败、用时、原因、足部越线时长、姿态指标、规划假设。
- `telemetry.csv`：时间、真实位姿、速度指令、传感器速度、整圈进度。
- `states.npz`：每个策略周期的完整 `qpos` 与时间，用于物理结果回放。
- `planned_path.csv`：规划路径；`scene.xml`：包含机器人和场地的实际模型。
- 渲染后另有 `replay.mp4`、`trajectory.png`、起点／障碍相机图、米制深度 `.npy`。

生成的运行 `scene.xml` 使用本机资产绝对路径；跨机器迁移时应在新位置重新运行生成。
视频通过播放已记录的动力学状态生成，视频上的相机画面不参与控制。
`output/sim/slip_audit/` 保存滑移审计和 1 倍速 50 fps 局部回放；
审计重放与原记录的最大 qpos 误差为 0。接触力与速度在复制的状态上用 `mj_forward`
同步计算，不改变原仿真；统计只包含法向力大于 1 N 的足地接触，滑移阈值不是官方规则。

## 判罚和后续边界

机器人／箱体实际接触和非足部接地以 500 Hz 检查；脚掌凸包的地面投影以 50 Hz 检查。
摔倒还包含基座高度低于 20 cm 或倾角超过 60° 的工程阈值。
跑道区域取两条白线中心之间，终点取相对抽签起点前进一圈的中心线里程；
这些实现细节不是官方裁判算法。碰撞、摔倒、双脚出界任一发生会立即记录失败。

相机假设安装在基座坐标 `(0.09, 0, 0.22) m`，向下俯视 30°、垂直视场 70°。
当前 macOS 离屏深度渲染提示缺少 `ARB_clip_control`，深度精度受限；尚未据此验证深度测量误差。

接下来接入 RGB 白线／终点、深度箱体感知，用局部观测替换已知地图，并注入姿态误差、延迟、
摩擦和障碍位置变化。全视觉闭环、随机扰动成功率、实机部署和正式规则资格均未验证。

## 来源

官方仓库：[HighTorque Mini-Pi-Plus AMP](https://github.com/HighTorque-Robotics/Mini-Pi-Plus_AMP)。
机器人资产固定提交 `6a71f08177e0488122264250dbd1bfd4af70097f`，历史基线权重为 `Exported_policy/model_40000.pt`。
历史部署参数依据该版本 `legged_lab/scripts/sim2sim_pi_plus.py`；当前运控来源见本文顶部。
完整上游版权声明和 BSD-3-Clause
许可证保留在下载目录。模型、权重、媒体等第三方资产的再分发条款以其上游说明为准。

本次环境为 macOS、Python 3.13.5、MuJoCo 3.17.0、PyTorch 2.15.0、NumPy 2.5.4。
