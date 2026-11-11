# 共享导航版本：部署与验收

本版本将 RGB-D 感知、视觉里程计、局部绕障规划、路径跟踪和 AMP 速度适配放到同一个 `navigation/` 包。
仿真入口 `scripts/run_visual.py`、离线回放 `scripts/replay_navigation.py` 和机载入口
`deploy/shared_navigation.py` 使用同一个 `NavigationRuntime`。`sim/visual_navigation.py` 只保留兼容导出。
旧 `line_follow_core.py` 与 `--calibrated-turns` 保留用于历史诊断，不是共享导航的执行路径。

## 当前版本的边界

- 冻结真机 `amp.yaml`、`amp.trt`、PD 参数和主控制器二进制的 SHA-256；启动前只读比对。
- 导航不推理另一套 AMP，也不覆盖原厂程序。仿真使用该 TensorRT 引擎提取的同一组 FP32 权重，已有数值对照。
- 视觉输入是 RGB、米制深度、相机内外参和 AHRS 姿态。无世界位置、已知跑道地图、箱体真值或仿真接触信息输入。
- 规划器是局部横向网格上的动态规划；跟踪器根据前方路径点、路径方向计算 `vx, vy, wz`，不是 MPPI。
- 地面由重力约束下的深度平面估计，不使用绿色像素定高度。箱体按相对地面高度、表面和尺寸识别，
  红、黄、绿分别作为同类障碍；已观测箱体的方向与遮挡记忆用于避免局部碎片误增尺寸。
- 短时定位优先配准白线/箱面；地面纹理以米制鸟瞰图作相关匹配，并检查深度、纹理可观测性和匹配一致性。
  箱体之间的窄通道采用观测跑道方向下的位置闭环，先减速再侧移；实际位移不足时允许有限次站立恢复。
- `AMPCommandAdapter` 在导航侧补偿原厂 0.3 滤波和死区，限制原始请求在 `±0.4, ±0.4, ±1.0`。
  小于策略实际起步能力的前进目标会升到 0.20 m/s；这是导航目标适配，不能把 `--speed` 理解为硬件实际速度保证。
- 原厂滤波后的速度量是策略输入，不是机身真实速度。视觉里程计才提供当前实现的平移估计。
- 视觉跟踪失败：保留最后可信参考帧和障碍记忆，输出零；三帧可信观测后恢复。
  不用速度积分或仿真专用的脚底里程计冒充实际位置。无箱体记忆、无终点目标时，停车后可重建局部参考段；重建次数单独记录，累计路程不是精确全局定位。
- 机器人动力学仍是上游 Mini Pi Plus；Pro 惯量/碰撞/摩擦、相机外参、传感器时延未完成实机标定。
- 包内 `pro_config.json` 是明确标记的仿真假设。它只允许旁路，不允许执行运动。

导航依赖也已固定：NumPy 1.24.4、OpenCV 4.10.0（wheel 4.10.0.84）。
旧本机的 OpenCV 5 与目标环境会产生不同指令，不能混用；入口会拒绝版本不符的环境。

当前验证结果见项目根目录 `SHARED_NAVIGATION_STATUS.md`；旧策略的整圈成绩不能用于此版本验收。

## 明天连接真机后的顺序

1. 将不可变压缩包传到 `/home/nvidia/competition_nav/incoming/`，先核对包的 SHA-256。
   使用同一文件重传；传输回包不确定时先查哈希，不创建新的任务。
2. 解压到新的 `releases/<release_id>/`。旧版本、原厂目录、自启动都保留。
3. 在新目录执行以下命令（均不启动行走）：

```bash
bash deploy/install_shadow.sh
source /opt/ros/noetic/setup.bash
source /home/nvidia/sim2real/install/setup.bash
.venv/bin/python -m unittest scripts.test_shared_navigation
.venv/bin/python scripts/replay_navigation.py validation/sensor_replay.npz --output runs/replay.json
.venv/bin/python deploy/shared_navigation.py --preflight --output runs/preflight
.venv/bin/python deploy/shared_navigation.py --duration 20 --output runs/shadow
```

`install_shadow.sh` 只在本版本内建虚拟环境/相机采集程序；复用版本完全匹配且通过检查的原厂 NumPy/OpenCV，
否则只安装到本版本虚拟环境。不会启动或替换原厂控制器。
`--preflight` 检查代码清单、依赖与原厂身份；`shadow` 只订阅传感器并计算，不创建速度或关节发布者。

4. 对比 `preflight.json` 与仿真/本地回放的 `navigation_core_sha256`。
   回放要求 `maximum_command_difference < 1e-5`；Jetson 性能以真实 `camera_cycle_ms_p95` 为准。
   本地已创建 Python 3.8 环境做实际回放与单元测试；这仍不等于 Jetson/ROS/相机检查已经通过。
5. 核验头部反馈、相机内参、50 cm 跑道尺度、相机到机身外参、IMU 轴向和曝光/接收时延。
   保存独立的标定配置与证据；不能直接把占位配置中的两个 `verified` 开关改成真。
6. 标定通过后才进入运动验收：确认现场状态后，经 `--shared-navigation` 的监督器执行。
   标定配置需包含 `calibration_evidence`、`head_reference_rad`（两个头关节）、已验证的矩阵与位置。

机载回放、旁路、标定和受监督闭环未通过之前，不把仿真通过解释为真机可直接跑一圈。

## 执行路径和保护

机载视觉进程仅发 `/competition/lane_request`，内容是期望策略速度。
独立 50 Hz 监督器读取策略观测的第 6–8 项，在实际发布前做同一份 AMP 速度适配。
只有原厂手柄翻译器的受控入口发布 `/cmd_vel`，手柄输入优先；视觉请求超过 0.2 秒失效。
监督器校验运行模式、AMP 身份、关节反馈、标定头部角度和输入数值的有效性，任务按 `--duration` 指定的时长结束。
电量、机身倾斜、实测转速和累计指令量仅写入日志。电池消息缺失或过期也仅记录。
`guard.jsonl` 的 `battery` 记录电量与消息年龄，`attitude` 记录倾角与转速。
旧日志键 `distance_command_budget`、`turn_command_budget` 继续记录指令积分，预算上限在任务摘要中为 `null`。
监督器使用跨版本共享锁，`launch_line_trial.py` 使用任务目录标记防止断网重试重复启动。
退出后恢复原厂手柄翻译器；导航停止后实际脚步仍有停车距离，不能把零请求当作瞬时静止。

该过程完全在机载运行；SSH 的延迟只影响取日志和管理，不放宽传感器新鲜度要求。
累计指令量由原始速度请求积分得到，实测行走距离由视觉里程计提供。

## 旁路输出

- `preflight.json`：Python/NumPy/OpenCV、文件哈希与标定抑制原因。
- `camera_intrinsics.json`、`config.json`：本次成像参数和标定输入。
- `telemetry.jsonl`：视觉观测与定位状态；`processing_ms` 仅为感知用时。
- `commands.json`：候选原始速度、期望策略速度、抑制原因、包含规划的循环用时。
- `summary.json`：旁路发布者数量、包含规划的 `camera_cycle_ms_p95`、最终状态。
- `latest_rgb.jpg`、`latest_overlay.jpg`：当前相机画面与感知结果。

标定未验证时，旁路速度只是诊断候选，不能作为运动依据。
