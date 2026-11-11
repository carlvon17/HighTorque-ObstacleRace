# 共享导航与固定真机 AMP：2026-11-11

最新共享源码已通过 12 个物理边界场景、13 个传感器场景，以及 115.90 s 的正常三箱整圈回归。
本轮发现、修复和适用边界记录在 [边界场景测试](CORNER_CASES.md)。

以下记录此前 `full_06` 与 `shared_nav_20261111_02` 包的验证结果。
该版本已在带红、黄、绿三个箱子的仿真赛道完成一圈，并通过独立过线与停车核验。
最终运行是 `output/integration/obstacle_fix_20261111/full_06`，01 起点，D435i 成像参数，传感器噪声开启。
这次没有连接或操作真机；Pro 动力学、相机外参和真实传感器延迟仍待实机验证。

## 现在运行的方案

```text
RGB + 米制深度 + AHRS 姿态
  → 深度平面估计地面，识别白线、终点和箱体
  → 白线/箱面配准 + 地面纹理匹配，估计短时运动
  → 局部障碍记忆 + 横向网格动态规划
  → 路径跟踪 / 箱间减速与侧移闭环，生成 vx、vy、wz
  → 原厂速度滤波、死区适配
  → 固定真机 AMP → 关节控制
```

仿真 `scripts/run_visual.py`、离线回放 `scripts/replay_navigation.py`、机载 `deploy/shared_navigation.py`
都调用 `navigation/runtime.py` 的 `NavigationRuntime`。导航没有世界位姿、世界速度、预设跑道或箱子坐标、
物体分割、真值接触信息接口；世界真值只供独立裁判与事后分析使用。规划器是局部动态规划，不是 MPPI。
原 `deploy/line_follow_core.py` 保留为历史巡线实验，不参与共享导航路径。

基础运控固定为这台 Pro 的 `amp.trt`，配置名 `walk/pi_plus_amp_1110_policy_1.rknn`。
权重、69 维单帧观测与五帧历史、关节顺序、动作缩放、50 Hz 策略、1000 Hz PD、原厂速度滤波与死区均未改动。
仿真使用同一组导出的 FP32 权重，和已保存的 Jetson 引擎输出及真实运行观测/动作记录做数值比对。
这保证运控计算对应，不代表上游 Mini Pi Plus 的物理模型已等同于 Pro 实机。

## 本轮修复

1. **三个箱子分别为红、黄、绿。** 只修改视觉颜色；位置、20 × 20 × 40 cm 尺寸、碰撞盒、跑道和摩擦不变。
   [颜色预览](output/integration/obstacle_fix_20261111/three_color_scene.png)。
2. **地面高度不再由绿色像素决定。** 使用深度点云、重力方向、平面残差和连通面积判断支撑地面，排除箱顶。
   当前深度无法支持地面估计时明确判无效，不用旧高度冒充新观测。
   红黄绿、全绿（与地毯完全同色）、全白等输入回归得到一致的地面和箱体几何结果。
3. **修复遮挡期间的箱体尺寸和位置跳动。** 已知箱子保留测得的方向与遮挡记忆；内部木板碎片不能覆盖外盒尺寸。
   真正旋转 45° 的 20 cm 方箱仍按约 28.3 cm 投影宽度检查，不把它硬缩为 20 cm。
4. **修复视觉定位失效与错误累计。** 配准区分白线、箱面和箱顶轮廓，拒绝不可观测方向；
   纹理不足的直线场景使用深度校验后的米制地面图像相关匹配。全部插值来源必须有有效地面深度。
   连续地面跟踪之后，新几何先作候选，下一帧配准确认后才接回几何参考；弱稀疏光流不能更新几何锚点。
   不积分下发速度，不使用仿真专用关节/接触里程计。
5. **修复窄箱间隙内的跟踪。** 用观测到的直道方向作为侧移参考，先减速、再闭环侧移；
   共同位置修正同步到记忆中的箱子和侧移目标。实际不移动时有限次站立恢复，仍不移动则停车。
6. **修复终点晚判。** 终点仍在视野内时持续更新测得位置，离开视野后由短时视觉里程计判断过线。
   过线法向不随机器人原地转身反向，也不用计时或指令积分预测完成。

## 最终闭环结果

| 项目 | 实测结果 |
|---|---|
| 自主完成时间 | **111.56 s** |
| 独立裁判确认机身实际过线 | 111.54 s |
| 完成触发时距终点线 | 前方 6.7 mm；横向距线中点 11.9 mm |
| 碰箱 / 摔倒 / 双脚均离开跑道 | 均未发生 |
| 完成后观察 | 3.00 s |
| 停车阶段总位移 | 6.27 cm |
| 最后 1 s 位移与指令 | 0.177 mm；三轴原始请求均为 0 |
| 独立整圈核验 | 通过；裁判进度为中心线一圈的 100.32% |

接触在原始 1000 Hz 物理步中检查，脚与跑道重叠在 50 Hz 检查。允许单脚部分越线，
不能将该结果描述为双脚始终完全在线内。裁判没有放宽；事后验收另外增加终点位置检查，
拒绝“绕满一圈但远离终点才宣布完成”的结果。

- [整圈回放：1 倍速 / 50 fps](output/integration/obstacle_fix_20261111/full_06/replay_1x_50fps.mp4)
- [独立轨迹图](output/integration/obstacle_fix_20261111/full_06/trajectory.png)
- [独立评估数据](output/integration/obstacle_fix_20261111/full_06/evaluation.json)
- [完整汇总与文件身份](output/integration/obstacle_fix_20261111/validation.json)

之前的失败记录继续保留：`full_02` 箱间碰撞；`full_03` 终点前定位失效；
`full_04` 内部碎片误增箱宽导致阻塞；`full_05` 虽走完全程但终点晚判 66 cm，按更严格的终点检查不通过。
这些记录不作为最终版本通过证据。

## 测试与复现

- Python 3.11：117 项测试通过，含 AMP 数值对照、相机投影、共享导航、地面/颜色、定位恢复、绕障与终点。
- 本机 Python 3.8：87 项纯导航与接口测试通过。
- 完整 114.56 s 传感器记录（5,728 个控制周期、1,719 帧图像）在 Python 3.11 与 3.8 下重放，
  速度指令与仿真原始记录的最大差值均为 **0**，源文件哈希也逐项一致。
  报告：`full06_replay_python311.json`、`full06_replay_python38.json`，位于本轮结果目录。
  这验证本机跨 Python 计算一致性，不是 Jetson 实时性或真实相机检查。
- 日志：`output/integration/obstacle_fix_20261111/tests_release_final.log` 与 `tests_python38_final.log`。
- 固定 NumPy 1.24.4、OpenCV 4.10.0；入口拒绝依赖版本不符。
  本机旧 OpenCV 5 环境不能作为相同导航的验证环境。

```bash
uv venv --python 3.11 .venv-shared
uv pip install --python .venv-shared/bin/python -r requirements-shared-sim.txt
OPENBLAS_NUM_THREADS=1 .venv-shared/bin/python scripts/run_visual.py --duration 180 --video --record-sensors 150 --output output/sim/shared_three_colors
.venv-shared/bin/python scripts/evaluate_visual.py output/sim/shared_three_colors
.venv-shared/bin/python scripts/render_visual.py output/sim/shared_three_colors
```

## 部署边界

本轮验证的是上述单一起点、场景、噪声种子下的物理闭环与针对性回归，不是所有布局、光照和材质的可靠性保证。
箱体检测不使用颜色标签，但相机仍需有效深度，视觉定位仍需要可观测的几何或纹理。

模型仍采用上游 Mini Pi Plus 动力学；Pro 实机的惯量、碰撞、摩擦与相机安装尚未校准。
AHRS 是带噪姿态测量的仿真，选定起点的 3 cm 横向终点线是明确的场景假设。
跑道与箱子具体布置也保留 README 中注明的建模假设。

新部署包为 `shared_nav_20261111_02`；旧 `_01` 不覆盖。默认旁路，不创建运动发布者。
包内附带整圈开头 6 s 的轻量传感器样例；完整记录与两份整圈重放报告保留在本机结果目录。
包内外参和 IMU 标定标志仍为未验证，原厂文件只读校验。
连接真机后按 [部署步骤](deploy/SHARED_NAVIGATION.md) 完成依赖/哈希预检、同文件回放、真实传感器旁路、
头部/相机/IMU 标定与现场可接管的闭环验收。不能直接把占位标志改成真来跳过标定。
