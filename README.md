# 2026 华北五省：类人机器人田径赛场

> 2026-11-11：默认视觉仿真使用真机 `rt1` 的导航逻辑、D435i 配置和实测视觉时序，AMP 固定为真机 Pro 版本。
> 01 三箱复跑见 [rt1 时序与起步结果](SIMULATION.md#rt1-机载导航复跑)。此前的共享导航与建图结果见 [共享导航状态](SHARED_NAVIGATION_STATUS.md)。

`feat/visual-mapping` 在共享导航中加入 RGB-D 地面纹理关键帧地图，提供整圈地图位置与短时视觉失效后的位移重获。当前地图按 Pro 头部反馈和逐帧机身姿态投影。地图预览、定位误差和三箱赛道进度见 [Pro 任务头姿下的建图验证](SHARED_NAVIGATION_STATUS.md#pro-任务头姿下的建图验证)。

首版已从 01 起点以 **115.90 s** 完成红、黄、绿三箱场景，独立裁判未记录碰箱、出界或摔倒，
并核验终点触发与停车 3 秒。[旧 Mini 模型整圈回放](output/integration/corner_cases_20261111/full_lap_final/replay_1x_50fps.mp4)。
单边线、弯道、遮挡、±30°起步和传感器中断的验证见 [边界场景测试](CORNER_CASES.md)。
该结果来自仿真，尚未验证 Pro 真机部署与闭环表现。

场地部分依据用户指定的规则第 28 页、图 2.1.1 搭建，只包含田径赛场。
原始截图保存在 `reference/athletics_rule_page28.png`。障碍物布局和外观已根据用户补充的实景照片 `reference/athletics_site_photo.jpg` 修订。

## Pi Plus Pro 导航和行走仿真

已增加独立的 MuJoCo 机器人闭环。2026-11-10 起，默认基础运控已同步为真机 Pro 当前
`amp.trt` 的同一组 FP32 权重和控制参数（配置标识 `pi_plus_amp_1110_policy_1.rknn`），
默认机器人采用随真机提供的 Pro 全身 MJCF，相机接在可动头部的 `camera_link`。站姿标定、当前赛道试验及复现命令见 [Pro 头部与相机](SIMULATION.md#pro-头部与相机)。
同步内容、数值对照及运行边界见 [运控同步记录](SIMULATION.md#2026-11-10真机基础运控同步)。
以下整圈视频和成绩使用同步前的 `model_40000.pt`，不能作为当前策略的整圈通过记录。
原导航的默认 `0.22–0.25 m/s` 原始指令曾被真机速度滤波后的死区抑制；当前共享导航已包含原厂指令滤波和死区适配，真机运动响应仍需标定。
保留**已知地图＋仿真位姿导航基线**作为对照，并新增独立的 RGB-D 自主导航入口。
当前共享导航使用 RGB-D 与 AHRS 姿态估计平移，不使用仿真专用的编码器/接触里程计；
关节反馈仍供固定 AMP 与机载监督器使用。下面所述旧版本实现见 [历史自主导航说明](VISUAL_NAVIGATION.md)。
原宽视场配置（`--camera-profile legacy`）三个起点各完成一圈，包含一次箱体整体移动 10 cm 的测试，均未触发当前工程判罚。
可查看 [自主导航整圈回放（1 倍速 / 50 fps）](output/sim/visual_final_v5_01/replay_1x_50fps.mp4)。
此前 D435i 配置以 15 Hz 采集 640×480 并缩小到 320×240。2026-11-10 的 01 起点测试在
修复箱体中心估计和遮挡跟踪后，以 128.18 s 完赛并通过停车核验，见
[测试记录](VISUAL_NAVIGATION.md#2026-11-10窄视场下的箱体中心与遮挡跟踪修复)；修复前 70.24% 进度处阻塞的记录仍保留。
绕障允许单脚部分越线，按“双脚均离开跑道”判罚，不代表双脚始终完全在线内。

运行、结果和实现边界见 [SIMULATION.md](SIMULATION.md)。
真值基线的 [整圈回放](output/sim/start_01/replay.mp4) 和 [轨迹对比图](output/sim/start_01/trajectory.png) 保留作对照。

```sh
# RGB-D 自主导航；默认使用 D435i 实机成像参数，保存感知视频和物理状态
.venv-shared/bin/python scripts/run_visual.py --video --output output/sim/visual_demo

# 原真值导航基线
.venv/bin/python scripts/run_sim.py
# macOS 实时交互窗口必须使用 mjpython：
.venv/bin/mjpython scripts/run_sim.py --viewer
```

## 文件入口

- MuJoCo 提供两个可独立选择的入弯版本。默认的 `corner` 按现场照片让两条直线各自以折角接上大圆弧，文件位于 `output/athletics_corner/`。`smooth` 保留原有的平滑过渡，文件位于 `output/athletics/`。视觉仿真和真值仿真均可用 `--bend-profile corner` 或 `--bend-profile smooth` 切换。[GUI 启动示例](SIMULATION.md#rt1-机载导航复跑)
- `output/athletics/athletics.blend`：可编辑 Blender 场景，单位米，Z 轴向上；白线、障碍物、起点分集合。
- `output/athletics/overview.png`：三维预览。
- `output/athletics/obstacle_detail.png`：照片修订后的木箱障碍区特写。
- `output/athletics/top_view.png`：无标注俯视图。
- `output/athletics/dimensioned_plan.png`：尺寸核对图；标注只属于展示层，不进入 MuJoCo。
- `output/athletics/mujoco/scene.xml`：可独立加载的 MuJoCo 场景。
- `output/athletics/mujoco/field.xml`：场地 include 片段，不覆盖宿主的重力、时间步、灯光或相机。
- `output/athletics/mujoco/assets/carpet.png`：6144 × 4096 场地贴图，包含白线和起点标记。
- `output/athletics/spawn_points.json`：三个出发位置，给的是地面位置，机器人基座高度需按机器人设置。
- `output/athletics/centerline.csv`：逆时针车道中心几何参考；不包含避障规划。
- `output/athletics/measurements.json`：实际尺寸与建模假设。
- `output/athletics/validation.json`、`validation_wasm.json`：实际验证结果。

## 尺寸及建模取舍

| 项目 | 模型数值 | 依据 |
|---|---:|---|
| 小端内边线半径 | 1.00 m | 图示箭头按内边线半径解释 |
| 大端内边线半径 | 2.00 m | 图示箭头按内边线半径解释 |
| 两圆中心距 | 5.00 m | 图示尺寸 |
| 跑道宽 | 0.50 m | 按两条白线中心线间距计算 |
| 白线宽 | 8 mm | 规则给出的 5–10 mm 范围内 |
| 障碍物 | 3 个，0.20 × 0.20 × 0.40 m | 规则明确尺寸 |
| 起点 | 3 个，逆时针出发 | 规则与示意图 |
| 地毯范围 | 10.40 × 6.40 m | 给跑道外围留边的建模取值 |

两版都使用 1 m、2 m 的内边线半径和 5 m 圆心距。平滑版以三次贝塞尔曲线接入大圆弧，内边线周长 **19.9927 m**，车道中心周长 **21.5635 m**。折角版把直线延伸至各自与大圆的交点，内边线周长 **20.1494 m**，车道中心周长 **21.6718 m**。现场照片提供折角形状，具体角度和两侧对称性采用几何推算。规则文字给出的内圈为 19 m，两版与该数值均有差距。

0.50 m 按边线中心间距理解，扣除两侧半条 8 mm 胶带后，可见净宽为 0.492 m。没有采用群聊中的加宽建议。

2026-11-11 按本次测试要求，将 MuJoCo 中 01、02、03 箱分别改为红、黄、绿，
保留照片版本的木板、盖板和缝隙外观。当前颜色见
[三色障碍预览](output/integration/obstacle_fix_20261111/three_color_scene.png)；旧 Blender/照片版本预览保留作历史记录。
内部仍是完整的 20 × 20 × 40 cm 实心碰撞盒；视觉木板不参与接触计算。
此次只改颜色，没有改位置、大小或碰撞。障碍物固定，无质量或可推倒行为。
起点坐标仍按示意图估计，红点直径 6 cm。地毯厚度 12 mm、摩擦参数 `(0.8, 0.005, 0.0001)` 是工程假设，并非实测值。

### 实景照片修订

从小弯端沿障碍区向大弯端看，照片中的左、右两个箱体靠内边线，中间箱体靠外边线。因此将原来的“外-内-外”改为“内-外-内”。照片中的箱体底边接近白线，模型采用箱体边缘与白线中心线对齐，取消原先预留的 5 cm 间隔。

以已知 20 cm 箱宽作局部尺度参照，照片中第一与第三箱体的纵向距离约为数个箱宽。为形成便于调整的初版，采用相邻中心纵向间距 **0.50 m**，纵向箱间净距 **0.30 m**。这只是单张斜拍照片的近似解释，未做相机标定或现场测量；0.50 m 不是规则指定值，也不是测量结果。

| 障碍物 | 位置 X, Y（m） | 布置 | 照片对应 |
|---|---|---|---|
| 01 | 0.65, 1.10 | 靠内线 | 照片左侧箱体 |
| 02 | 1.15, 1.40 | 靠外线 | 照片中间靠后箱体 |
| 03 | 1.65, 1.10 | 靠内线 | 照片右侧靠前箱体 |

第一箱体的绝对纵向位置、精确贴线方式、中间箱体旋转 90°及木板构造均是照片辅助的建模假设。此版本保留规则明确的尺寸和数量，不能据此声称障碍物坐标已严格符合比赛实测布置。照片中的杆、器材箱和室内家具未作为赛道障碍物添加。

旧版完整文件包保留在 `output/history/huabei_athletics_v1_diagram.zip`。

原点位于小圆中心的地面上，地毯顶面 Z=0。三个起点分别为 `(0.55,-1.25)`、`(1.45,-1.25)`、`(2.35,-1.25)`，出发朝 +X，随后逆时针绕行。

## MuJoCo 使用

保持 `scene.xml`、`field.xml` 和 `assets/` 的相对结构不变即可加载：

```python
import mujoco
model = mujoco.MjModel.from_xml_path("output/athletics/mujoco/scene.xml")
data = mujoco.MjData(model)
```

场地接触由一个有限范围的平坦地毯碰撞盒和三个障碍物碰撞盒承担。木箱共 30 个木板视觉盒不参与碰撞，且均位于对应完整碰撞盒包络内。白线及起点烘焙到视觉贴图，没有凸起和碰撞。背景摄影地面、尺寸文字和灯具不进入场地碰撞。

把整个 `mujoco/` 文件夹复制到机器人项目的 `assets/scene/athletics/` 后，在机器人 MJCF 的 `<mujoco>` 根节点下加入：

```xml
<include file="assets/scene/athletics/field.xml"/>
```

如机器人 MJCF 已有同高度的地面，合并时只保留一个接触地面，避免重复接触。场地所有 MJCF 名称使用 `hb_` 前缀。

## BotLab 接入

已建立并在本机 BotLab 中实际运行的 MicroDuck 工程：

- `botlab/microduck_athletics/`：完整源码、机器人、七个 ONNX 策略及赛场资产。
- `output/MicroDuck-Athletics.botlab.zip`：通过“文件 → 打开工程文件 / ZIP…”打开。
- `tmp/botlab/validation.json`：原生站立检查和浏览器手动前进、转向检查记录。

运行后按 W/S 前后走、A/D 转向，松键回到零速度指令。出生点为第一出发点 `(0.55, -1.25, 0.12)`，朝 +X。当前是键盘步行控制，未加入自动巡线和避障。

**已核实的接入方式：** 当前 BotLab 仅自动合成 `procedural://` 的 Scene 资源，任意 MJCF `Scene(...)` 声明不会自动加入物理模型。因此新工程直接在 `assets/robot/scene.xml` 中 include 同目录下的 `athletics_field.xml`，并移除机器人模板自带的无限地面。贴图位于 `assets/robot/assets/carpet.png`。为适配浏览器显示，将地毯视觉平面改成带 UV 的有限网格，物理碰撞盒保持不变。

此工程独立于通用场景导出，不修改 BotLab 产品源码。重新打包运行 `.venv/bin/python scripts/package_botlab.py`。

## 重新生成

在本目录运行，Blender 路径按本机安装位置调整：

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/generate_field.py
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python scripts/build_blender.py
.venv/bin/python scripts/validate_scene.py
.venv/bin/python scripts/validate_scene.py --bend-profile smooth
```

`scene_config.json` 保存几何及位置参数，两个生成器共用同一份数据。调整关键尺寸后也应同步核对 Blender 脚本中的展示标注。直接编辑 `.blend` 不会自动回写 MJCF，需在配置中同步修改并重新生成。

## 验证边界

已使用 Blender 5.2.1 生成、渲染并在桌面应用中重新打开文件。已在 MuJoCo 3.17.0 编译、渲染场景，核对跑道宽度、障碍物尺寸、三个起点；三个起点和三个障碍物顶部共六次 2 秒落球测试均稳定接触。已验证将场景片段移至嵌套项目目录后仍能加载贴图。

已通过本机 BotLab 安装的 `@mujoco/mujoco` **3.8.0 WASM** 加载检查。该 WASM 检查未包含 BotLab 网页导入流程、3.3.8 备用运行时或机器人巡线/避障闭环；上方 Mini Pi Plus 的自主导航结果来自独立的 Python MuJoCo 运行链路。
