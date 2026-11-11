# MicroDuck Athletics

华北五省田径赛场 + MicroDuck 多策略机器人独立工程。

在 BotLab 中打开对应的 `.botlab.zip`，点击运行；点击仿真视窗后按 W/S 前进与后退，A/D 左右转向，松键停止速度指令。X 坐下/站起，R 摔倒恢复。默认从第一个出发点 (0.55, -1.25) 朝 +X 出发，基座初始高度 0.12 m。

复用 BotLab 内置 MicroDuck 的 7 个 ONNX 策略、关节标定和 50 Hz 控制 / 200 Hz 物理。机器人模型中已移除默认无限地面，接触由赛场地毯和三个障碍物承担。实际物理场景通过 `assets/robot/scene.xml` 的 `athletics_field.xml` include 接入；该 BotLab 版本不会自动合成任意 MJCF Scene 声明。

当前是键盘控制的步行工程，不含自动巡线或障碍规划。场地采用照片修订版；障碍物尺寸 20×20×40 cm，内/外/内布局与约 0.5 m 纵向中心距属于照片估计，非现场测量。

地毯视觉使用带 UV 的有限网格，避免浏览器把 plane 延展到无限远；碰撞地面仍为顶面 Z=0 的原尺寸盒体。已在本机 BotLab 浏览器中验证前进、转向和暂停，另通过 8 秒原生站立接触检查；没有声称完成自动绕圈或避障测试。

显示网格厚度与地毯设定一致，为 12 mm；顶面仍为 Z=0.0001 m，UV、纹理和物理接触参数保持原样。原先 1 mm 的薄显示壳容易放大阴影采样伪影；地面没有波纹几何。BotLab 产品修复统一使用受当前 Three.js 支持的 VSM 软阴影，并在切换清晰度时重建阴影及模糊缓存。仅加厚场景不能完全消除低清晰度下的伪影。

在 Competition 根目录运行 `.venv/bin/python scripts/package_botlab.py` 可重新打包。它会先由 `output/athletics/mujoco/field.xml` 和纹理生成 BotLab 的有限网格版本；修改赛场配置后，先运行 `scripts/generate_field.py`。

模型与策略来源：Pollen Robotics microduck / microduck_rl，经本机 BotLab MicroDuck 模板复用。仅保存在本地，未发布。

源码运行：安装 botlab-python 后在本目录执行 `botlab run`。
