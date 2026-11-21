<p align="center"><img src="docs/brand/MeguvizRIFE.svg" width="96" alt="MeguvizRIFE"></p>

# MeguvizRIFE

面向 C/C++ 的 RIFE v4.6 双帧插值库。给定相邻两帧与插值位置 `alpha`，生成对应中间帧；提供 CPU 参考实现与由 MeguvizCompute 托管的 Vulkan GPU 推理路径。

## C/C++ 接入

源码构建前先安装含 ncnn 后端的 [MeguvizCompute SDK](https://github.com/Meguvizarchic/MeguvizCompute)（`1.2.2`），并按构建预设配置 `MEGUVIZ_COMPUTE_PREFIX`。安装完成后，将两份 SDK 的安装目录同时加入 `CMAKE_PREFIX_PATH`。[构建预设](CMakePresets.json) · [完整 CMake 示例](examples/external_consumer/)

```cmake
find_package(MeguvizRIFE 1.2.2 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE MeguvizRIFE::core)
```

调用方需自行提供与模型清单一致的 `flownet.param` 和 `flownet.bin`，并遵守模型自身的许可条款。Android 播放器场景请接入 [MeguvizMedia](https://github.com/Meguvizarchic/MeguvizMedia) 统一管理帧生成会话与播放时序。[模型清单](models/catalog/rife46.manifest.json)

## 命令行示例

准备两张相同分辨率的 P6 RGB 图像，在 `pair.json` 中填写图像路径与本地模型目录：

```json
{
  "schema_version": 1,
  "model_directory": "./rife-v4.6",
  "left": "./left.ppm",
  "right": "./right.ppm",
  "alpha": "0.5",
  "output": "./midpoint.ppm",
  "precision_profile": "CPU_FP32_REFERENCE"
}
```

```sh
mgr-cli interpolate --run-spec pair.json --report result.json
```

## 文档

- [公共 C 接口](include/meguviz/rife.h)
- [可用配置与能力](models/catalog/profiles.json)
- [Vulkan 资源与图像保护](docs/adr/0002-compute-gpu-guards.md)
- [推理任务生命周期](docs/INFERENCE_WORK.md)
- [模型上传所有权与清理服务](docs/UPLOAD_OWNERSHIP.md)
- [AI 接入与迁移指南](.agents/skills/meguviz-rife/SKILL.md)

[GPL-3.0 许可证](LICENSE) · [上游实现与许可](src/backends/ncnn/legacy/NOTICE.md)
