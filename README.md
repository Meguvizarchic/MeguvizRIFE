<p align="center"><img src="docs/brand/MeguvizRIFE.svg" width="96" alt="MeguvizRIFE"></p>

# MeguvizRIFE

面向 C/C++ 的 RIFE v4.6 插帧库。给定两个相邻帧和插值位置，生成中间帧。提供 CPU 参考实现和由 MeguvizCompute 管理的 Vulkan 路径。

## C/C++ 接入

先安装含 ncnn 后端的 [MeguvizCompute SDK](https://github.com/Meguvizarchic/MeguvizCompute)，再按 RIFE 的构建预设设置 `MEGUVIZ_COMPUTE_PREFIX`。安装后，将两份 SDK 的安装目录加入 `CMAKE_PREFIX_PATH`。[构建预设](CMakePresets.json) · [完整 CMake 示例](examples/external_consumer/)

```cmake
find_package(MeguvizRIFE 1.2.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE MeguvizRIFE::core)
```

使用方提供与清单一致的 `flownet.param` 和 `flownet.bin`，并遵循模型的使用许可。Android 播放器可通过 MeguvizMedia 管理帧生成和播放时序。[模型清单](models/catalog/rife46.manifest.json) · [MeguvizMedia](https://github.com/Meguvizarchic/MeguvizMedia)

## 命令行示例

准备两张相同尺寸的 P6 RGB 图像，在 `pair.json` 中填写文件和模型目录。

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
- [可用配置](models/catalog/profiles.json)
- [Vulkan 资源与图像保护](docs/adr/0002-compute-gpu-guards.md)
- [Codex 接入与迁移](.agents/skills/meguviz-rife/SKILL.md)

[GPL-3.0 许可证](LICENSE) · [上游实现与许可](src/backends/ncnn/legacy/NOTICE.md)
