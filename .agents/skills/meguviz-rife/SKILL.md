---
name: meguviz-rife
description: 在 C/C++ 原生工程中接入 MeguvizRIFE（1.2.2）双帧插值内核，或将旧版 ncnn RIFE 调用与自建 GPU 实例迁移到公开的模型、计划、会话及结果 C ABI；当需求为 Android 播放器、音画同步、SurfaceTexture 采集或显示调度时，应转交 MeguvizMedia（$meguviz-media）。
---

# MeguvizRIFE 接入与迁移

MeguvizRIFE（`1.2.2`）提供面向 C/C++ 的 RIFE v4.6 双帧插值内核，支持 CPU 参考执行与由 MeguvizCompute 托管的 Vulkan GPU 推理。

## 第 1 阶段：盘点工程现状

修改任何代码前，先读取消费方的以下配置与源码，并输出简要现状评估：

1. **构建与工具链**：读取工程根 `CMakeLists.txt`、`CMakePresets.json` 与依赖配置，确认 CMake 3.31+、C++17+、目标平台 ABI 及 C++ 运行库（MSVC 须使用 `MultiThreaded$<$<CONFIG:Debug>:Debug>`，Android 须使用 API 27+ 与 `c++_static`）。
2. **Compute SDK 基线**：确认环境中已安装含 ncnn 后端的 `MeguvizCompute`（`1.2.2`），记录 `MEGUVIZ_COMPUTE_PREFIX` 路径与构建时的 `INTERFACE_MGC_PRIVATE_SDK_FINGERPRINT`（若校验严格，可通过 `MGR_REQUIRED_PRIVATE_SDK_FINGERPRINT` 锁定）。
3. **现有插帧与模型盘点**：定位旧代码中的 `ncnn::create_gpu_instance` / 自建 `VkDevice`、模型加载方式（`flownet.param` / `flownet.bin` 路径）、输入输出像素格式（`RGB8` / `RGBA8`）、内存载体（CPU / GPU Buffer / Android `AHardwareBuffer`）以及任务回调和资源归还点。

## 第 2 阶段：选择模块与接入依赖

根据调用方定位选择最小依赖组合，避免引入重复的 ncnn 所有者或无关可选桥：

| 使用场景 | 接入方式与目标 | 边界与说明 |
| --- | --- | --- |
| 原生 C/C++ 算法库、离线双图服务或直接调用推理内核 | `find_package(MeguvizRIFE 1.2.2 CONFIG REQUIRED)`，链接 `MeguvizRIFE::core`（或兼容别名 `MeguvizRIFE::ncnn`），包含 `<meguviz/rife.h>` | 仅暴露纯 C ABI，需在 `CMAKE_PREFIX_PATH` 同时包含 MeguvizCompute 与 MeguvizRIFE 安装前缀 |
| Android 播放器、OES 采集、音画同步与显示调度 | **转交 `MeguvizMedia`（`$meguviz-media`）** | 使用官方单一静态组合 AAR 或 `FrameGenerationSession`，严禁再单独引入第二个 ncnn/RIFE 动态库 |
| 需零拷贝消费 `MeguvizMemory` 管理的 CPU 像素 | 通过 `MeguvizCompute` / `MeguvizMemory` 的 `host_bridge` 导入为 `MgcLease` 后再交给 RIFE | 未消费 Memory 数据时严禁盲目引入桥接目标 |

模型由调用方自行随应用分发，SDK 与 APK 均不内置权重；所提供文件须与 `models/catalog/rife46.manifest.json` 的体积及双 SHA256 一致。

## 第 3 阶段：执行最小迁移与生命周期对齐

### 最小接入骨架

```c
// 1. 在外部 MgcDomain 上创建不可变模型与计划（所有描述符先清零并初始化头部）
MgrModelDesc md = {0};
md.header = (MgcHeader)MGC_HEADER_INIT(MgrModelDesc);
md.manifest_utf8 = manifest_bytes; md.manifest_size = manifest_len;
md.graph_data = param_bytes;       md.graph_size = param_len;
md.weights_data = bin_bytes;       md.weights_size = bin_len;
md.graph_sha256 = graph_hash;      md.weights_sha256 = weights_hash;
md.private_sdk_fingerprint = build_info.private_sdk_fingerprint;
MgcStatus st = mgr_model_create(domain, &md, &model);

MgrPlanDesc pd = {0};
pd.header = (MgcHeader)MGC_HEADER_INIT(MgrPlanDesc);
pd.width = w; pd.height = h;
pd.input_format = MGC_FORMAT_RGB8;
pd.precision_profile = MGR_PRECISION_FP32_REFERENCE; // Vulkan 请用 LEGACY_FP16 / VULKAN_FP32_LEGACY
pd.flow_scale = 1; // 开启 MGR_EXECUTION_ADAPTIVE_FLOW 时 plan 必须填 4
st = mgr_plan_create(model, &pd, &plan);

MgcSessionDesc sd = {0};
sd.header = (MgcHeader)MGC_HEADER_INIT(MgcSessionDesc);
sd.max_in_flight = 1;
st = mgr_session_create(plan, &sd, &session);

// 2. 提交请求：仅 MGC_ACCEPTED 转移输入租约引用，拒绝时由调用方释放
MgrRequest req = {0};
req.header = (MgcHeader)MGC_HEADER_INIT(MgrRequest);
req.request_id = id; req.left = left_lease; req.right = right_lease; req.alpha = 0.5;
req.output.header = (MgcHeader)MGC_HEADER_INIT(MgrOutputSpec);
req.output.width = w; req.output.height = h;
req.output.format = MGC_FORMAT_RGB8; req.output.row_stride_bytes = (uint64_t)w * 3;
mgc_lease_retain(left_lease); mgc_lease_retain(right_lease);
if (mgr_submit(session, &req, on_done, ctx, NULL, NULL, &ticket) != MGC_ACCEPTED) {
    mgc_lease_release(left_lease); mgc_lease_release(right_lease);
}

// 3. 回调内读取或保留输出并在用完后立即释放，退出前排空再关闭
// on_done(..., MgrResult *owned_result): mgr_result_describe -> 读取输出 -> mgr_result_release(owned_result)
mgr_cancel(session, ticket);
mgr_drain(session, UINT64_MAX);
mgr_session_close(session);
```

### 迁移与生命周期铁律

- **统一 GPU 所有权**：移除旧代码中的全局 `ncnn::create_gpu_instance` 与自建 `VkDevice`，统一复用外部 `MgcDomain`；先用 `mgr_query_capabilities` 与实际资源/fence 探测核对能力，再决定配置，不把“实现存在”当成“当前设备已达标”。
- **描述符与 ABI**：所有描述符结构体必须先清零并用 `MGC_HEADER_INIT(Type)` 填头，不可传截断的尾部字段。
- **端点直出 vs 推理**：`alpha` 须为 `[0.0, 1.0]` 内的有限 `double`；仅精确 `0.0` / `1.0` 走端点直出（`MGR_RESULT_ENDPOINT`），其余区间值即便在 FP32 下四舍五入接近 0 或 1 仍执行真实推理。
- **自适应倍率、动态尺寸与统计**：仅当计划启用 `MGR_EXECUTION_ADAPTIVE_FLOW` 且 `pd.flow_scale == 4` 时，单次请求才可指定 `4` / `6` / `8`；启用 `MGR_EXECUTION_DYNAMIC_SHAPE` 必须显式提供 `max_pixels`；`MGR_ENABLE_GROUPED_STATISTICS` 默认为 `OFF`，仅可用于内置 Vulkan shader 的同设备试验对照，加载外部 shaderpack 时禁止开启。
- **输出池防回压与残留清理**：RIFE 使用默认 4 槽（可配 1–16 槽）有界输出池；消费者在回调中应尽快调用 `mgr_result_release`（或及时提取数据后再另行保留），长期持有会导致后续提交返回 `MGC_BACKPRESSURE`。`mgr_plan_create` 失败时可能残留已提交的权重上传任务，应在工作线程调用 `mgr_uploads_poll` 推进回收；关闭会话前必须先 `mgr_cancel` 并 `mgr_drain` 等待底层任务结束并归还所有资源。
- **同步与回退归调用方**：AHB `SYNC_FD` 不可用时不会静默回退 CPU，须由调用方在探测验证备用同步方式后切换，并始终保留原帧或旧路径作为兜底。

> **按需深入阅读**：完整 C ABI 签名、编译宏、模型清单与 `mgr-cli` 参数请查阅 [API 与模块速查](references/api-and-modules.md)；旧版引擎迁移对比、高频错误码排查及回退范式请查阅 [迁移对照与排障](references/migration-patterns.md)。

## 第 4 阶段：验证与交付清单

1. **本库源码与基础测试**（在 `MeguvizRIFE` 根目录，提前设置 `MEGUVIZ_COMPUTE_PREFIX`）：
   ```sh
   cmake --preset host-reference
   cmake --build --preset host-reference
   ctest --preset host-reference --no-tests=error
   ```
2. **安装后消费工程验证**：按 `examples/external_consumer` 使用 `find_package(MeguvizRIFE 1.2.2 CONFIG REQUIRED)` 独立编译并运行。
3. **模型清单与双图推理核对**：先运行 `mgr-cli inspect-model`（可加 `--manifest`）核对清单，再备妥本地模型与 P6 PPM 图像运行：
   ```sh
   mgr-cli interpolate --run-spec pair.json --report result.json
   ```
   GPU 路径须额外实测输出池回压、取消后 `drain` 完成状态、`0.0`/`1.0` 端点直出与中间插帧、以及实际执行的精度档位、flow 倍率、Guard 与同步方式。
4. **交付摘要**：在总结中明确写出所用精度档位、flow 倍率、输入/输出存储与同步方式、回退路径、以及实测覆盖的范围。
