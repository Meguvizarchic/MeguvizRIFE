---
name: meguviz-rife
description: 在 C/C++ 中接入 MeguvizRIFE 的双帧插值，或把旧 ncnn RIFE 调用迁移到公开模型、计划、会话及结果接口。覆盖 CPU 参考、Vulkan 资源租约和 Android Media 组合的接入选择。
---

# 接入 MeguvizRIFE

## 选择接入层

- 原生算法、服务或离线双图验证使用 `find_package(MeguvizRIFE 0.1 CONFIG REQUIRED)`、`MeguvizRIFE::core` 和 `<meguviz/rife.h>`。找到调用方当前的模型、权重、ncnn/GPU owner、输入与输出的格式和资源归还点。先跑通同一输入上的 CPU 参考或原路径 A/A。
- Android 播放、捕获与显示优先走已配套的 Media API 与单一组合包。新播放器集成用 Media 的帧生成接口维护时钟、补帧预算和原帧回退；直接用 RIFE ABI 时仍由调用方负责这些行为。按已选 SDK 的公开头与包配置匹配依赖，不引入第二个 ncnn owner。

源码构建需同工具链、ABI、CRT 的 Compute ncnn SDK。设置 `MEGUVIZ_COMPUTE_PREFIX`，必要时用对应 `build-info.json` 的私有指纹设 `MGR_REQUIRED_PRIVATE_SDK_FINGERPRINT`。如果编译提示缺少 `<meguviz/shaderpack.h>`，先核对该前缀中的头文件、静态库及 API 是否属于同一套兼容 SDK。调用方提供 `models/catalog/rife46.manifest.json` 匹配的本地 RIFE v4.6 模型，权重随调用方单独管理。

## 从旧引擎迁移

1. 在调用方的 Compute domain 上依次使用 `mgr_model_create`、`mgr_plan_create` 和 `mgr_session_create`，复用 model/plan。描述符置零并以 `MGC_HEADER_INIT(类型)` 设置头。CPU 参考用 `MGR_PRECISION_FP32_REFERENCE` 和 flow 1。Vulkan 用同域的 GPU buffer 或 RGBA8 storage AHB；使用 `mgr_query_capabilities` 与实际资源/fence 探测，区分实现能力和已验证配置。
2. 定义请求尺寸、格式、stride、alpha 和输出类型。只在对应 plan 启用 `MGR_EXECUTION_ADAPTIVE_FLOW` 时逐请求选择 4/6/8。动态尺寸需要启用 `MGR_EXECUTION_DYNAMIC_SHAPE` 并提供 `max_pixels`；依照实际 SDK 的 `MgrPlanDesc` 建立限制。精确 double 0/1 是端点，其他有限区间值是推理请求。输出由 RIFE 拥有的有界池分配。
3. 通过 `mgr_submit` 交付每份输入的独立提交引用。`MGC_ACCEPTED` 转移它们，拒绝时由调用方归还。回调拥有 `MgrResult`，在 `mgr_result_release` 前使用其输出或另行保留 lease，消费者长时间持有会导致有界池回压。用 `mgr_cancel` / `mgr_drain` 等待实际退役后关闭，保留仍可能完成的 owner。
4. 把配置回退放在调用方。AHB SYNC_FD 不可用时，仅在验证了另一同步配置后切换。保留可用的 legacy 或原帧路径。Memory 只在 CPU backing 确有消费需求时经 Compute 的可选 bridge 接入，Memory 借用跟随 Compute lease 和任务寿命。Vulkan/AHB 的生产、导入、同步和设备退休由 Compute/Media 处理。

`MGR_ENABLE_GROUPED_STATISTICS` 默认关闭。仅在内置 Vulkan shader 上试验，并在同设备、同模型、输入和实际配置下对照统计、输出与延迟。使用外部 shaderpack 的计划保留原统计 shader 合同。

## 验证并交付

在 RIFE 源码根目录设置上述 Compute 前缀后，复用：

```sh
cmake --preset host-reference
cmake --build --preset host-reference
ctest --preset host-reference --no-tests=error
```

安装消费使用 `examples/external_consumer`。提供真实本地模型后，按仓库 README 的双图 run-spec 用 `mgr-cli interpolate` 验证实际推理，保留独立报告；`inspect-model` 只核对清单。对 GPU 路径同时验证回压、取消/退役、端点与中间帧，以及实际执行的精度、flow、保护与同步。报告所用身份、路径和证据范围，播放呈现及音画同步在真实宿主验证。
