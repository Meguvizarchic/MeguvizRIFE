# MeguvizRIFE API 与模块速查

本文档整理 MeguvizRIFE `1.2.2` 的 CMake 构建目标、编译选项、模型清单、公开 C ABI 签名、数据结构与 CLI 子命令，均以仓库源码核实为准。

---

## 1. CMake 目标与构建选项

### 消费方接入

```cmake
find_package(MeguvizRIFE 1.2.2 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE MeguvizRIFE::core)
```

- **可用 Target**：
  - `MeguvizRIFE::core`：静态库 `meguviz_rife`，公开 `<meguviz/rife.h>`，传递依赖 `MeguvizCompute::core`。
  - `MeguvizRIFE::ncnn`：接口兼容别名，指向 `MeguvizRIFE::core`。
- **依赖校验**：`MeguvizRIFEConfig.cmake` 会自动调用 `find_dependency(MeguvizCompute 1.2.2 CONFIG)`，并严格校验 `MeguvizCompute::backend_ncnn_sdk` 的 `INTERFACE_MGC_PRIVATE_SDK_FINGERPRINT` 与 RIFE 构建时记录的 `@MGR_PRIVATE_FP@` 完全一致。
- **工具链与运行库基线**：
  - CMake `3.31+`，C++17（`CMAKE_CXX_STANDARD 17`），PIC 开启。
  - MSVC：`MultiThreaded$<$<CONFIG:Debug>:Debug>`（静态 CRT `/MT` 或 `/MTd`）。
  - Android：`ANDROID_PLATFORM=android-27`（minSdk 27）、`ANDROID_STL=c++_static`。

### 源码构建选项（`CMakeLists.txt`）

| CMake 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MGR_BUILD_CLI` | `ON` | 构建双图参考命令行工具 `mgr-cli` |
| `MGR_BUILD_TESTS` | `ON` | 构建 C ABI 边界与生命周期测试 |
| `MGR_ENABLE_JOB_OBSERVATION` | `OFF` | 启用单任务观测快照 `mgr_result_observation`（会改变构建指纹） |
| `MGR_ENABLE_SHARED_PLAN` | `OFF` | 构建仅供 Android 官方组合使用的私有共享 GPU 计划扩展 |
| `MGR_ENABLE_GROUPED_STATISTICS` | `OFF` | 为内置 Vulkan shader 启用工作组局部归约统计变体（试验用途） |
| `MGR_REQUIRED_PRIVATE_SDK_FINGERPRINT` | `""` | 可选，构建配置期强制核对 Compute 私有 SDK 指纹 |
| `MGR_TEST_MODEL_DIR` | `""` | 可选，指向含 `flownet.param` 与 `flownet.bin` 的本地目录以注册模型测试 |

---

## 2. RIFE v4.6 模型清单与文件约束

RIFE SDK 与示例均不捆绑模型权重。调用方须自行提供符合 `models/catalog/rife46.manifest.json` 规范的模型文件：

- `model_id`：`rife-v4.6`
- 图结构 `flownet.param`：`16532` 字节，SHA256 `724569596bcd1e7b9fa50455c604777ebed99746d2ef40aa86e31b5725f1053c`
- 权重 `flownet.bin`：`10614320` 字节，SHA256 `f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958`
- 几何与编码约束：自顶向下 `RGB8 / 255`，输入节点 `["in0", "in1", "in2"]`，输出节点 `"out0"`，32 倍数填充，内部流图节点 `"327"` 与遮罩节点 `"332"`。

---

## 3. 公开 C ABI 函数签名（`<meguviz/rife.h>`）

宏与版本：`MGR_ABI_MAJOR = 0u`、`MGR_ABI_MINOR = 1u`。回调类型：`typedef void (*MgrCompletionFn)(void *context, uint64_t request_id, MgcStatus status, MgrResult *owned_result);`

### 模型与能力查询

```c
MgcStatus mgr_model_create(MgcDomain *domain, const MgrModelDesc *desc, MgrModel **model);
void      mgr_model_retain(MgrModel *model);
void      mgr_model_release(MgrModel *model);
MgcStatus mgr_query_capabilities(MgcDomain *domain, MgrCapabilities *capabilities);
```

- `mgr_model_create` 会在解析前校验文件大小、双 SHA256 与私有 SDK 指纹，并完整拷贝模型字节（无网络请求，也不做隐式下载）。

### 计划（Plan）管理

```c
MgcStatus mgr_plan_create(MgrModel *model, const MgrPlanDesc *desc, MgrPlan **plan);
MgcStatus mgr_plan_describe(MgrPlan *plan, MgrPlanInfo *info);
MgcStatus mgr_plan_shader_catalog(MgrPlan *plan, const char **borrowed_utf8, uint64_t *size_bytes);
MgcStatus mgr_plan_execution_info(MgrPlan *plan, const char **borrowed_utf8, uint64_t *size_bytes);
void      mgr_plan_retain(MgrPlan *plan);
void      mgr_plan_release(MgrPlan *plan);
```

- `mgr_plan_shader_catalog` 返回随计划生命周期存活的 JSON，包含实际启用的目标设备特征快照、清洗后的 per-kernel Options 与编译请求模板；CPU 计划无 GPU 目录，返回 `MGC_UNSUPPORTED_CAPABILITY`。
- `mgr_plan_execution_info` 返回不可变执行身份 JSON（`schema_version=1`，固定包含 `main`、`downscale`、`upscale`、`downscale_6`、`upscale_6`、`downscale_8`、`upscale_8` 七个键及对应 19 个真实生效选项）。

### 会话、请求调度与上传清理

```c
MgcStatus mgr_session_create(MgrPlan *plan, const MgcSessionDesc *desc, MgrSession **session);
MgcStatus mgr_session_close(MgrSession *session);
MgcStatus mgr_submit(MgrSession *session, const MgrRequest *request, MgrCompletionFn completion,
                     void *context, MgcContextFn retain_context, MgcContextFn release_context,
                     MgcTicket **ticket);
MgcStatus mgr_cancel(MgrSession *session, MgcTicket *ticket);
MgcStatus mgr_drain(MgrSession *session, uint64_t timeout_ns);
MgcStatus mgr_uploads_poll(MgcDomain *domain, uint64_t wait_ns);
```

- `mgr_uploads_poll` 仅可由工作线程调用，且调用期间需保持 `domain` 与动态库加载器引用有效；`wait_ns` 单次传输等待上限为 `100000000` ns（100 ms），传 `0` 仅非阻塞轮询。传 `domain = NULL` 时清理本模块所有遗留的失败模型上传；传非空 `domain` 时仅清理来源一致的上传任务。

### 结果描述与观测

```c
MgcStatus mgr_result_describe(MgrResult *result, MgrResultInfo *info);
MgcStatus mgr_result_observation(MgrResult *result, MgrJobObservation *observation);
void      mgr_result_release(MgrResult *result);
```

- `mgr_result_observation` 仅在编译时启用 `MGR_ENABLE_JOB_OBSERVATION=ON` 时可用；关闭时返回 `MGC_UNSUPPORTED_CAPABILITY`。

---

## 4. 核心描述符与枚举速查

### 精度档位与可用范围（`models/catalog/profiles.json`）

| 常量 | 值 | 计算提供方 | 支持 flow_scale | 支持输入格式 | 输出存储 |
| --- | --- | --- | --- | --- | --- |
| `MGR_PRECISION_FP32_REFERENCE` | `1` | `NCNN_CPU` | `1` | `RGB8` | CPU 托管存储 |
| `MGR_PRECISION_LEGACY_FP16` | `2` | `NCNN_VULKAN` | `1, 2, 4, 6, 8` | `RGB8`, `RGBA8` | CPU 桥、`GPU_BUFFER`、`RGBA8_AHB` |
| `MGR_PRECISION_VULKAN_FP32_LEGACY` | `3` | `NCNN_VULKAN` | `1, 2, 4, 6, 8` | `RGB8`, `RGBA8` | CPU 桥、`GPU_BUFFER`、`RGBA8_AHB` |
| `MGR_PRECISION_CPU_FP16_LEGACY` | `4` | `NCNN_CPU` | `1` | `RGB8` | CPU 托管存储 |

### 执行、优化与保护位标志

- **执行标志（`MgrPlanDesc.execution_flags`）**：
  - `MGR_EXECUTION_ADAPTIVE_FLOW` (`1`)：允许按请求切换 `flow_scale` 为 `4`/`6`/`8`；创建计划时 `pd.flow_scale` 必须设为 `4`（像素数超过 `2073600` 即 1080p 时自动倾向 8）。
  - `MGR_EXECUTION_DYNAMIC_SHAPE` (`2`)：支持同计划内动态尺寸，最大宽高 `4096`，最大总像素 `8847360`（4096×2160）；必须显式填写 `max_pixels` 且不小于初始宽高乘积。
  - `MGR_EXECUTION_SEPARATE_WARP` (`4`)：分拆 Warp 阶段；不可与 `MGR_OPT_DIRECT_REDUCED_INPUT` 同时启用。
  - `MGR_EXECUTION_RAW_GPU_INPUT` (`8`)：直接原始 GPU 输入试验标志；不可与 `GPU_IO_DISABLED` 混用。
  - `MGR_EXECUTION_GPU_IO_DISABLED` (`16`)：禁用 GPU IO 链，仅支持 CPU `RGB8` 与旧同步 Vulkan 兼容录制路径。
- **优化标志（`MgrPlanDesc.optimization_flags`，`flow_scale == 1` 时不允许开启）**：
  - `MGR_OPT_PRUNE_UNUSED_TAIL` (`1`)：裁剪 327/332 祖先之外的冗余尾部层。
  - `MGR_OPT_DIRECT_REDUCED_INPUT` (`2`)：直接降采样输入（保留全分辨率历史）。
  - `MGR_OPT_FUSE_ADD_LEAKY` (`4`)：融合单消费者等形状残差 Add 与 LeakyReLU。
- **Guard 保护标志（`MgrPlanDesc.guard_flags`）**：
  - `MGR_GUARD_STATIC_DETAIL` (`1`)：细节静止区域保护；`MGR_GUARD_MOTION` (`2`)：运动置信度与分块保护。CPU 档位不允许开启任何 Guard。
- **结果帧类型（`MgrResultInfo.frame_kind`）**：
  - `MGR_RESULT_INFERRED` (`1`)：完成模型推理。
  - `MGR_RESULT_ENDPOINT` (`2`)：精确 `0.0` 或 `1.0` 端点直接输出。
  - `MGR_RESULT_GUARD_BYPASS` (`3`)：因 Guard 判定或非有限值保护切回原帧。
- **缓存与 Guard 观测状态（`MgrJobObservation`）**：
  - 缓存状态：`MGR_CACHE_OBSERVATION_UNKNOWN` (`0`)、`NOT_APPLICABLE` (`1`)、`DISABLED` (`2`)、`MISS` (`3`)、`HIT` (`4`)、`SHARED_PENDING` (`5`)。
  - Guard 状态：`MGR_GUARD_OBSERVATION_UNKNOWN` (`0`)、`NOT_APPLICABLE` (`1`)、`DISABLED` (`2`)、`ENABLED_AT_DISPATCH` (`3`)。

### 请求输出与离线 Shaderpack 绑定

- `MgrRequest.output_storage_usage`：`0` 表示通过 CPU 桥输出 `MGC_FORMAT_RGB8`；`MGC_RESOURCE_GPU_BUFFER` (`1`) 输出到 GPU 缓冲；`MGC_RESOURCE_ANDROID_AHB` (`2`) 输出到 `RGBA8` AHB。
- `MgrRequest.output_sync_mode`：仅当 `output_storage_usage == MGC_RESOURCE_ANDROID_AHB` 时有效，可选 `MGC_ANDROID_HOST_WAIT_COMPAT` (`0`) 或 `MGC_ANDROID_GPU_SYNC_FD` (`1`)。
- `MgrPlanDesc.output_pool_capacity`：`0` 默认启用 `4` 个槽位，显式指定范围 `1..16`。
- `MgrPlanDesc.shaderpack` / `shader_bindings`（上限 `64` 项）：共 9 个 RIFE 自有核（`MGR_SHADER_PREPROC` `1` 至 `MGR_SHADER_GPU_IMAGE_WRITE` `9`），仅在计划创建时借用，成功后由计划保留；显式绑定校验失败直接报错，未绑定的活跃核先记一次 `variant_misses` 再回退运行时编译。

---

## 5. `mgr-cli` 命令行工具速查

```sh
mgr-cli inspect-model [--manifest FILE]
mgr-cli interpolate --run-spec pair.json --report result.json
mgr-cli compare --run-spec cmp.json --report cmp-report.json
mgr-cli benchmark --run-spec bench.json --report bench-report.json
mgr-cli export-shader-requests --run-spec capture.json --report capture-report.json
mgr-cli --version-json
```

- **不可覆盖保护**：`--report`、生成输出路径、`raw_events` 以及 `shader_request_output_directory` 若已存在会直接拒绝覆盖。
- **`interpolate` Run-Spec 核心字段**：`schema_version` (`1`)、`model_directory`、`left`（P6 PPM）、`right`（P6 PPM）、`alpha`（字符串如 `"0.5"`）、`output`、`precision_profile`（`CPU_FP32_REFERENCE` / `CPU_FP16_LEGACY` / `VULKAN_LEGACY_FP16` / `VULKAN_LEGACY_FP32`）；可选 `flow_scale`、`guard_flags`、`execution_flags`、`optimization_flags`、`num_threads` 及 `shaderpack_directory` / `shaderpack_manifest_sha256` / `shader_bindings_file`。
- **离线制包配合**：使用 `mgr-cli export-shader-requests` 导出真实设备采集的 `catalog.json` 后，交由 `python tools/prepare_shaderpack.py --catalog <catalog.json> --tool <MeguvizShaderTools-cli> --out <pack-dir>` 生成部分兼容着色器包与绑定文件。

---

## 6. 家族仓库协作边界

- **MeguvizCompute（`1.2.2`）**：拥有 `MgcRuntime`、`MgcDomain`、Vulkan 设备与同步原语、`MgcResource`/`MgcLease` 生命周期及底层提交退役；RIFE 绝不自行创建 GPU 实例。
- **MeguvizMedia（`1.2.2`）**：负责播放器时间线、OES/AHB 采集、呈现调度、音画对齐保护与 Android 组合分发；Android 播放器应直接接入 `MeguvizMedia`。
- **MeguvizMemory（`1.2.2`）**：仅当应用已有由 Memory 托管的 CPU 帧且需零拷贝交由 RIFE 时，才经 Memory `adapters/compute_host` 取得 `MgcLease`；无需对接 Memory 时不要引入任何桥接库。
