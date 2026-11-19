<div align="center">

<img src="docs/brand/MeguvizRIFE.svg" width="96" height="96" alt="MeguvizRIFE">

# MeguvizRIFE

**独立 RIFE 插帧算法库**

v4.6 CPU FP32 reference · Vulkan plan · C ABI

<br>

[![License](https://img.shields.io/badge/license-GPL--3.0-00AEEC?style=flat-square&labelColor=2d3a55)](LICENSE)
[![Language](https://img.shields.io/badge/lang-C%2B%2B-DEA584?style=flat-square&labelColor=2d3a55)](https://github.com/Meguvizarchic/MeguvizRIFE)
[![Issues](https://img.shields.io/github/issues/Meguvizarchic/MeguvizRIFE?style=flat-square&labelColor=2d3a55)](https://github.com/Meguvizarchic/MeguvizRIFE/issues)

[构建](#构建) · [真实模型验证](#真实模型验证) · [证据与状态](#证据与状态)

</div>

---

<details>
<summary><b>目录</b></summary>

- [构建](#构建)
- [真实模型验证](#真实模型验证)
- [证据与状态](#证据与状态)

</details>

---

独立 RIFE 算法库，已实现真实 v4.6 CPU FP32 reference、旧 CPU FP16 兼容配置，以及 Compute 管理的 Vulkan FP16/FP32、reduced-flow、GPU buffer/AHB 输入输出和完整算法 guard 链。
公开 C ABI 仅依赖 Compute；不依赖 Media、bil、Android UI，不自行创建或销毁 ncnn GPU 实例。

上一版本已经验证固定权重双 SHA、有界读取、奇数尺寸、alpha 端点、独立旧 CPU 引擎逐字节对照、A/A、失败请求、输出跨 session/plan 关闭存续、纯 C 安装消费及双 ABI 交叉构建。本次功能修改已做必要 Android C++ 与新 shader 编译检查；最终构建、CPU 回归和 GPU 资格统一重新验证，旧版本结果不代替新版本资格。

GPU 路径已经可以由公开 plan 选择，当前没有取得 GPU 数值等价、AHB 生产者/消费者、设备兼容或性能收益资格。真实呈现与音画属于 Media/宿主的独立验证范围。`models/catalog/profiles.json` 同时记录实现能力、默认值和待验证资格。

## 构建

先按 Compute 的说明构建固定 ncnn development SDK，设置 `MEGUVIZ_COMPUTE_PREFIX`，在对应编译器环境运行：

```sh
cmake --preset host-reference
cmake --build --preset host-reference
ctest --preset host-reference
cmake --install build/host-reference --prefix out/install/reference
```

可显式指定 `-DMGR_REQUIRED_PRIVATE_SDK_FINGERPRINT=<实际Compute私有SDK指纹>`；失配会在配置阶段拒绝。
安装包提供 `MeguvizRIFE::core` 与 `MeguvizRIFE::ncnn`，静态链接闭包由 Compute 的安装配置传递。

独立构建默认关闭内部共享 GPU 计划。Android 组合可启用该能力，运行时仍须显式选择。构建指纹与 `build-info.json` 记录 `shared_plan` 的开关状态。

Android 使用 API27、NDK30、`c++_static`，设置 `MEGUVIZ_NDK_ROOT`、`MEGUVIZ_COMPUTE_ANDROID_PREFIX` 后使用 `android-arm64` preset。
ARMv7 可以在独立构建目录显式传 `ANDROID_ABI=armeabi-v7a`。交叉构建不等于设备运行资格。

## 真实模型验证

模型权重外置，不随仓库、SDK、APK 分发。只有 `models/catalog/rife46.manifest.json` 对应的精确 v4.6 模型可加载。

启用 `MGR_BUILD_TESTS` 时，可显式设置 `-DMGR_TEST_MODEL_DIR=<已有模型目录>` 注册真实模型的 context 异常清理与单槽回调重入 CTest。未指定目录时不会注册这两项测试；模型仍由调用方提供。通过这些主机测试不代表 GPU 退休或设备资格。

```sh
mgr-cli inspect-model
mgr-cli inspect-model --manifest models/catalog/rife46.manifest.json
mgr-cli --version-json
mgr-cli interpolate --run-spec pair.json --report result.json
mgr-cli compare --run-spec compare.json --report comparison.json
mgr-cli benchmark --run-spec benchmark.json --report benchmark-report.json
python tests/run_reference.py --binary build/host-reference/mgr-cli --oracle build/host-reference/mgr-legacy-oracle --model-dir /approved/rife-v4.6 --out artifacts/R1/reference
```

`pair.json` 使用 `schema_version:1`、`model_directory`、P6 RGB8 `left/right` 文件、字符串 `alpha`、`output` 路径和 `precision_profile:"CPU_FP32_REFERENCE"`。`CPU_FP16_LEGACY` 恢复旧 `configure(fp16=true)` 的 CPU full-graph/flow1 配置；实际算子精度仍由固定 ncnn 的 CPU 硬件与 layer 支持策略决定，不能仅凭请求名字认定 FP16 生效。Vulkan 双图命令使用 `VULKAN_LEGACY_FP16` 或 `VULKAN_LEGACY_FP32`，可附 `flow_scale`、`guard_flags`、`execution_flags`、`optimization_flags`、`num_threads`；默认 flow=4、static/motion guard 开启。CLI 的 GPU 路径是明确的 CPU 像素桥，输出报告执行结果与质量资格分开。

公开请求以 double 的精确 `0` 和 `1` 判定端点；其他有限区间内值会执行推理，即使转换到模型 FP32 后舍入到端点。模型和设备仍可能按其数值精度舍入或将次正规数归零，输出不保证与高精度模型一致。
`inspect-model --manifest` 只验证指定清单与内置固定模型契约语义一致，不读取或验证模型图和权重，也不证明模型已加载或设备可用。
CLI 生成文件均采用排他创建；已有报告、图像或原始事件文件不会被覆盖，重复执行请指定新路径。失败时若报告路径可安全创建会保留 JSON 失败报告，否则标准错误仍会输出 JSON 失败信息。

`compare.json` 包含 `schema_version:1`、`baseline_report`、`baseline_output`、`candidate_report`、`candidate_output`，这些路径指向两次本 CLI 实际插帧报告及 P6 输出。命令校验文件哈希、模型、输入、形状、精度、图选项和请求配置，记录逐字节差异及保护统计；未观察到的实际 guard 状态、缓存和设备身份仍使质量结论为 `INCONCLUSIVE`，它不授予性能资格。运行时派生图可不同，并单独记录是否相同。

`benchmark.json` 沿用双图/模型/profile 字段，额外要求 `warmup_seconds:30`、`run_seconds:120` 和一个尚不存在的 `raw_events` 路径；alpha 必须严格位于 0 与 1 之间。命令在一个持久 session 中逐任务运行，记录成功、失败、frame kind、CPU/GPU 可观察状态和 submit 到 drain 的墙钟区间；每阶段结束对最后一个输出计算哈希。此区间包含 wrapper/排队，且单次结果没有配对、温度、显示和质量证据，因此不等于性能门禁通过。正式 A/B、持续播放及其他指标仍需独立协议和证据。
示例数据可由 `tests/run_reference.py` 确定性生成。报告将端点与成功推理分开，并明确 CPU reference 不能作为原 GPU 缩流配置的等价 oracle。

## 公共 GPU 接口

`MgrPlanDesc` 保留不可变模型、精度、flow scale 与 guard。`DYNAMIC_SHAPE` 加显式 `max_pixels` 允许每次请求使用不同尺寸；不为每个尺寸重新加载权重。`ADAPTIVE_FLOW` 以4为基础，逐请求可选4/6/8。`mgr_plan_describe` 返回 sanitize 后实际 Options hash 和派生图 hash；`mgr_query_capabilities` 返回实现能力，`workload_qualified=0` 不授予设备资格。

GPU 请求的输入是同 Compute domain 的 RGB8/RGBA8 buffer 或 RGBA8 storage AHB lease。`MgrRequest.output_storage_usage` 为 `MGC_RESOURCE_GPU_BUFFER` 或 `MGC_RESOURCE_ANDROID_AHB`；AHB 输出格式为 RGBA8，`output_sync_mode` 使用 Compute 的 HOST_WAIT/SYNC_FD 模式。输出仍由内部池分配。SYNC_FD 缺能力时明确失败，不切换为 CPU。CPU 输出把该字段保持零。

GPU 算法先读回391 words的原图均值/分块/直方图摘要，再读回146 words的 nonfinite/motion 计数。host 完成原有阈值、邻块和 moving-ring 判定后，由最后 GPU batch 进行分块回填与输出写入。AlgorithmAccepted、ready 与 completion 都覆盖最后写入。GPU 输出没有 CPU 全帧像素往返；这仍是有明确 host 等待的小摘要反馈路径，不是完整异步录图。

输入的完整内容引用及 full-resolution raw/planar history 保留到最终 patch。取消不提前销毁 GPU 对象。成功结果包含 owned READ lease，consumer 释放前一直保活 allocation/domain；session drain/close 不收回它。`MgrResultInfo` 提供 unreliable、patched tiles、histogram、实际 flow scale 和三个阶段耗时。CPU 桥的上传/下载计入对应阶段。

CPU→GPU 桥另有3帧有界 READ staging 缓存，key 使用原 CPU lease 的完整 allocation/content/domain/shape/stride 身份和不可变计划；只在 miss 时上传，同左右内容共享首次上传。每次调用独立 retain 命中 lease，LRU 不驱逐当前调用使用的槽，旧槽须安全取得独占 WRITE 后才能改写或换尺寸。GPU 下载 scratch 只有1个 Resource，CPU 下载 Mat 和必要的尾部补零上传 Mat 同样复用；对齐且容量足够的 CPU 输入直接借用映射范围，不做额外全帧 hash。RIFE 的 prepared cache ID 保持原 CPU 内容身份，避免临时 GPU allocation 使缓存失效。

计划持有真正有界的内部输出 Resource 池，默认4槽，可在 `output_pool_capacity` 选择1–16槽。CPU、GPU buffer 和 AHB 输出共用这套准入规则：按 domain generation、尺寸、stride、format、usage 和同步模式匹配，只有取得新的独占 WRITE lease 才能复用；consumer 持有 READ 输出时不可改写，全部槽不可用则 `mgr_submit` 直接返回 BACKPRESSURE，输入引用仍归 caller。每次内容使用递增 content ID。动态尺寸只在安全取得 WRITE 后替换空闲旧槽，先退役旧 allocation 再分配新尺寸，避免瞬间越过槽位上限。GPU buffer 容量向4字节取整但 visible w/h/stride 不变；CPU 输出不添加像素 padding。计划关闭只释放池自己的 Resource 引用，已交付输出继续保活 Compute domain/allocator。`mgr_plan_describe` 提供容量与已分配槽数。

## 可选离线 shaderpack

`MgrPlanDesc` 尾部接收同 domain 的 Compute `MgcShaderpack` 和 `MgrShaderBinding` 数组。计划复制 bindings 并 retain pack，所有 accepted job 通过计划引用保活到物理退役。九类自有 pipeline 实际调用 Compute `ncnn_shaderpack_pipeline` 选择 raw SPIR-V。模型内置 ncnn layers、Warp 和 AddLeaky 的运行时编译器继续保留。

每个显式 binding 固定 kernel/logical ID、variant key、sanitize 后有效 Options hash、真实 target-facts hash、源码 hash、SPIR-V hash 和 Compute 私有 SDK fingerprint。消费前还检查 pack domain、descriptor 类型顺序、push constant 数量、local size 和借用 pipeline 的实际 device。显式 binding 的任何不匹配都会拒绝 plan。只有未提供某个活动 kernel 的 binding，才在 catalog 记录 variant miss 后使用兼容运行时编译。`mgr_plan_describe` 返回 offline/runtime/miss 计数，plan execution hash 纳入整个 shader catalog 身份。

同一 catalog 的 `graphs[]` 提供 main/downscale/upscale 各图在固定 ncnn 加载并 sanitize 后的实际 `Net.opt`、单图 Options hash 与请求快照，覆盖 CPU 和 `GPU_IO_DISABLED` 兼容计划。CPU 计划没有 GPU variants 或 target facts。基准应比较这些实际读回，不能把整体 plan profile hash 或请求精度名字当作单图有效 Options。

每次测量可以使用短查询 `mgr_plan_execution_info`：schema1、ready 后的 `effective_options_observed=true`、固定七个图键，缺图为null，存在图是共享规范的19个实际 Options 字段。计划默认 `frame_kind=0/stage_ns=0`，由拥有逐 job 结果的绑定层覆盖；只对 canonical graphs 计算 Options 身份。`MgrResultInfo.worker_cpu_ns/worker_cpu_observed` 在 Compute 最终 completion 到达后才复制，覆盖真实执行线程 CPU 时间且不含用户 callback；不可用时 observed=0，不能当零成本。旧同线程路径的 caller CPU 已覆盖执行，汇总时不要再重复相加。

在有 Vulkan 的真实目标上先不带 pack 创建相同 plan，调用 `mgr_plan_shader_catalog` 获取借用 JSON，或运行：

```sh
mgr-cli export-shader-requests --run-spec capture.json --report capture-report.json
```

`capture.json` 沿用双图/模型/profile 字段，增加新的空 `shader_request_output_directory` 与 `shader_toolchain_lock_hash`（取开发机 `mgst --version-json` 的实际工具指纹）。命令只准备 plan 并导出，不执行两图插帧。输出按 kernel 分目录保存精确源码和可直接给 ShaderTools 的 `request.json`，总 `catalog.json` 包含 requested/effective Options、captured target facts、源码与布局预期。GPU 缺失时明确失败，不生成 synthetic facts。

将导出的目录交给开发机后，可准备只覆盖五个新增 GPU 资源内核的 partial compatibility pack：

```sh
python tools/prepare_shaderpack.py --catalog capture/catalog.json --tool /sdk/bin/mgst --out new-pack-generation
```

工具核对捕获、编译收据、实际 SPIR-V、Options、local size 与固定 loader 范围，再生成 `pack/`、`bindings.json` 和 `result.json`。CLI 消费时增加 `shaderpack_directory`、`shaderpack_manifest_sha256`（canonical manifest hash）和 `shader_bindings_file`。其他内核仍记录 miss 并使用运行时编译器；没有 no-runtime opt-in。

目标事实与 Options 使用 Compute 私有 SDK 的权威 canonical JSON。GPU pipeline 选择把真实有效 Options 交回 Compute，由它依据实际 VkDevice 创建时启用的 feature snapshot 检查 Vulkan 目标与 SPIR-V capabilities；不以物理设备 advertised 特性代替 enabled 特性。超出真实能力范围会明确拒绝，不修改 capture 或放宽哈希绕过限制。设备执行与数值资格仍待统一验证。可复用契约见 `models/catalog/shader-contract.json`。

## 优化与回退

`optimization_flags` 的 `PRUNE_UNUSED_TAIL=1`、`DIRECT_REDUCED_INPUT=2`、`FUSE_ADD_LEAKY=4` 可分别开关，默认均为零。裁剪只删除327/332无关且无权重的层；direct reduced 保留源 FP16 storage 舍入并用 FP32 坐标/插值；Add+LeakyReLU 支持 pack1/4，并保留 Add 输出到 storage 后再执行激活的中间舍入。每个 flag 都改变派生图或执行 hash，质量与 A/B 性能仍待统一验证。

`GPU_IO_DISABLED` 保留原 CPU 预处理、同步 Vulkan 图和 CPU 输出配置，限 RGB8。未被证明等价的 geometry 精度改变、新模型与第二后端不进入现有 profile。详细生命周期和退役决策见 `docs/adr/0002-compute-gpu-guards.md`。

## 证据与状态

实际命令、原始输出和逐文件哈希位于 `artifacts/R0` 与 `artifacts/R1`。`ARTIFACTS.json` 指向当前构建与验证收据；`STATUS.json` 逐项列出 R0—R5 的已完成子项和阻塞。
`tests/oracle` 是原 bil 固定提交的未改算法，作为另一个进程运行；新实现不能调用它来制造自身对照。

当前交付是 development SDK。安装包包含 public headers、静态链接 target、精确 Compute 私有 SDK fingerprint、模型 catalog、能力矩阵、来源 notice 和外部 C consumer 源码。R5 开发包准备不等待 bil I5；GPU 质量、性能和多设备门槛仍须真实证据，不称已取得发布资格。实验 R4 不属于默认支持范围。

本库包含 bil GPLv3 派生代码，保留根 `LICENSE`；上游 Warp 的 MIT 许可保留在私有算法目录。模型的研究用途与分发限制单独记录，不由源码许可代替。
