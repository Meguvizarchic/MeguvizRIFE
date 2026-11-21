# MeguvizRIFE 迁移对照与排障

本文档提供从旧版 ncnn RIFE（独立 `FrameEngine` / 全局 GPU 实例）迁移到 MeguvizRIFE `1.2.2` 公开 C ABI 的对照范式、生命周期铁律、错误码排查表与安全回退方案。

---

## 1. 存量代码迁移前后对照

### 范式一：GPU 所有权与能力探测

- **旧版写法**：在业务库中自行调用 `ncnn::create_gpu_instance()`、`ncnn::get_gpu_device()`，并在各处直接构造多个 `ncnn::VulkanDevice` 或独立推理实例，导致与播放器或其他模块冲突。
- **MeguvizRIFE 写法**：由宿主或统一引擎层唯一维护 `MgcRuntime` 与 `MgcDomain`；RIFE 先通过 `mgr_query_capabilities(domain, &caps)` 检查能力位，再结合 `MgcDomain` 的真实探测结果决定精度和存储模式。

```c
MgcBuildInfo bi = {0};
bi.header = (MgcHeader)MGC_HEADER_INIT(MgcBuildInfo);
mgc_query_build_info(&bi);

MgcRuntimeDesc rd = {0};
rd.header = (MgcHeader)MGC_HEADER_INIT(MgcRuntimeDesc);
rd.provider = MGC_PROVIDER_NCNN_VULKAN; // CPU 参考请用 MGC_PROVIDER_NCNN_CPU
rd.memory_budget_bytes = 256ull * 1024 * 1024;
mgc_runtime_create(&rd, &runtime);

MgcDomainDesc dd = {0};
dd.header = (MgcHeader)MGC_HEADER_INIT(MgcDomainDesc);
dd.private_sdk_fingerprint = bi.private_sdk_fingerprint;
mgc_domain_acquire(runtime, &dd, &domain);

MgrCapabilities caps = {0};
caps.header = (MgcHeader)MGC_HEADER_INIT(MgrCapabilities);
mgr_query_capabilities(domain, &caps);
```

### 范式二：模型加载、计划冻结与动态尺寸

- **旧版写法**：按文件路径随手加载任意 param/bin；尺寸变化或切换 flow 倍率时直接修改运行中对象的成员变量。
- **MeguvizRIFE 写法**：
  1. `MgrModel` 负责校验与托管冻结的 `rife-v4.6` 清单、双 SHA256 与 SDK 指纹；创建完成后即可释放调用方持有的文件缓冲。
  2. `MgrPlan` 是不可变配置单位：分辨率变化须开启 `MGR_EXECUTION_DYNAMIC_SHAPE` 并设置 `max_pixels`；倍率逐帧切换须开启 `MGR_EXECUTION_ADAPTIVE_FLOW` 且初始 `flow_scale = 4`。一旦更换模型、精度档位或优化标志，创建新的 `MgrPlan` 并排空旧会话后再销毁旧计划。

```c
MgrPlanDesc pd = {0};
pd.header = (MgcHeader)MGC_HEADER_INIT(MgrPlanDesc);
pd.width = 1920;
pd.height = 1080;
pd.input_format = MGC_FORMAT_RGBA8;
pd.precision_profile = MGR_PRECISION_LEGACY_FP16;
pd.flow_scale = 4;
pd.guard_flags = MGR_GUARD_STATIC_DETAIL | MGR_GUARD_MOTION;
pd.execution_flags = MGR_EXECUTION_ADAPTIVE_FLOW | MGR_EXECUTION_DYNAMIC_SHAPE;
pd.max_pixels = 3840ull * 2160ull; // 允许同计划内动态调整至 4K
pd.output_pool_capacity = 4;       // 可配 1..16，0 默认 4 槽
mgr_plan_create(model, &pd, &plan);
```

### 范式三：异步提交、输出租约与池防回压

- **旧版写法**：同步阻塞调用推理函数，调用方直接把外部纹理或缓冲传入函数内部覆写。
- **MeguvizRIFE 写法**：RIFE 只允许使用内部有界输出池分配结果（不支持外部写目标）。提交前由调用方为左右帧各准备一次提交引用（`mgc_lease_retain`），在收到回调后尽快读取并释放结果：

```c
// 提交前增加输入引用
mgc_lease_retain(req.left);
mgc_lease_retain(req.right);
MgcStatus st = mgr_submit(session, &req, on_complete, user_ctx, retain_ctx, release_ctx, &ticket);
if (st != MGC_ACCEPTED) {
    // 拒绝时不转移任何引用，必须由调用方退还
    mgc_lease_release(req.left);
    mgc_lease_release(req.right);
}
```

回调侧遵循以下规则：
```c
void on_complete(void *ctx, uint64_t request_id, MgcStatus status, MgrResult *owned_result) {
    if (status == MGC_SUCCESS && owned_result) {
        MgrResultInfo info = {0};
        info.header = (MgcHeader)MGC_HEADER_INIT(MgrResultInfo);
        if (mgr_result_describe(owned_result, &info) == MGC_SUCCESS) {
            // 读取 info.output（READ lease）或转交给渲染队列
        }
    }
    // 只要不再需要该结果即立即释放，避免占用输出池槽位引发 MGC_BACKPRESSURE
    mgr_result_release(owned_result);
}
```

### 范式四：取消、等待完成与失败上传清理

- **旧版写法**：直接释放上下文或停止线程，可能引发 GPU 正在读取的内存野指针或驱动卡死。
- **MeguvizRIFE 写法**：
  1. `mgr_cancel` 只标记逻辑取消并抑制成功结果发布，**不会立即终止已提交到驱动的工作或提前释放输入**。
  2. 会话关闭前必须先调用 `mgr_drain(session, UINT64_MAX)` 等待所有在途工作彻底结束，随后再调用 `mgr_session_close(session)`。
  3. 若 `mgr_plan_create` 因超时或驱动繁忙失败（如返回 `MGC_BUSY`），可能已有后台上传任务留在预留槽位（模块上限 16 个槽位）。此时不能直接卸载动态库或销毁外部对象，应在工作线程循环调用 `mgr_uploads_poll(domain, wait_ns)` 直至返回 `MGC_SUCCESS` 后再重试建计划或继续关机。

---

## 2. 线程、所有权与寿命铁律

1. **回调函数不得跨 C ABI 抛出 C++ 异常**：`completion`、`retain_context`、`release_context` 与 `batch_event` 均在底层边界内执行，必须保证不抛异常且不重入阻塞。
2. **输出生命周期独立于会话排空**：回调获取的 `MgrResult` 会自动保活输出 `MgcLease` 及其依赖的 `MgcDomain`；即便执行了 `mgr_session_close`，只要消费方尚未调用 `mgr_result_release`，底层资源就不会被提前销毁，此时调用 `mgc_runtime_close` 会正确返回 `MGC_BUSY`。
3. **禁止跨域混用资源**：`MgrModel`、`MgrRequest.left`、`MgrRequest.right` 及可选 `MgcShaderpack` 必须归属同一个 `MgcDomain` 与同一代 `domain_generation`。
4. **统计试验开关约束**：`MGR_ENABLE_GROUPED_STATISTICS=ON` 仅影响内置 Vulkan 统计 shader；启用后会改变构建指纹，若同时导出或校验外部 shaderpack，请保持该选项为 `OFF` 以确保语义和指纹一致。

---

## 3. 高频错误码排查表

| 状态码 | 常见原因 | 排查与修正方法 |
| --- | --- | --- |
| `MGC_INVALID_ARGUMENT` (`-1`) | 描述符未清零、`reserved` 非零、`alpha` 超出 `[0,1]` 或非有限值、`flow_scale` 非法、在未开启自适应时请求 `6`/`8`、开启动态形状未设 `max_pixels`、或 `mgr_uploads_poll` 的 `wait_ns > 100ms` | 用 `= {0}` 清零并用 `MGC_HEADER_INIT` 初始化；检查 `pd.flow_scale == 4` 时再配合 `ADAPTIVE_FLOW`；确认步长 `row_stride_bytes >= width * bpp` |
| `MGC_UNSUPPORTED_ABI` (`-2`) | 传入了半截尾部字段、`struct_size` 不匹配、`required_flags` 非零或 `abi_major` 不符 | 使用当前头文件编译并使用 `MGC_HEADER_INIT(Type)` |
| `MGC_UNSUPPORTED_CAPABILITY` (`-3`) | 在 CPU 档位请求 `RGBA8` / Guard / 非 1 倍 flow / 优化位 / GPU 输出、或关闭观测宏时调用 `mgr_result_observation` | 按 `mgr_query_capabilities` 结果选型；需要观测接口时开启 `-DMGR_ENABLE_JOB_OBSERVATION=ON` 重编 |
| `MGC_INCOMPATIBLE_DOMAIN` (`-4`) | 左右输入或 `MgcShaderpack` 与 `MgrModel` 不在同一个 `MgcDomain` 或 `generation` 已更新 | 保证同一渲染会话中从同一 `MgcDomain` 创建资源与模型 |
| `MGC_MODEL_MISMATCH` (`-5`) | `manifest_utf8`、`flownet.param`、`flownet.bin` 大小或双 SHA256 与冻结清单不匹配 | 使用官方 `models/catalog/rife46.manifest.json` 原文并校验模型文件完整无误 |
| `MGC_ARTIFACT_MISMATCH` (`-6`) | `private_sdk_fingerprint` 或 `MgrShaderBinding` 校验值与当前构建不一致 | 同步重建 Compute 与 RIFE，并使用同一次导出/制包产出的绑定 JSON |
| `MGC_BUSY` (`-7`) | 存在遗留未完成上传导致无法建计划、或尚有未释放的 `MgrResult` 时关闭 `MgcRuntime` | 调用 `mgr_uploads_poll` 完成上传清理；释放所有 `MgrResult` 与会话后再关闭运行时 |
| `MGC_BACKPRESSURE` (`-8`) | 有界输出池（默认 4 槽）已满，通常是消费者持有 `MgrResult` 过久 | 在回调读取完毕后尽快 `mgr_result_release`，或适当调大 `output_pool_capacity`（上限 16） |

---

## 4. 安全回退方案

1. **AHB `SYNC_FD` 不可用**：当探测发现设备或 Compute 域不支持双向 `SYNC_FD` 时，不要强行提交 GPU 同步任务，也不会自动回退 CPU；应由调用方切回已验证的 `MGC_ANDROID_HOST_WAIT_COMPAT`（`0`）或保留原始非插帧路径。
2. **输出池拥堵或硬件异常**：收到 `MGC_BACKPRESSURE` 或非成功提交状态时，立即归还输入引用并直接向用户展示最近一帧原画面，待后续节拍空出槽位再尝试恢复提交，切勿在未完成 `mgr_drain` 前强制销毁 `MgcDomain` 或卸载动态库。
