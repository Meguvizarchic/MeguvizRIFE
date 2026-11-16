# Compute GPU scope 与完整算法 guard

本次实现将实现状态和资格状态分开。Vulkan、AHB 与优化路径可以由 public API 选择；只有统一验证取得对应 oracle、设备和性能证据后，才可提高资格声明。新增优化默认关闭。

`MgrPlanDesc`、`MgrRequest`、`MgrResultInfo` 只在结构尾追加字段。`MgrOutputSpec` 保持原尺寸，输出存储类型放在请求尾部，避免其作为嵌入结构时改变后续字段偏移。旧描述符按 size 读取，缺失尾部按零处理。动态尺寸仍属于同一个不可变计划，尺寸上限和 max_pixels 在计划建立时冻结，逐请求尺寸来自输出规格。更换模型、精度或策略创建新计划。

Compute 的 `with_ncnn_gpu` 提供真实同 domain buffer/image、command 与受控的提交退役。RIFE 每次作用域只录制，所有 Extractor、Mat、pipeline 和下载目标跨整个作用域存活。作用域成功返回代表其物理退役和 host finalize 已完成；多个作用域仍属于接受该请求的同一 Compute ticket。

GPU 图像链分三段。第一段把 RGBA8 storage image 或 RGB/RGBA buffer 转为紧凑 RGB8，生成预处理和 reduced tensor，并以原始 RGB8 整数统计输出 391 words。host 用与原路径相同的 double 除法得到均值、亮度偏移和增益。第二段运行固定 ncnn 图和原有 fused/static-detail/motion kernels，仅读回 146 words 的 nonfinite 和分块计数。host 按原采样分母、邻块扩张和 moving-ring 判定产生 144 项 patch mask。第三段在 GPU 回填最近原图并写最终 GPU buffer 或 AHB。

最终只在第三段退役、nonfinite 与全部 host policy 完成后发布 owned READ lease 和 AlgorithmAccepted。取消只丢弃发布，不提前释放命令和输入。两张原图提交引用与 full-resolution raw/planar 历史至少保留到最后 GPU patch，不能因 reduced tensor 已完成而释放。输出自身通过 Compute lease 保活 allocation/domain，session drain 不回收 consumer 输出。

CPU 桥对 CPU 输入显式上传，在上述 GPU 算法链结束后下载完整输出；这条路径不宣称无全帧往返。仅为保留原配置语义的 `GPU_IO_DISABLED` 继续走旧同步 Vulkan 兼容路径。AHB 只接受 Compute 已验证的 RGBA8 storage 格式；sync_fd 必须由 Compute 实际支持，否则失败，不隐式回 CPU。

输入缓存使用 domain/generation、allocation/generation、content_id、stride、offset、length、尺寸和格式的组合身份；计划自身锁定模型、精度、预处理与优化配置。缓存仅在预处理 batch 成功退役后提交，flow/mask 从不跨 alpha 复用。计划 hash 使用 sanitize 后 ncnn Options，派生图另有 runtime_graph_hash。

R3 的 tail 裁剪、direct reduced 和 Add+LeakyReLU 融合各有独立 flag。裁剪只保留 327/332 祖先，拒绝删除带权重的层。融合仅处理单消费者且紧邻的 residual Add/LeakyReLU，不支持广播时明确失败；pack1/4 显式保留中间 storage 舍入。dispatch 数量不转换成时间收益。任何改变舍入或采样语义的后续实验必须独立 profile。

当前证据仅含必要 C++ 与 shader 编译检查；原有 CPU 证据属于前一版本。完整图数值、guard decisions、AHB 生产者/消费者、取消与 late completion 等统一验证仍待执行。R5 开发包准备不等待 bil 资格；发布资格仍按实际验证范围声明。

可选离线 pack 随不可变 plan retain，任务引用使 pack 和借用 pipeline 至少活到物理退役。pack 的 domain ID/generation 必须与模型 domain 一致。RIFE 自有九类 pipeline 优先使用明确 binding 的 raw SPIR-V，并核对实际 sanitize 后 Options、captured target facts、源/SPIR-V/私有 SDK hash、descriptor 类型、push 常量数量和 local size。显式 binding 失败拒绝创建；只有未配置活动 kernel 才先记录 compatibility miss 再运行 glslang。内置 ncnn、Warp、AddLeaky 仍保留运行时编译，不提供 no-runtime profile。

`mgr_plan_shader_catalog` 从 Compute 权威 canonical JSON 导出 owner-used device macros、requested/effective Options、明确 local size 的源和 ShaderTools 编译请求模板。facts 来自实际 VkDevice 创建时启用的 feature snapshot，不读取物理设备 advertised 支持来假定特性已启用。CLI 用实际开发机 toolchain hash 填充模板，不能在无 GPU 时生成 captured facts。宿主 partial-pack 工具复核 compile receipt 的 semantic identity 与真实 SPIR-V 后才产生 bindings，执行时由带真实 Options 的 Compute factory 检查目标和 capabilities；不得通过改写 facts 或回退未知产物绕过。

内部输出由 plan 的有界 Resource 池提供，默认4槽，显式容量1–16。`mgr_submit` 接管输入之前预留独占 WRITE lease，busy 槽尝试其他槽，全满直接 BACKPRESSURE；拒绝不接管输入。key 包含 domain/generation、shape、stride、format、usage 和同步模式，content ID 单调递增。跨 shape 替换必须先试取得独占 WRITE，释放旧 owner/probe 后再分配，避免瞬时越界扩池。结果在最后写入退役后 seal 为 READ；consumer-held READ 防止复用。plan close 仅移除池的 Resource 引用，输出自身继续保活 Compute。GPU buffer allocation 向4字节取整但 visible geometry/stride 不变，CPU 输出不加 padding。AHB 新 lease 继承原 allocation 的同步模式。准入分配/等待计入阶段0，不能作为免费框架成本。

短查询 `mgr_plan_execution_info` 输出固定 schema1，只有已加载的实际图才置 `effective_options_observed=true`。graphs 的七个固定键以真实 Net.opt 的19个值表示，缺图为null；静态 plan 的 frame_kind/stage_ns 是未知/零，Media 按每个结果覆盖，身份只取 canonical graphs。结果 frame kind 按实际 float32 alpha 判断端点，避免 double alpha 舍入为端点后误计一次真实NN推理。
