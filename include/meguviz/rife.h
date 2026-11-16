#ifndef MEGUVIZ_RIFE_H
#define MEGUVIZ_RIFE_H
#include <meguviz/compute.h>
#include <meguviz/shaderpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#define MGR_API MGC_API
#define MGR_ABI_MAJOR 0u
#define MGR_ABI_MINOR 1u
typedef struct MgrModel MgrModel;
typedef struct MgrPlan MgrPlan;
typedef struct MgrSession MgrSession;
typedef struct MgrResult MgrResult;
enum { MGR_RESULT_INFERRED = 1, MGR_RESULT_ENDPOINT = 2, MGR_RESULT_GUARD_BYPASS = 3 };
enum { MGR_PRECISION_FP32_REFERENCE = 1, MGR_PRECISION_LEGACY_FP16 = 2,
       MGR_PRECISION_VULKAN_FP32_LEGACY = 3, MGR_PRECISION_CPU_FP16_LEGACY = 4 };
enum { MGR_GUARD_STATIC_DETAIL = 1, MGR_GUARD_MOTION = 2 };
/* Observation describes a branch reached by a completed request. UNKNOWN is
 * unobserved; NOT_APPLICABLE and DISABLED are distinct from a cache miss. */
enum { MGR_CACHE_OBSERVATION_UNKNOWN = 0, MGR_CACHE_OBSERVATION_NOT_APPLICABLE = 1,
       MGR_CACHE_OBSERVATION_DISABLED = 2, MGR_CACHE_OBSERVATION_MISS = 3,
       MGR_CACHE_OBSERVATION_HIT = 4, MGR_CACHE_OBSERVATION_SHARED_PENDING = 5 };
/* ENABLED_AT_DISPATCH records the guard bit on the successfully completed
 * dispatch, not that any particular pixel was modified. */
enum { MGR_GUARD_OBSERVATION_UNKNOWN = 0, MGR_GUARD_OBSERVATION_NOT_APPLICABLE = 1,
       MGR_GUARD_OBSERVATION_DISABLED = 2, MGR_GUARD_OBSERVATION_ENABLED_AT_DISPATCH = 3 };
enum { MGR_EXECUTION_ADAPTIVE_FLOW = 1, MGR_EXECUTION_DYNAMIC_SHAPE = 2,
       MGR_EXECUTION_SEPARATE_WARP = 4, MGR_EXECUTION_RAW_GPU_INPUT = 8,
       MGR_EXECUTION_GPU_IO_DISABLED = 16 };
enum { MGR_OPT_PRUNE_UNUSED_TAIL = 1, MGR_OPT_DIRECT_REDUCED_INPUT = 2, MGR_OPT_FUSE_ADD_LEAKY = 4 };
enum { MGR_QUALIFICATION_UNVERIFIED = 0, MGR_QUALIFICATION_REFERENCE = 1 };
enum { MGR_SHADER_PREPROC = 1, MGR_SHADER_POSTPROC = 2, MGR_SHADER_REDUCE = 3,
       MGR_SHADER_FUSED = 4, MGR_SHADER_GPU_UNPACK = 5, MGR_SHADER_GPU_IMAGE_READ = 6,
       MGR_SHADER_GPU_STATISTICS = 7, MGR_SHADER_GPU_PATCH = 8, MGR_SHADER_GPU_IMAGE_WRITE = 9 };
typedef struct MgrShaderBinding {
    MgcHeader header;
    uint32_t kernel;
    uint32_t required_flags;
    MgcHash variant_key;
    MgcHash effective_options_hash;
    MgcHash target_facts_hash;
    MgcHash source_sha256;
    MgcHash private_sdk_fingerprint;
    MgcHash spirv_sha256;
    uint32_t reserved[4];
} MgrShaderBinding;
typedef struct MgrModelDesc {
    MgcHeader header;
    const uint8_t *manifest_utf8;
    uint64_t manifest_size;
    const uint8_t *graph_data;
    uint64_t graph_size;
    const uint8_t *weights_data;
    uint64_t weights_size;
    MgcHash graph_sha256;
    MgcHash weights_sha256;
    MgcHash private_sdk_fingerprint;
    uint32_t required_flags;
    uint32_t reserved[3];
} MgrModelDesc;
typedef struct MgrPlanDesc {
    MgcHeader header;
    uint32_t width;
    uint32_t height;
    uint32_t input_format;
    uint32_t precision_profile;
    uint32_t flow_scale;
    uint32_t guard_flags;
    MgcHash effective_options_hash;
    uint32_t required_flags;
    uint32_t reserved[3];
    /* Optional tail. Zero preserves the legacy graph and preprocessing. */
    uint32_t execution_flags;
    uint32_t optimization_flags;
    uint64_t max_pixels;
    uint32_t num_threads;
    uint32_t reserved_tail;
    /* Borrowed only during create; successful plans retain pack and copy the
     * bindings. An absent kernel binding is a recorded compatibility miss.
     * An explicit binding mismatch is an error, never a runtime fallback.
     * Array stride is the first element's header.struct_size; every element
     * must use that same size. Count <=64 and descriptor size <=65536. */
    MgcShaderpack *shaderpack;
    const MgrShaderBinding *shader_bindings;
    uint32_t shader_binding_count;
    uint32_t shader_required_flags;
    /* Zero selects four internal output Resource slots. Consumer-held READ
     * leases keep slots unavailable; a full pool rejects with BACKPRESSURE. */
    uint32_t output_pool_capacity;
    uint32_t output_pool_reserved;
} MgrPlanDesc;
typedef struct MgrOutputSpec {
    MgcHeader header;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t required_flags;
    uint64_t row_stride_bytes;
    uint32_t reserved[4];
} MgrOutputSpec;
typedef struct MgrRequest {
    MgcHeader header;
    uint64_t request_id;
    MgcLease *left;
    MgcLease *right;
    double alpha;
    MgrOutputSpec output;
    uint32_t required_flags;
    uint32_t reserved[3];
    /* Zero uses the plan's fixed/adaptive policy; adaptive permits 4/6/8. */
    uint32_t flow_scale;
    uint32_t reserved_tail;
    /* Compute resource usage; zero selects the CPU bridge. Kept outside the
     * embedded output structure to preserve old MgrRequest field offsets. */
    uint32_t output_storage_usage;
    uint32_t output_sync_mode;
    /* Actual Compute batch events, dispatched after the execution slot and
     * before final completion. Uses mgr_submit's retained callback context. */
    MgcBatchEventFn batch_event;
} MgrRequest;
typedef struct MgrResultInfo {
    MgcHeader header;
    uint64_t request_id;
    MgcLease *output;
    uint32_t frame_kind;
    uint32_t algorithm_accepted;
    uint32_t guard_outcome;
    uint32_t reserved[3];
    uint32_t unreliable_permille;
    uint32_t patched_tiles;
    uint32_t histogram_permille;
    uint32_t effective_flow_scale;
    int64_t stage_ns[3];
    uint64_t worker_cpu_ns;
    uint32_t worker_cpu_observed;
    uint32_t worker_cpu_reserved;
    uint64_t gpu_elapsed_ns;
    uint64_t gpu_batch_count;
    uint64_t gpu_observed_batch_count;
    uint32_t gpu_elapsed_observed;
    uint32_t gpu_timestamp_valid_bits;
    uint32_t gpu_timestamp_period_bits;
    uint32_t batch_event_coverage_observed;
} MgrResultInfo;
/* Optional versioned result-owned observation. All fields before reserved
 * are required; a partial uint32_t tail is rejected. Values describe this
 * result's request, not a shared plan's most recent execution. */
typedef struct MgrJobObservation {
    MgcHeader header;
    uint64_t request_id;
    uint32_t frame_kind;
    uint32_t effective_flow_scale;
    uint32_t cache_lookup_scale;
    uint32_t static_guard_dispatch;
    uint32_t motion_guard_dispatch;
    uint32_t tensor_cache_left;
    uint32_t tensor_cache_right;
    uint32_t bridge_cache_left;
    uint32_t bridge_cache_right;
    uint32_t tensor_cache_publish_stage_completed;
    uint32_t tensor_cache_published_count;
    uint32_t reserved[4];
} MgrJobObservation;
typedef struct MgrCapabilities {
    MgcHeader header;
    uint32_t precision_profiles; /* bit N describes precision profile N */
    uint32_t flow_scales;        /* bit N describes flow scale N */
    uint32_t input_formats;      /* bit N describes Compute format N */
    uint32_t guard_flags;
    uint32_t gpu_input_output;
    uint32_t host_wait;
    uint32_t workload_qualified;
    uint32_t reserved[3];
} MgrCapabilities;
typedef struct MgrPlanInfo {
    MgcHeader header;
    MgcHash effective_options_hash;
    MgcHash runtime_graph_hash;
    uint32_t precision_profile;
    uint32_t flow_scale;
    uint32_t execution_flags;
    uint32_t optimization_flags;
    uint32_t workload_qualified;
    uint32_t reserved[3];
    uint32_t offline_pipelines;
    uint32_t runtime_fallbacks;
    uint32_t variant_misses;
    uint32_t reserved_tail;
    uint32_t output_pool_capacity;
    uint32_t output_pool_allocated_slots;
} MgrPlanInfo;
typedef void (*MgrCompletionFn)(void *context, uint64_t request_id, MgcStatus status, MgrResult *owned_result);
/* create validates hashes/sizes before parsing and copies all bytes; no network
 * or implicit model download. A model and plan are immutable after success. */
MGR_API MgcStatus mgr_model_create(MgcDomain *domain, const MgrModelDesc *desc, MgrModel **model);
MGR_API void mgr_model_retain(MgrModel *model);
MGR_API void mgr_model_release(MgrModel *model);
/* Implementation availability is separate from device/quality qualification. */
MGR_API MgcStatus mgr_query_capabilities(MgcDomain *domain, MgrCapabilities *capabilities);
MGR_API MgcStatus mgr_plan_create(MgrModel *model, const MgrPlanDesc *desc, MgrPlan **plan);
MGR_API MgcStatus mgr_plan_describe(MgrPlan *plan, MgrPlanInfo *info);
/* Borrowed UTF-8 JSON lives until plan release. Contains captured target facts,
 * sanitized per-kernel Options, source bytes/hashes, compile-request templates
 * and selection events. graphs[] is available for CPU/GPU compatibility plans;
 * CPU plans have no GPU variants/facts. No synthetic GPU facts are produced. */
MGR_API MgcStatus mgr_plan_shader_catalog(MgrPlan *plan, const char **borrowed_utf8, uint64_t *size_bytes);
/* Short immutable execution identity. schema_version=1; graphs has seven fixed
 * main/downscale/upscale/downscale_6/upscale_6/downscale_8/upscale_8 keys, absent
 * graphs are null, present graphs contain the 19 actual Options fields (15
 * shader booleans plus num_threads/use_vulkan_compute/use_packing_layout/
 * use_int8_inference). effective_options_observed is true only for a ready plan.
 * frame_kind=0 and stage_ns=0 are plan defaults; a job owner overlays
 * its actual per-job result. Only canonical graphs belongs in Options identity. */
MGR_API MgcStatus mgr_plan_execution_info(MgrPlan *plan, const char **borrowed_utf8, uint64_t *size_bytes);
MGR_API void mgr_plan_retain(MgrPlan *plan);
MGR_API void mgr_plan_release(MgrPlan *plan);
MGR_API MgcStatus mgr_session_create(MgrPlan *plan, const MgcSessionDesc *desc, MgrSession **session);
MGR_API MgcStatus mgr_session_close(MgrSession *session);
/* ACCEPTED takes one submission reference to each input; rejection takes none.
 * Only internal pool allocation is supported: no external output write target.
 * alpha is finite [0,1]; only exact double 0/1 use ENDPOINT. Interior requests
 * run inference even when their value rounds to 0/1 in FP32 model arithmetic;
 * the model/backend may round or flush subnormal values to its native precision.
 * callback follows Compute context/worker/retirement rules. Result ownership
 * transfers to callback and survives session drain/close until result_release. */
MGR_API MgcStatus mgr_submit(MgrSession *session, const MgrRequest *request, MgrCompletionFn completion,
                     void *context, MgcContextFn retain_context, MgcContextFn release_context,
                     MgcTicket **ticket);
/* Both handles must be valid and ticket must have been accepted by session.
 * A foreign ticket returns INVALID_ARGUMENT without changing its state.
 * Cancellation is logical; drain/completion still await physical retirement. */
MGR_API MgcStatus mgr_cancel(MgrSession *session, MgcTicket *ticket);
MGR_API MgcStatus mgr_drain(MgrSession *session, uint64_t timeout_ns);
/* Explicit service for abandoned model-upload owners. domain may be NULL to
 * service this module's abandoned uploads; a non-NULL domain filters by exact
 * origin domain. SUCCESS means no matching abandoned upload remains, not that
 * all sessions, live engines or consumer outputs are closed. BUSY retains all
 * objects for another call. wait_ns is at most 100 ms per transfer; zero polls
 * without waiting for the device slot. Driver calls have no hard wall deadline.
 * Worker-only; caller keeps its valid domain and loader reference through return. */
MGR_API MgcStatus mgr_uploads_poll(MgcDomain *domain, uint64_t wait_ns);
MGR_API MgcStatus mgr_result_describe(MgrResult *result, MgrResultInfo *info);
/* Available only in builds with MGR_ENABLE_JOB_OBSERVATION=ON. OFF builds
 * return MGC_UNSUPPORTED_CAPABILITY; no observation is fabricated for failed
 * requests, which do not publish an owned MgrResult. */
MGR_API MgcStatus mgr_result_observation(MgrResult *result, MgrJobObservation *observation);
MGR_API void mgr_result_release(MgrResult *result);
#ifdef __cplusplus
}
#endif
#endif
