#include <meguviz/rife.h>
#include <meguviz/backend_ncnn_sdk.hpp>
#include <meguviz/compute_android.h>
#include "backends/ncnn/legacy/frame_engine.h"
#include "private/data.hpp"
#include "private/alpha.hpp"
#include "private/catalog.hpp"
#include "build_identity.hpp"
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <map>
#include <set>
#include <tuple>
#include <chrono>
#include <mutex>
#include <array>

namespace legacy = meguviz::rife::legacy;
using mgr_detail::J;

struct MgrModel {
    std::atomic<unsigned> refs{1};
    MgcDomain* domain = nullptr;
    std::string graph, weights;
    MgcHash fingerprint{};
    ~MgrModel() { mgc_domain_release(domain); }
};
struct MgrPlan {
    std::atomic<unsigned> refs{1};
    MgrModel* model = nullptr;
    MgrPlanDesc desc{};
    std::unique_ptr<legacy::Engine> engine;
    MgcHash runtime_graph_hash{};
    MgcShaderpack* shaderpack = nullptr;
    std::vector<MgrShaderBinding> shader_bindings;
    using CacheKey = std::tuple<uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t>;
    struct CacheIdentity {int64_t id;uint64_t touch;};
    std::map<CacheKey,CacheIdentity> cache_ids;
    int64_t next_cache_id = 0;
    uint64_t next_cache_touch=0;
    using OutputKey=std::tuple<uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,uint64_t,uint32_t,uint32_t>;
    struct OutputSlot {
        OutputKey key{};MgcResource* resource=nullptr;uint64_t length=0;
        ~OutputSlot(){mgc_resource_release(resource);}
    };
    std::mutex output_mutex;
    std::mutex job_mutex;
#if MGR_SHARED_PLAN_ENABLED
    std::mutex shared_admission_mutex;
    uint64_t shared_pending=0;
    uint32_t shared_owners=0;
    bool shared_ahb=false;
    // Set only by the private factory before the plan is published.
    bool shared_tracking=false;
#endif
    std::vector<std::unique_ptr<OutputSlot>> output_pool;
#if MGR_SHARED_PLAN_ENABLED
    // The RGB and private AHB consumers each retain the original per-plan
    // capacity and independent eviction domain, guarded by output_mutex.
    std::vector<std::unique_ptr<OutputSlot>> shared_ahb_output_pool;
#endif
    uint64_t next_output_content=1;
    using BridgeLayout=std::tuple<uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,uint64_t>;
    struct BridgeSlot {
        CacheKey key{};BridgeLayout layout{};MgcResource* resource=nullptr;MgcLease* read=nullptr;
        uint64_t length=0,age=0;
        ncnn::Mat upload_scratch;
        void clear(){mgc_lease_release(read);read=nullptr;mgc_resource_release(resource);resource=nullptr;upload_scratch.release();length=age=0;}
        ~BridgeSlot(){clear();}
    };
    BridgeSlot bridge_inputs[3],bridge_output;
    ncnn::Mat bridge_download;
    uint64_t bridge_clock=0,next_bridge_content=1;
    ~MgrPlan();
};
struct MgrSession { MgrPlan* plan; MgcSession* compute; };
struct MgrResult {
    MgcLease* output = nullptr; MgrResultInfo info{};
#if defined(MGR_ENABLE_JOB_OBSERVATION)
    legacy::JobObservation observation{};
#endif
    ~MgrResult() { mgc_lease_release(output); }
};

namespace {
template<class T> bool valid(const T* d) {
    return d && d->header.abi_major == MGC_ABI_MAJOR && d->header.struct_size >= sizeof(T);
}
template<class T> MgcStatus read_desc(const T* source, size_t minimum, T& copy) {
    if (!source) return MGC_INVALID_ARGUMENT;
    if (source->header.abi_major != MGC_ABI_MAJOR || source->header.struct_size < minimum) return MGC_UNSUPPORTED_ABI;
    copy = {};
    std::memcpy(&copy, source, std::min<size_t>(source->header.struct_size, sizeof(T)));
    copy.header.struct_size = sizeof(T);
    return MGC_SUCCESS;
}
bool partial_field(uint32_t size, size_t offset, size_t width) {
    return size > offset && size < offset + width;
}
bool partial_u32_array(uint32_t size, size_t offset, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (partial_field(size, offset + i * sizeof(uint32_t), sizeof(uint32_t))) return true;
    return false;
}
bool partial_u64_array(uint32_t size, size_t offset, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (partial_field(size, offset + i * sizeof(uint64_t), sizeof(uint64_t))) return true;
    return false;
}
template<class T, size_t N> bool zero(const T (&v)[N]) { for (auto x : v) if (x) return false; return true; }
std::string hex(const MgcHash& h) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out; out.reserve(64);
    for (auto b : h.bytes) { out += digits[b >> 4]; out += digits[b & 15]; }
    return out;
}
MgcStatus code(legacy::Status s) {
    switch (s.code) {
        case legacy::Code::Ok: return MGC_SUCCESS;
        case legacy::Code::InvalidArgument: return MGC_INVALID_ARGUMENT;
        case legacy::Code::Busy: return MGC_BUSY;
        case legacy::Code::AllocationFailed: return MGC_OUT_OF_MEMORY;
        case legacy::Code::ArtifactMismatch: return MGC_ARTIFACT_MISMATCH;
        case legacy::Code::IncompatibleDomain: return MGC_INCOMPATIBLE_DOMAIN;
        case legacy::Code::DeviceLost: return MGC_DEVICE_LOST;
        case legacy::Code::UnsupportedBackend: case legacy::Code::GpuUnavailable: return MGC_UNSUPPORTED_CAPABILITY;
        case legacy::Code::InvalidModel: case legacy::Code::ModelLoadFailed: case legacy::Code::ModelTooLarge: return MGC_MODEL_MISMATCH;
        default: return MGC_INTERNAL_ERROR;
    }
}
#if NCNN_VULKAN
using DomainDevice = ncnn::VulkanDevice;
#else
using DomainDevice = void;
#endif
MgcStatus destroy_engine(void* p, const DomainDevice*) {
    static_cast<MgrPlan*>(p)->engine.reset();
    return MGC_SUCCESS;
}
MgcStatus create_engine(void* p, const DomainDevice* device) {
    auto& plan = *static_cast<MgrPlan*>(p);
    legacy::Options options;
    options.backend = device ? legacy::Backend::Vulkan : legacy::Backend::Cpu;
    options.maxPixels = plan.desc.max_pixels ? size_t(plan.desc.max_pixels) : size_t(plan.desc.width) * plan.desc.height;
    options.threads = plan.desc.num_threads ? int(plan.desc.num_threads) : 1;
    options.pruneUnusedTail = (plan.desc.optimization_flags & MGR_OPT_PRUNE_UNUSED_TAIL) != 0;
    options.directReducedInput = (plan.desc.optimization_flags & MGR_OPT_DIRECT_REDUCED_INPUT) != 0;
    options.fuseAddLeaky = (plan.desc.optimization_flags & MGR_OPT_FUSE_ADD_LEAKY) != 0;
    options.shaderpack = plan.shaderpack; options.shaderBindings = plan.shader_bindings;
#if NCNN_VULKAN
    options.domainDevice = device;
    options.computeDomain = plan.model->domain;
#endif
    const bool reference = plan.desc.precision_profile == MGR_PRECISION_FP32_REFERENCE;
    const bool cpu=reference||plan.desc.precision_profile==MGR_PRECISION_CPU_FP16_LEGACY;
    if (cpu != (device == nullptr)) return MGC_UNSUPPORTED_CAPABILITY;
    if (!device && (plan.desc.flow_scale != 1 || plan.desc.guard_flags || plan.desc.execution_flags & ~MGR_EXECUTION_DYNAMIC_SHAPE)) return MGC_UNSUPPORTED_CAPABILITY;
    legacy::Status status;
    auto engine = legacy::Engine::create(options, &status);
    if (!engine) return code(status);
    const bool gpuIo = !cpu && !(plan.desc.execution_flags & MGR_EXECUTION_GPU_IO_DISABLED);
    const bool fused = gpuIo && plan.desc.flow_scale > 1 && !(plan.desc.execution_flags & MGR_EXECUTION_SEPARATE_WARP);
    status = engine->configure(plan.desc.precision_profile == MGR_PRECISION_LEGACY_FP16 ||
        plan.desc.precision_profile == MGR_PRECISION_CPU_FP16_LEGACY, int(plan.desc.flow_scale), gpuIo,
        (plan.desc.guard_flags & MGR_GUARD_STATIC_DETAIL) != 0, fused,
        (plan.desc.execution_flags & MGR_EXECUTION_RAW_GPU_INPUT) != 0 || options.directReducedInput,
        (plan.desc.execution_flags & MGR_EXECUTION_ADAPTIVE_FLOW) != 0);
    if (!status.ok()) return code(status);
    engine->setMotionGuard((plan.desc.guard_flags & MGR_GUARD_MOTION) != 0);
    status = engine->load({reinterpret_cast<const uint8_t*>(plan.model->graph.data()), plan.model->graph.size(),
        reinterpret_cast<const uint8_t*>(plan.model->weights.data()), plan.model->weights.size()});
    if (!status.ok()) return code(status);
    plan.engine = std::move(engine);
    return MGC_SUCCESS;
}
struct Job {
    std::atomic<unsigned> refs{1};
    MgrPlan* plan;
    MgrRequest request;
    MgrCompletionFn callback;
    void* context;
    MgcContextFn retain_context;
    MgcContextFn release_context;
    std::unique_ptr<MgrResult> result;
    int64_t bridge_ns[2]{};
#if defined(MGR_ENABLE_JOB_OBSERVATION)
    legacy::CacheObservation bridge_observation[2]{legacy::CacheObservation::NotApplicable,legacy::CacheObservation::NotApplicable};
#endif
    MgcLease* output_writing=nullptr;
    std::shared_ptr<void> gpu_work;
#if MGR_SHARED_PLAN_ENABLED
    bool shared_tracked=false;
#endif
    ~Job() { gpu_work.reset();mgc_lease_release(output_writing);
#if MGR_SHARED_PLAN_ENABLED
        std::vector<std::unique_ptr<MgrPlan::OutputSlot>> retired_outputs;
        if(shared_tracked){std::lock_guard<std::mutex> lock(plan->shared_admission_mutex);
            --plan->shared_pending;
            if(!plan->shared_pending && !plan->shared_owners && plan->shared_ahb){
                plan->shared_ahb=false;std::lock_guard<std::mutex> outputs(plan->output_mutex);
                retired_outputs.swap(plan->shared_ahb_output_pool);
            }}
#endif
        mgr_plan_release(plan); }
};
void retain_job(void* p) {
    auto& job = *static_cast<Job*>(p);
    job.refs.fetch_add(1);
    try {
        if (job.retain_context) job.retain_context(job.context);
    } catch (...) {
        // Compute regards a throwing retain as not acquired. Undo only our
        // own reference; external side effects of the callback are unknown.
        job.refs.fetch_sub(1);
        throw;
    }
}
void release_job(void* p) {
    auto* job = static_cast<Job*>(p);
    // User release may throw; it must not suppress our own reference release.
    try { if (job->release_context) job->release_context(job->context); }
    catch (...) {}
    if (job->refs.fetch_sub(1) == 1) delete job;
}
void drop_caller_job(Job* job) { if (job->refs.fetch_sub(1) == 1) delete job; }
MgcStatus read_info(MgcLease* lease, MgcLeaseInfo& info) {
    info = {}; info.header = MGC_HEADER_INIT(MgcLeaseInfo);
    return mgc_lease_info(lease, &info);
}
bool image_fits(const MgcLeaseInfo& info, const MgrOutputSpec& p, uint32_t format, bool gpu) {
    const uint64_t row = uint64_t(p.width) * (format == MGC_FORMAT_RGBA8 ? 4 : 3);
    return info.format == format && info.width == p.width && info.height == p.height &&
        info.row_stride_bytes >= row && info.row_stride_bytes <= UINT64_MAX / p.height &&
        (uint64_t(p.height) - 1) * info.row_stride_bytes + row <= info.byte_length &&
        (info.access & MGC_ACCESS_READ) && (info.dependency_kind == MGC_DEPENDENCY_READY ||
            (gpu && (info.dependency_kind == MGC_DEPENDENCY_HOST_WAIT || info.dependency_kind == MGC_DEPENDENCY_SYNC_FD)));
}
MgrPlan::CacheKey input_key(const MgcLeaseInfo& i) {
    return {i.domain_id,i.domain_generation,i.allocation_id,i.allocation_generation,
        i.content_id,i.row_stride_bytes,i.byte_offset,i.byte_length,i.width,i.height,i.format};
}
int64_t input_cache_id(MgrPlan& plan, const MgcLeaseInfo& i) {
    const auto key=input_key(i);
    if(++plan.next_cache_touch==0){for(auto& entry:plan.cache_ids)entry.second.touch=0;plan.next_cache_touch=1;}
    const auto found = plan.cache_ids.find(key);
    if (found != plan.cache_ids.end()) {found->second.touch=plan.next_cache_touch;return found->second.id;}
    if(plan.next_cache_id==INT64_MAX)return -1;
    if(plan.cache_ids.size()>=32){auto oldest=std::min_element(plan.cache_ids.begin(),plan.cache_ids.end(),
        [](const auto& x,const auto& y){return x.second.touch<y.second.touch;});plan.cache_ids.erase(oldest);}
    return plan.cache_ids.emplace(key,MgrPlan::CacheIdentity{plan.next_cache_id++,plan.next_cache_touch}).first->second.id;
}
MgcStatus reserve_output(MgrPlan& plan,const MgrRequest& request,MgcLease** out,bool private_ahb) {
    *out=nullptr;
    std::lock_guard<std::mutex> lock(plan.output_mutex);
#if MGR_SHARED_PLAN_ENABLED
    auto& output_pool=private_ahb?plan.shared_ahb_output_pool:plan.output_pool;
#else
    (void)private_ahb;
    auto& output_pool=plan.output_pool;
#endif
    uint64_t domain=0,generation=0;
    auto status=mgc_domain_identity(plan.model->domain,&domain,&generation);if(status)return status;
    if(plan.next_output_content==UINT64_MAX)return MGC_OUT_OF_MEMORY;
    const uint64_t content=plan.next_output_content++;
    const MgrPlan::OutputKey key{domain,generation,request.output.width,request.output.height,request.output.format,
        request.output.row_stride_bytes,request.output_storage_usage,request.output_sync_mode};
    auto writing=[&](MgrPlan::OutputSlot& slot,MgcLease** lease) {
        MgcLeaseDesc d{};d.header=MGC_HEADER_INIT(MgcLeaseDesc);d.content_id=content;
        d.byte_length=slot.length;d.access=MGC_ACCESS_WRITE;
        return mgc_lease_create(slot.resource,&d,lease);
    };
    MgrPlan::OutputSlot* freeSlot=nullptr;
    // An independent WRITE lease is the authority to reuse storage. A resource
    // reference count, stale result pointer or guessed GPU fence is not enough.
    for(auto& slot:output_pool) {
        if(!slot->resource){if(!freeSlot)freeSlot=slot.get();continue;}
        if(slot->key!=key)continue;
        status=writing(*slot,out);
        if(status==MGC_SUCCESS)return status;
        if(status!=MGC_BUSY)return status;
    }
    const uint32_t capacity=plan.desc.output_pool_capacity?plan.desc.output_pool_capacity:4;
    if(!freeSlot&&output_pool.size()<capacity) {
        output_pool.push_back(std::make_unique<MgrPlan::OutputSlot>());freeSlot=output_pool.back().get();
    }
    if(!freeSlot) {
        for(auto& slot:output_pool) {
            if(slot->key==key)continue;
            MgcLease* probe=nullptr;status=writing(*slot,&probe);
            if(status==MGC_BUSY)continue;if(status!=MGC_SUCCESS)return status;
            // Old shape may be retired only after exclusive write acquisition.
            // Drop our resource owner and the probe before allocating replacement
            // so even shape churn cannot temporarily grow the physical pool.
            mgc_resource_release(slot->resource);slot->resource=nullptr;slot->length=0;
            mgc_lease_release(probe);freeSlot=slot.get();break;
        }
    }
    if(!freeSlot)return MGC_BACKPRESSURE;
    struct Allocation {
        MgrPlan* plan;const MgrRequest* request;uint64_t content;
        MgcResource* resource=nullptr;MgcLease* lease=nullptr;
        ~Allocation(){mgc_lease_release(lease);mgc_resource_release(resource);}
    } allocation{&plan,&request,content};
    auto allocate=[](void* p,const DomainDevice*)->MgcStatus {
        auto& a=*static_cast<Allocation*>(p);const auto& r=*a.request;
        if(r.output_storage_usage==MGC_RESOURCE_ANDROID_AHB) {
            MgcAndroidAllocateDesc d{};d.header=MGC_HEADER_INIT(MgcAndroidAllocateDesc);d.width=r.output.width;d.height=r.output.height;
            d.content_id=a.content;d.access=MGC_ACCESS_WRITE;d.sync_mode=r.output_sync_mode;
            auto status=mgc_android_allocate(a.plan->model->domain,&d,&a.lease);if(status)return status;
            return mgc_lease_resource(a.lease,&a.resource);
        }
        MgcResourceDesc d{};d.header=MGC_HEADER_INIT(MgcResourceDesc);d.width=r.output.width;d.height=r.output.height;
        d.format=r.output.format;d.usage=r.output_storage_usage;d.row_stride_bytes=r.output.row_stride_bytes;
        d.capacity_bytes=d.row_stride_bytes*d.height;
        if(d.usage==MGC_RESOURCE_GPU_BUFFER)d.capacity_bytes=(d.capacity_bytes+3)&~uint64_t(3);
        auto status=mgc_resource_create(a.plan->model->domain,&d,&a.resource);if(status)return status;
        MgcLeaseDesc lease{};lease.header=MGC_HEADER_INIT(MgcLeaseDesc);lease.content_id=a.content;
        lease.byte_length=d.capacity_bytes;lease.access=MGC_ACCESS_WRITE;
        return mgc_lease_create(a.resource,&lease,&a.lease);
    };
    // Acquire Compute's domain slot before its GPU allocation/identity locks;
    // never invert that order against a currently executing worker.
    status=request.output_storage_usage?meguviz::with_ncnn_domain(plan.model->domain,allocate,&allocation):allocate(&allocation,nullptr);
    if(status!=MGC_SUCCESS)return status;
    MgcLeaseInfo info{};info.header=MGC_HEADER_INIT(MgcLeaseInfo);
    status=mgc_lease_info(allocation.lease,&info);if(status)return status;
    if(info.byte_offset||info.width!=request.output.width||info.height!=request.output.height||info.format!=request.output.format)return MGC_ARTIFACT_MISMATCH;
    freeSlot->resource=allocation.resource;allocation.resource=nullptr;freeSlot->length=info.byte_length;freeSlot->key=key;
    *out=allocation.lease;allocation.lease=nullptr;return MGC_SUCCESS;
}
#if NCNN_VULKAN
#include "cpu_gpu_bridge.inc"
#endif
MgcStatus execute_job(void* p, MgcTicket* ticket) {
    auto& job = *static_cast<Job*>(p);
    if(meguviz::ncnn_domain_is_lost(job.plan->model->domain))return MGC_DEVICE_LOST;
    std::unique_lock<std::mutex> serial(job.plan->job_mutex,std::try_to_lock);
    if(!serial.owns_lock())return MGC_BUSY;
    if (mgc_private_ticket_cancelled(ticket)) return MGC_CANCELLED;
    try {
        MgcLeaseInfo left{}, right{};
        auto s = read_info(job.request.left, left); if (s != MGC_SUCCESS) return s;
        s = read_info(job.request.right, right); if (s != MGC_SUCCESS) return s;
        struct Unmap { MgcLease* lease; ~Unmap() { if (lease) mgc_lease_unmap(lease); } } ua{job.request.left};
        ua.lease = nullptr;
        MgcResourceDesc resource{}; resource.header = MGC_HEADER_INIT(MgcResourceDesc);
        resource.width = job.request.output.width; resource.height = job.request.output.height;
        resource.format = job.request.output.format; resource.row_stride_bytes = job.request.output.row_stride_bytes;
        resource.usage = job.request.output_storage_usage;
        MgcLease* writing=job.output_writing;
        if(!writing)return MGC_INTERNAL_ERROR;
        if (resource.usage) {
            s = job.plan->engine->processGpu(job.plan->model->domain,job.request.left,job.request.right,writing,
                job.request.alpha,input_cache_id(*job.plan,left),input_cache_id(*job.plan,right),int(job.request.flow_scale),&job.gpu_work);
        } else {
            void *a = nullptr, *b = nullptr, *out = nullptr;
            uint64_t aSize = 0, bSize = 0, outSize = 0;
            s = mgc_lease_map(job.request.left, &a, &aSize); if (s != MGC_SUCCESS) return s;
            ua.lease = job.request.left;
            Unmap ub{nullptr};
            if (job.request.right == job.request.left) { b = a; bSize = aSize; }
            else {
                s = mgc_lease_map(job.request.right, &b, &bSize); if (s != MGC_SUCCESS) return s;
                ub.lease = job.request.right;
            }
            s = mgc_lease_map(writing, &out, &outSize); if (s != MGC_SUCCESS) return s;
            Unmap uo{writing};
            if(job.plan->engine->backend()==legacy::Backend::Vulkan &&
                !(job.plan->desc.execution_flags & MGR_EXECUTION_GPU_IO_DISABLED)) {
#if NCNN_VULKAN
#if defined(MGR_ENABLE_JOB_OBSERVATION)
                s=gpu_cpu_bridge(*job.plan,job.request,left,right,a,b,out,job.bridge_ns,job.bridge_observation,&job.gpu_work);
#else
                s=gpu_cpu_bridge(*job.plan,job.request,left,right,a,b,out,job.bridge_ns,&job.gpu_work);
#endif
#else
                s=MGC_UNSUPPORTED_CAPABILITY;
#endif
            } else {
            const auto status = job.plan->engine->processRgb8(
                {static_cast<const uint8_t*>(a), int(resource.width), int(resource.height), size_t(left.row_stride_bytes), size_t(aSize)},
                {static_cast<const uint8_t*>(b), int(resource.width), int(resource.height), size_t(right.row_stride_bytes), size_t(bSize)},
                job.request.alpha,
                {static_cast<uint8_t*>(out), int(resource.width), int(resource.height), size_t(resource.row_stride_bytes), size_t(outSize)},
                input_cache_id(*job.plan,left), input_cache_id(*job.plan,right), int(job.request.flow_scale),&job.gpu_work);
            s = code(status);
            }
        }
        if(mgc_private_ticket_is_deferred(ticket))return MGC_BUSY;
        job.gpu_work.reset();
        if (s != MGC_SUCCESS || mgc_private_ticket_cancelled(ticket)) return s == MGC_SUCCESS ? MGC_CANCELLED : s;
        auto result = std::make_unique<MgrResult>();
#if defined(MGR_ENABLE_JOB_OBSERVATION)
        // Compute's execution slot still protects this shared plan. Copy the
        // completed observation into the owned result before another job runs.
        result->observation=job.plan->engine->diagnosticJob();
        result->observation.bridgeCache[0]=job.bridge_observation[0];
        result->observation.bridgeCache[1]=job.bridge_observation[1];
#endif
        s=mgc_private_ticket_release_retired_write(ticket,writing);
        if(s!=MGC_SUCCESS)return s;
        s=mgc_lease_publish_read(writing,&result->output);
        if (s != MGC_SUCCESS) return s;
        mgc_lease_release(job.output_writing);job.output_writing=nullptr;
        result->info.header = MGC_HEADER_INIT(MgrResultInfo);
        result->info.request_id = job.request.request_id;
        result->info.output = result->output;
        result->info.algorithm_accepted = 1;
        result->info.frame_kind = mgr_detail::alpha_endpoint(mgr_detail::classify_alpha(job.request.alpha)) ? MGR_RESULT_ENDPOINT : MGR_RESULT_INFERRED;
        result->info.guard_outcome = job.plan->engine->patchedTiles() > 0 ? 1u : 0u;
        result->info.unreliable_permille = job.plan->engine->unreliablePermille();
        result->info.patched_tiles = job.plan->engine->patchedTiles();
        result->info.histogram_permille = job.plan->engine->histogramPermille();
        result->info.effective_flow_scale = job.request.flow_scale ? job.request.flow_scale :
            ((job.plan->desc.execution_flags & MGR_EXECUTION_ADAPTIVE_FLOW) && uint64_t(resource.width)*resource.height > 1920u*1080u ? 8u : job.plan->desc.flow_scale);
        job.plan->engine->timings(result->info.stage_ns);
        result->info.stage_ns[0]+=job.bridge_ns[0];result->info.stage_ns[2]+=job.bridge_ns[1];
        job.result = std::move(result);
        return MGC_SUCCESS;
    } catch (const std::bad_alloc&) { return MGC_OUT_OF_MEMORY; }
    catch (...) { return MGC_INTERNAL_ERROR; }
}
void complete_job(void* p, const MgcCompletion* c) {
    auto& job = *static_cast<Job*>(p);
    auto* result = c->terminal == MGC_TERMINAL_SUCCESS && c->status == MGC_SUCCESS && !c->result_discarded ? job.result.release() : nullptr;
    const auto status = result || c->status != MGC_SUCCESS ? c->status :
        c->result_discarded ? MGC_CANCELLED : MGC_INTERNAL_ERROR;
    if(result&&c->header.abi_major==MGC_ABI_MAJOR&&c->header.struct_size>=offsetof(MgcCompletion,worker_cpu_observed)+sizeof(c->worker_cpu_observed)&&
        c->worker_cpu_observed==1) {
        result->info.worker_cpu_ns=c->worker_cpu_ns;
        result->info.worker_cpu_observed=1;
    }
    if(result&&c->header.abi_major==MGC_ABI_MAJOR&&c->header.struct_size>=
        offsetof(MgcCompletion,batch_event_coverage_observed)+sizeof(c->batch_event_coverage_observed)) {
        result->info.gpu_elapsed_ns=c->gpu_elapsed_ns;
        result->info.gpu_batch_count=c->gpu_batch_count;
        result->info.gpu_observed_batch_count=c->gpu_observed_batch_count;
        result->info.gpu_elapsed_observed=c->gpu_elapsed_observed;
        result->info.gpu_timestamp_valid_bits=c->gpu_timestamp_valid_bits;
        result->info.gpu_timestamp_period_bits=c->gpu_timestamp_period_bits;
        result->info.batch_event_coverage_observed=c->batch_event_coverage_observed;
    }
    if(result&&job.plan->engine->backend()==legacy::Backend::Vulkan&&
        (job.plan->desc.execution_flags&MGR_EXECUTION_GPU_IO_DISABLED)) {
        // This explicitly preserved compatibility recorder predates Compute's
        // scoped batch observer. Never call its unobserved work fully covered.
        result->info.gpu_elapsed_observed=0;
        result->info.batch_event_coverage_observed=0;
    }
    job.callback(job.context, job.request.request_id, status, result);
}
void batch_job(void* p,const MgcBatchEvent* event) {
    auto& job=*static_cast<Job*>(p);
    if(job.request.batch_event)job.request.batch_event(job.context,event);
}
}

MgrPlan::~MgrPlan() {
    // The Engine now serializes its own retired cleanup or hands its complete
    // upload context to a pre-reserved orphan slot. Do not wait for the device
    // slot before giving that destructor the opportunity to defer.
    engine.reset();
    output_pool.clear();
#if MGR_SHARED_PLAN_ENABLED
    shared_ahb_output_pool.clear();
#endif
    for(auto& slot:bridge_inputs)slot.clear();bridge_output.clear();bridge_download.release();
    mgc_shaderpack_release(shaderpack);
    mgr_model_release(model);
}
extern "C" {
MgcStatus mgr_model_create(MgcDomain* domain, const MgrModelDesc* d, MgrModel** out) {
    if (!out) return MGC_INVALID_ARGUMENT; *out = nullptr;
    MgrModelDesc copied{};
    const auto abi = read_desc(d, offsetof(MgrModelDesc, required_flags) + sizeof(uint32_t), copied);
    if (abi != MGC_SUCCESS) return abi;
    if (partial_u32_array(d->header.struct_size, offsetof(MgrModelDesc, reserved), 3)) return MGC_UNSUPPORTED_ABI;
    d = &copied;
    if (d->required_flags) return MGC_UNSUPPORTED_ABI;
    if (!domain || !d->manifest_utf8 || !d->graph_data || !d->weights_data || !d->manifest_size ||
        d->manifest_size > 65536 || !d->graph_size || d->graph_size > 1024*1024 || !d->weights_size ||
        d->weights_size > 32*1024*1024 || d->required_flags || !zero(d->reserved)) return MGC_INVALID_ARGUMENT;
    try {
        const auto manifest = mgr_detail::Parser(std::string(reinterpret_cast<const char*>(d->manifest_utf8), size_t(d->manifest_size))).parse();
        const auto expected = mgr_detail::Parser(meguviz::rife::catalog_manifest).parse();
        if (mgr_detail::dump(manifest) != mgr_detail::dump(expected)) return MGC_MODEL_MISMATCH;
        const std::string graph(reinterpret_cast<const char*>(d->graph_data), size_t(d->graph_size));
        const std::string weights(reinterpret_cast<const char*>(d->weights_data), size_t(d->weights_size));
        if (d->graph_size != uint64_t(manifest.at("graph").number("size_bytes")) ||
            d->weights_size != uint64_t(manifest.at("weights").number("size_bytes")) ||
            mgr_detail::sha256(graph) != manifest.at("graph").text("sha256") ||
            mgr_detail::sha256(weights) != manifest.at("weights").text("sha256") ||
            hex(d->graph_sha256) != manifest.at("graph").text("sha256") ||
            hex(d->weights_sha256) != manifest.at("weights").text("sha256")) return MGC_MODEL_MISMATCH;
        if (hex(d->private_sdk_fingerprint) != meguviz::private_sdk_fingerprint()) return MGC_ARTIFACT_MISMATCH;
        auto model = std::make_unique<MgrModel>();
        mgc_domain_retain(domain); model->domain = domain;
        model->graph = graph; model->weights = weights; model->fingerprint = d->private_sdk_fingerprint;
        *out = model.release(); return MGC_SUCCESS;
    } catch (const std::bad_alloc&) { return MGC_OUT_OF_MEMORY; }
    catch (...) { return MGC_MODEL_MISMATCH; }
}
void mgr_model_retain(MgrModel* m) { if (m) m->refs.fetch_add(1); }
void mgr_model_release(MgrModel* m) { if (m && m->refs.fetch_sub(1) == 1) delete m; }
MgcStatus mgr_query_capabilities(MgcDomain* domain, MgrCapabilities* caps) {
    if (!domain || !caps) return MGC_INVALID_ARGUMENT;
    if (caps->header.abi_major != MGR_ABI_MAJOR || caps->header.struct_size < offsetof(MgrCapabilities,reserved)) return MGC_UNSUPPORTED_ABI;
    if (partial_u32_array(caps->header.struct_size, offsetof(MgrCapabilities,reserved), 3)) return MGC_UNSUPPORTED_ABI;
    MgrCapabilities value{}; value.header = MGC_HEADER_INIT(MgrCapabilities);
    MgcCapability gpu{}; gpu.header = MGC_HEADER_INIT(MgcCapability);
    const auto status = mgc_domain_capability(domain,MGC_CAP_VULKAN_COMPUTE,&gpu);
    if (status != MGC_SUCCESS) return status;
    if (gpu.enabled) {
        value.precision_profiles = (1u<<MGR_PRECISION_LEGACY_FP16)|(1u<<MGR_PRECISION_VULKAN_FP32_LEGACY);
        value.flow_scales = (1u<<1)|(1u<<2)|(1u<<4)|(1u<<6)|(1u<<8);
        value.input_formats = (1u<<MGC_FORMAT_RGB8)|(1u<<MGC_FORMAT_RGBA8);
        value.guard_flags = MGR_GUARD_STATIC_DETAIL|MGR_GUARD_MOTION;
        value.gpu_input_output = 1; value.host_wait = 1;
    } else {
        value.precision_profiles = (1u<<MGR_PRECISION_FP32_REFERENCE)|(1u<<MGR_PRECISION_CPU_FP16_LEGACY);
        value.flow_scales = 1u<<1; value.input_formats = 1u<<MGC_FORMAT_RGB8;
    }
    // Algorithm/device qualification requires the independent final oracle run.
    value.workload_qualified = MGR_QUALIFICATION_UNVERIFIED;
    const auto size = caps->header.struct_size; value.header.struct_size = size;
    std::memcpy(caps,&value,std::min<size_t>(size,sizeof(value))); return MGC_SUCCESS;
}
MgcStatus mgr_plan_create(MgrModel* model, const MgrPlanDesc* d, MgrPlan** out) {
    if (!out) return MGC_INVALID_ARGUMENT; *out = nullptr;
    MgrPlanDesc copied{};
    const auto abi = read_desc(d, offsetof(MgrPlanDesc, required_flags) + sizeof(uint32_t), copied);
    if (abi != MGC_SUCCESS) return abi;
    const auto size = d->header.struct_size;
    if (partial_u32_array(size, offsetof(MgrPlanDesc, reserved), 3) ||
        partial_field(size, offsetof(MgrPlanDesc, execution_flags), sizeof(copied.execution_flags)) ||
        partial_field(size, offsetof(MgrPlanDesc, optimization_flags), sizeof(copied.optimization_flags)) ||
        partial_field(size, offsetof(MgrPlanDesc, max_pixels), sizeof(copied.max_pixels)) ||
        partial_field(size, offsetof(MgrPlanDesc, num_threads), sizeof(copied.num_threads)) ||
        partial_field(size, offsetof(MgrPlanDesc, reserved_tail), sizeof(copied.reserved_tail)) ||
        partial_field(size, offsetof(MgrPlanDesc, shaderpack), sizeof(copied.shaderpack)) ||
        partial_field(size, offsetof(MgrPlanDesc, shader_bindings), sizeof(copied.shader_bindings)) ||
        partial_field(size, offsetof(MgrPlanDesc, shader_binding_count), sizeof(copied.shader_binding_count)) ||
        partial_field(size, offsetof(MgrPlanDesc, shader_required_flags), sizeof(copied.shader_required_flags)) ||
        partial_field(size, offsetof(MgrPlanDesc, output_pool_capacity), sizeof(copied.output_pool_capacity)) ||
        partial_field(size, offsetof(MgrPlanDesc, output_pool_reserved), sizeof(copied.output_pool_reserved))) return MGC_UNSUPPORTED_ABI;
    d = &copied;
    if (d->required_flags) return MGC_UNSUPPORTED_ABI;
    const bool dynamic = (d->execution_flags & MGR_EXECUTION_DYNAMIC_SHAPE) != 0;
    if(d->shader_required_flags)return MGC_UNSUPPORTED_ABI;
    if (!model || (!dynamic && (!d->width || !d->height)) || d->width > 4096 || d->height > 4096 ||
        uint64_t(d->width) * d->height > 4096u * 2160u || (d->input_format != MGC_FORMAT_RGB8 && d->input_format != MGC_FORMAT_RGBA8) ||
        d->required_flags || !zero(d->reserved) || (d->guard_flags & ~3u) ||
        (d->flow_scale != 1 && d->flow_scale != 2 && d->flow_scale != 4 && d->flow_scale != 6 && d->flow_scale != 8) ||
        (d->precision_profile < MGR_PRECISION_FP32_REFERENCE || d->precision_profile > MGR_PRECISION_CPU_FP16_LEGACY) ||
        (d->execution_flags & ~31u) || (d->optimization_flags & ~7u) || d->reserved_tail ||
        d->max_pixels > 4096u*2160u || (dynamic && !d->max_pixels) ||
        (d->max_pixels && d->max_pixels < uint64_t(d->width)*d->height) || d->num_threads > 8 ||
        ((d->execution_flags & MGR_EXECUTION_ADAPTIVE_FLOW) && d->flow_scale != 4) ||
        (d->optimization_flags && d->flow_scale == 1) ||
        ((d->optimization_flags & MGR_OPT_DIRECT_REDUCED_INPUT) && (d->execution_flags & MGR_EXECUTION_SEPARATE_WARP)) ||
        d->shader_required_flags || d->shader_binding_count > 64 || d->output_pool_capacity>16 || d->output_pool_reserved ||
        (d->shaderpack && (d->execution_flags & MGR_EXECUTION_GPU_IO_DISABLED)) ||
        (d->shader_binding_count && (!d->shaderpack || !d->shader_bindings)) ||
        ((d->execution_flags & MGR_EXECUTION_GPU_IO_DISABLED) && (d->input_format != MGC_FORMAT_RGB8 ||
            (d->execution_flags & MGR_EXECUTION_RAW_GPU_INPUT) || (d->guard_flags & MGR_GUARD_STATIC_DETAIL) ||
            (d->optimization_flags & MGR_OPT_DIRECT_REDUCED_INPUT)))) return MGC_INVALID_ARGUMENT;
    const bool cpu=d->precision_profile==MGR_PRECISION_FP32_REFERENCE||d->precision_profile==MGR_PRECISION_CPU_FP16_LEGACY;
    if (cpu && (d->input_format != MGC_FORMAT_RGB8 || d->flow_scale != 1 || d->guard_flags || d->optimization_flags || d->shaderpack || (d->execution_flags & ~MGR_EXECUTION_DYNAMIC_SHAPE))) return MGC_UNSUPPORTED_CAPABILITY;
    try {
        auto plan = std::make_unique<MgrPlan>();
        mgr_model_retain(model); plan->model = model; plan->desc = *d;
        plan->output_pool.reserve(d->output_pool_capacity?d->output_pool_capacity:4);
        if(d->shaderpack) {
            uint64_t pack_id=0,pack_generation=0,domain_id=0,domain_generation=0;
            auto status=mgc_shaderpack_domain_identity(d->shaderpack,&pack_id,&pack_generation);if(status)return status;
            status=mgc_domain_identity(model->domain,&domain_id,&domain_generation);if(status)return status;
            if(pack_id!=domain_id||pack_generation!=domain_generation)return MGC_INCOMPATIBLE_DOMAIN;
            mgc_shaderpack_retain(d->shaderpack);plan->shaderpack=d->shaderpack;
        }
        std::set<uint32_t> shader_ids;
        const auto* bindingBytes=reinterpret_cast<const uint8_t*>(d->shader_bindings);
        uint32_t bindingStride=0;
        if(d->shader_binding_count) {
            MgcHeader header{};std::memcpy(&header,bindingBytes,sizeof(header));bindingStride=header.struct_size;
            if(header.abi_major!=MGC_ABI_MAJOR||bindingStride<offsetof(MgrShaderBinding,reserved)||bindingStride>65536||
               partial_u32_array(bindingStride,offsetof(MgrShaderBinding,reserved),4))return MGC_UNSUPPORTED_ABI;
            const uint64_t span=uint64_t(bindingStride)*(d->shader_binding_count-1)+std::min<size_t>(bindingStride,sizeof(MgrShaderBinding));
            if(span>SIZE_MAX||span>UINTPTR_MAX-reinterpret_cast<uintptr_t>(bindingBytes))return MGC_INVALID_ARGUMENT;
            plan->shader_bindings.reserve(d->shader_binding_count);
        }
        for(uint32_t i=0;i<d->shader_binding_count;i++) {
            const auto* element=bindingBytes+size_t(i)*bindingStride;
            MgcHeader header{};std::memcpy(&header,element,sizeof(header));
            if(header.abi_major!=MGC_ABI_MAJOR||header.struct_size!=bindingStride)return MGC_UNSUPPORTED_ABI;
            MgrShaderBinding binding{};
            std::memcpy(&binding,element,std::min<size_t>(bindingStride,sizeof(binding)));
            binding.header.struct_size=sizeof(binding);
            if(binding.required_flags)return MGC_UNSUPPORTED_ABI;
            if(binding.required_flags||!zero(binding.reserved)||binding.kernel<1||binding.kernel>9||!shader_ids.insert(binding.kernel).second||
                zero(binding.variant_key.bytes)||zero(binding.effective_options_hash.bytes)||zero(binding.target_facts_hash.bytes)||
                zero(binding.source_sha256.bytes)||zero(binding.spirv_sha256.bytes)||
                hex(binding.private_sdk_fingerprint)!=meguviz::private_sdk_fingerprint())return MGC_ARTIFACT_MISMATCH;
            plan->shader_bindings.push_back(binding);
        }
        plan->desc.shaderpack=plan->shaderpack;
        plan->desc.shader_bindings=plan->shader_bindings.empty()?nullptr:plan->shader_bindings.data();
#if NCNN_VULKAN
        // Service abandoned uploads before ordinary GPU admission. CPU plan
        // creation never waits for this GPU-only retirement service.
        if (d->precision_profile != MGR_PRECISION_FP32_REFERENCE &&
            d->precision_profile != MGR_PRECISION_CPU_FP16_LEGACY)
            legacy::Engine::pollRetiredUploads(0, model->domain);
#endif
        const auto s = meguviz::with_ncnn_domain(model->domain, create_engine, plan.get());
        if (s != MGC_SUCCESS) return s;
        const auto options_hash=mgr_detail::sha256(d->precision_profile==MGR_PRECISION_FP32_REFERENCE && d->num_threads<=1 ?
            std::string(meguviz::rife::reference_options) : plan->engine->effectiveOptions()+
                ";optimizations="+std::to_string(d->optimization_flags));
        if(!zero(d->effective_options_hash.bytes)&&hex(d->effective_options_hash)!=options_hash)return MGC_ARTIFACT_MISMATCH;
        const auto graph_hash=mgr_detail::sha256(plan->engine->runtimeGraph());
        for(size_t i=0;i<32;i++) {
            plan->desc.effective_options_hash.bytes[i]=uint8_t(std::stoul(options_hash.substr(i*2,2),nullptr,16));
            plan->runtime_graph_hash.bytes[i]=uint8_t(std::stoul(graph_hash.substr(i*2,2),nullptr,16));
        }
        *out = plan.release(); return MGC_SUCCESS;
    } catch (const std::bad_alloc&) { return MGC_OUT_OF_MEMORY; }
    catch (...) { return MGC_INTERNAL_ERROR; }
}
static MgcStatus snapshot_output_pools(MgrPlan* plan,uint32_t* capacity,uint32_t* allocated) {
    if(!plan||!capacity||!allocated)return MGC_INVALID_ARGUMENT;
#if MGR_SHARED_PLAN_ENABLED
    // Global order matches shared-plan enable/release and Job finalization.
    std::lock_guard<std::mutex> admission(plan->shared_admission_mutex);
#endif
    std::lock_guard<std::mutex> outputs(plan->output_mutex);
    const uint64_t per_pool=plan->desc.output_pool_capacity?plan->desc.output_pool_capacity:4;
    uint64_t total_capacity=per_pool, total_allocated=0;
    for(const auto& slot:plan->output_pool)if(slot->resource)++total_allocated;
#if MGR_SHARED_PLAN_ENABLED
    if(!plan->shared_ahb && !plan->shared_ahb_output_pool.empty())return MGC_INTERNAL_ERROR;
    if(plan->shared_ahb)total_capacity+=per_pool;
    for(const auto& slot:plan->shared_ahb_output_pool)if(slot->resource)++total_allocated;
#endif
    if(total_capacity>UINT32_MAX||total_allocated>UINT32_MAX||total_allocated>total_capacity)return MGC_INTERNAL_ERROR;
    *capacity=uint32_t(total_capacity);*allocated=uint32_t(total_allocated);return MGC_SUCCESS;
}
#if MGR_SHARED_PLAN_ENABLED
// The plan is local until this function publishes *out; no submit can race setup.
MgcStatus mgr_private_create_share_capable_plan(MgrModel* model,const MgrPlanDesc* desc,MgrPlan** out) {
    if(!out)return MGC_INVALID_ARGUMENT;
    *out=nullptr;
    MgrPlan* unpublished=nullptr;
    const auto status=mgr_plan_create(model,desc,&unpublished);
    if(status!=MGC_SUCCESS)return status;
    unpublished->shared_tracking=true;
    *out=unpublished;
    return MGC_SUCCESS;
}
#endif
MgcStatus mgr_plan_describe(MgrPlan* plan, MgrPlanInfo* info) {
    if(!plan||!info)return MGC_INVALID_ARGUMENT;
    if(info->header.abi_major!=MGR_ABI_MAJOR||info->header.struct_size<offsetof(MgrPlanInfo,reserved))return MGC_UNSUPPORTED_ABI;
    if(partial_u32_array(info->header.struct_size,offsetof(MgrPlanInfo,reserved),9))return MGC_UNSUPPORTED_ABI;
    MgrPlanInfo value{};value.header=MGC_HEADER_INIT(MgrPlanInfo);
    value.effective_options_hash=plan->desc.effective_options_hash;value.runtime_graph_hash=plan->runtime_graph_hash;
    value.precision_profile=plan->desc.precision_profile;value.flow_scale=plan->desc.flow_scale;
    value.execution_flags=plan->desc.execution_flags;value.optimization_flags=plan->desc.optimization_flags;
    value.workload_qualified=MGR_QUALIFICATION_UNVERIFIED;
    uint32_t shaderCounts[3]{};plan->engine->shaderCounts(shaderCounts);
    value.offline_pipelines=shaderCounts[0];value.runtime_fallbacks=shaderCounts[1];value.variant_misses=shaderCounts[2];
    const auto pools=snapshot_output_pools(plan,&value.output_pool_capacity,&value.output_pool_allocated_slots);
    if(pools!=MGC_SUCCESS)return pools;
    const auto size=info->header.struct_size;value.header.struct_size=size;
    std::memcpy(info,&value,std::min<size_t>(size,sizeof(value)));return MGC_SUCCESS;
}
MgcStatus mgr_plan_shader_catalog(MgrPlan* plan,const char** text,uint64_t* size) {
    if(!plan||!text||!size)return MGC_INVALID_ARGUMENT;*text=nullptr;*size=0;
    const auto& catalog=plan->engine->shaderCatalog();if(catalog.empty())return MGC_UNSUPPORTED_CAPABILITY;
    *text=catalog.data();*size=catalog.size();return MGC_SUCCESS;
}
MgcStatus mgr_plan_execution_info(MgrPlan* plan,const char** text,uint64_t* size) {
    if(!plan||!text||!size)return MGC_INVALID_ARGUMENT;*text=nullptr;*size=0;
    const auto& info=plan->engine->executionInfo();if(info.empty())return MGC_UNSUPPORTED_CAPABILITY;
    *text=info.data();*size=info.size();return MGC_SUCCESS;
}
void mgr_plan_retain(MgrPlan* p) { if (p) p->refs.fetch_add(1); }
void mgr_plan_release(MgrPlan* p) { if (p && p->refs.fetch_sub(1) == 1) delete p; }
MgcStatus mgr_session_create(MgrPlan* plan, const MgcSessionDesc* d, MgrSession** out) {
    if (!out) return MGC_INVALID_ARGUMENT; *out = nullptr;
    if (!plan) return MGC_INVALID_ARGUMENT;
    MgcSessionDesc copied{};
    const auto abi = read_desc(d, offsetof(MgcSessionDesc, required_flags) + sizeof(uint32_t), copied);
    if (abi != MGC_SUCCESS) return abi;
    if (partial_u32_array(d->header.struct_size, offsetof(MgcSessionDesc, reserved), 4)) return MGC_UNSUPPORTED_ABI;
    d = &copied;
    try {
        auto session = std::make_unique<MgrSession>();
        const auto s = mgc_session_create(plan->model->domain, d, &session->compute);
        if (s != MGC_SUCCESS) return s;
        mgr_plan_retain(plan); session->plan = plan;
        *out = session.release(); return MGC_SUCCESS;
    } catch (...) { return MGC_OUT_OF_MEMORY; }
}
MgcStatus mgr_session_close(MgrSession* s) {
    if (!s) return MGC_INVALID_ARGUMENT;
    auto status = mgc_session_close(s->compute);
    if (status != MGC_SUCCESS) return status;
    mgr_plan_release(s->plan); delete s; return MGC_SUCCESS;
}
static MgcStatus mgr_submit_impl(MgrSession* s, const MgrRequest* d, MgrCompletionFn completion, void* context,
                     MgcContextFn retain_context, MgcContextFn release_context, MgcTicket** ticket, bool private_ahb) {
    if (!ticket) return MGC_INVALID_ARGUMENT; *ticket = nullptr;
    if (!s) return MGC_INVALID_ARGUMENT;
    MgrRequest copied{};
    auto abi = read_desc(d, offsetof(MgrRequest, required_flags) + sizeof(uint32_t), copied);
    if (abi != MGC_SUCCESS) return abi;
    const auto size = d->header.struct_size;
    if (partial_u32_array(size, offsetof(MgrRequest, reserved), 3) ||
        partial_field(size, offsetof(MgrRequest, flow_scale), sizeof(copied.flow_scale)) ||
        partial_field(size, offsetof(MgrRequest, reserved_tail), sizeof(copied.reserved_tail)) ||
        partial_field(size, offsetof(MgrRequest, output_storage_usage), sizeof(copied.output_storage_usage)) ||
        partial_field(size, offsetof(MgrRequest, output_sync_mode), sizeof(copied.output_sync_mode)) ||
        partial_field(size, offsetof(MgrRequest, batch_event), sizeof(copied.batch_event))) return MGC_UNSUPPORTED_ABI;
    d = &copied;
    MgrOutputSpec output{};
    abi = read_desc(&d->output, offsetof(MgrOutputSpec, row_stride_bytes) + sizeof(uint64_t), output);
    if (abi != MGC_SUCCESS) return abi;
    if (partial_u32_array(d->output.header.struct_size, offsetof(MgrOutputSpec, reserved), 4)) return MGC_UNSUPPORTED_ABI;
    copied.output = output;
    if (d->required_flags || d->output.required_flags) return MGC_UNSUPPORTED_ABI;
    if(private_ahb && (s->plan->desc.input_format!=MGC_FORMAT_RGB8 ||
        !(s->plan->desc.execution_flags&MGR_EXECUTION_DYNAMIC_SHAPE) ||
        (s->plan->desc.execution_flags&MGR_EXECUTION_GPU_IO_DISABLED) ||
        d->output_storage_usage!=MGC_RESOURCE_ANDROID_AHB || d->output.format!=MGC_FORMAT_RGBA8)) return MGC_UNSUPPORTED_CAPABILITY;
    if (!completion || !d->left || !d->right || mgr_detail::classify_alpha(d->alpha) == mgr_detail::Alpha::Invalid ||
        bool(retain_context) != bool(release_context) || d->required_flags || !zero(d->reserved) ||
        d->output.required_flags || !zero(d->output.reserved) ||
        (d->output.format != MGC_FORMAT_RGB8 && d->output.format != MGC_FORMAT_RGBA8) ||
        d->output_storage_usage > MGC_RESOURCE_ANDROID_AHB || d->output_sync_mode > MGC_ANDROID_GPU_SYNC_FD ||
        (d->output_storage_usage!=MGC_RESOURCE_ANDROID_AHB && d->output_sync_mode) ||
        (!d->output_storage_usage && (d->output.format != MGC_FORMAT_RGB8 || s->plan->desc.input_format != MGC_FORMAT_RGB8 || d->output_sync_mode)) ||
        (d->output_storage_usage && (s->plan->desc.precision_profile == MGR_PRECISION_FP32_REFERENCE ||
            s->plan->desc.precision_profile == MGR_PRECISION_CPU_FP16_LEGACY)) ||
        (d->output_storage_usage && (s->plan->desc.execution_flags & MGR_EXECUTION_GPU_IO_DISABLED)) ||
        (d->output_storage_usage == MGC_RESOURCE_ANDROID_AHB && d->output.format != MGC_FORMAT_RGBA8) ||
        !d->output.width || !d->output.height || d->output.width > 4096 || d->output.height > 4096 ||
        uint64_t(d->output.width)*d->output.height > (s->plan->desc.max_pixels ? s->plan->desc.max_pixels : uint64_t(s->plan->desc.width)*s->plan->desc.height) ||
        (!(s->plan->desc.execution_flags & MGR_EXECUTION_DYNAMIC_SHAPE) && (d->output.width != s->plan->desc.width || d->output.height != s->plan->desc.height)) ||
        d->reserved_tail ||
        (d->flow_scale && (!(s->plan->desc.execution_flags & MGR_EXECUTION_ADAPTIVE_FLOW) || (d->flow_scale != 4 && d->flow_scale != 6 && d->flow_scale != 8))) ||
        d->output.row_stride_bytes < uint64_t(d->output.width)*(d->output.format == MGC_FORMAT_RGBA8 ? 4 : 3) ||
        d->output.row_stride_bytes > (128u*1024u*1024u)/d->output.height) return MGC_INVALID_ARGUMENT;
    MgcLeaseInfo a{}, b{};
    auto status = read_info(d->left, a); if (status != MGC_SUCCESS) return status;
    status = read_info(d->right, b); if (status != MGC_SUCCESS) return status;
    if (a.domain_id != b.domain_id || a.domain_generation != b.domain_generation) return MGC_INCOMPATIBLE_DOMAIN;
    uint64_t domain_id = 0, generation = 0;
    status = mgc_domain_identity(s->plan->model->domain, &domain_id, &generation);
    if (status != MGC_SUCCESS) return status;
    if (a.domain_id != domain_id || a.domain_generation != generation) return MGC_INCOMPATIBLE_DOMAIN;
    const auto input_format=private_ahb?uint32_t(MGC_FORMAT_RGBA8):s->plan->desc.input_format;
    if (!image_fits(a, d->output, input_format,d->output_storage_usage!=0) ||
        !image_fits(b, d->output, input_format,d->output_storage_usage!=0)) return MGC_INVALID_ARGUMENT;
#if MGR_SHARED_PLAN_ENABLED
    if(private_ahb) {
        AHardwareBuffer *left_buffer=nullptr,*right_buffer=nullptr;
        status=mgc_android_get_buffer(d->left,&left_buffer);if(status)return status;
        status=mgc_android_get_buffer(d->right,&right_buffer);if(status)return status;
        if(!left_buffer||!right_buffer)return MGC_UNSUPPORTED_CAPABILITY;
    }
#endif
    try {
        auto owner=std::make_unique<Job>();auto* job=owner.get();
        job->plan = s->plan; mgr_plan_retain(job->plan);
#if MGR_SHARED_PLAN_ENABLED
        if(job->plan->shared_tracking) {
            std::lock_guard<std::mutex> lock(job->plan->shared_admission_mutex);
            if(private_ahb && !job->plan->shared_owners)return MGC_UNSUPPORTED_CAPABILITY;
            if(job->plan->shared_ahb && job->plan->shared_pending)return MGC_BACKPRESSURE;
            if(job->plan->shared_pending==UINT64_MAX)return MGC_BACKPRESSURE;
            ++job->plan->shared_pending;job->shared_tracked=true;
        } else if(private_ahb) return MGC_UNSUPPORTED_CAPABILITY;
#endif
        job->request = *d; job->callback = completion; job->context = context;
        job->retain_context = retain_context; job->release_context = release_context;
        const auto reservationBegan=std::chrono::steady_clock::now();
        status=reserve_output(*job->plan,job->request,&job->output_writing,private_ahb);
        if(status!=MGC_SUCCESS)return status;
        job->bridge_ns[0]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-reservationBegan).count();
        MgcLease* output_submission=job->output_writing;
        mgc_lease_retain(output_submission);
        MgcLease* leases[] = {d->left, d->right,output_submission};
        MgcHostJobDesc desc{}; desc.header = MGC_HEADER_INIT(MgcHostJobDesc);
        desc.request_id = d->request_id; desc.leases = leases; desc.lease_count = 3;
        desc.required_flags=MGC_HOST_JOB_EXPLICIT_GPU_SCOPES;
        desc.execute = execute_job; desc.completion = complete_job; desc.context = job;
        desc.retain_context = retain_job; desc.release_context = release_job;
        desc.resume=execute_job;
        desc.finalize=[](void* p){static_cast<Job*>(p)->gpu_work.reset();};
        desc.batch_event = d->batch_event?batch_job:nullptr;
        status = mgc_private_submit_host_job(s->compute, &desc, ticket);
        if(status!=MGC_ACCEPTED)mgc_lease_release(output_submission);
        owner.release();
        drop_caller_job(job); return status;
    } catch (const std::bad_alloc&) { return MGC_OUT_OF_MEMORY; }
    catch (...) { return MGC_INTERNAL_ERROR; }
}
MgcStatus mgr_submit(MgrSession* s,const MgrRequest* d,MgrCompletionFn completion,void* context,
                     MgcContextFn retain_context,MgcContextFn release_context,MgcTicket** ticket) {
    return mgr_submit_impl(s,d,completion,context,retain_context,release_context,ticket,false);
}
#if MGR_SHARED_PLAN_ENABLED
MgcStatus mgr_private_enable_shared_ahb(MgrPlan* p) {
    if(!p||!p->shared_tracking||p->desc.input_format!=MGC_FORMAT_RGB8||!(p->desc.execution_flags&MGR_EXECUTION_DYNAMIC_SHAPE)||
       (p->desc.execution_flags&MGR_EXECUTION_GPU_IO_DISABLED)||
       p->desc.precision_profile==MGR_PRECISION_FP32_REFERENCE||p->desc.precision_profile==MGR_PRECISION_CPU_FP16_LEGACY)
        return MGC_UNSUPPORTED_CAPABILITY;
    std::lock_guard<std::mutex> lock(p->shared_admission_mutex);
    if(!p->shared_ahb && p->shared_pending)return MGC_BUSY;
    if(p->shared_owners==UINT32_MAX)return MGC_BACKPRESSURE;
    if(!p->shared_ahb){
        try {p->shared_ahb_output_pool.reserve(p->desc.output_pool_capacity?p->desc.output_pool_capacity:4);}
        catch(const std::bad_alloc&){return MGC_OUT_OF_MEMORY;}
        catch(...){return MGC_INTERNAL_ERROR;}
    }
    ++p->shared_owners;p->shared_ahb=true;return MGC_SUCCESS;
}
void mgr_private_release_shared_ahb(MgrPlan* p) {
    if(!p)return;
    std::vector<std::unique_ptr<MgrPlan::OutputSlot>> retired_outputs;
    {
        std::lock_guard<std::mutex> lock(p->shared_admission_mutex);
        if(!p->shared_owners)return;
        --p->shared_owners;
        if(!p->shared_owners && !p->shared_pending){
            p->shared_ahb=false;std::lock_guard<std::mutex> outputs(p->output_mutex);
            retired_outputs.swap(p->shared_ahb_output_pool);
        }
    }
}
MgcStatus mgr_private_submit_shared_ahb(MgrSession* s,const MgrRequest* d,MgrCompletionFn completion,void* context,
                     MgcContextFn retain_context,MgcContextFn release_context,MgcTicket** ticket) {
    return mgr_submit_impl(s,d,completion,context,retain_context,release_context,ticket,true);
}
#endif
MgcStatus mgr_cancel(MgrSession* s, MgcTicket* t) { return s ? mgc_private_session_cancel(s->compute, t) : MGC_INVALID_ARGUMENT; }
MgcStatus mgr_drain(MgrSession* s, uint64_t ns) { return s ? mgc_session_drain(s->compute, ns) : MGC_INVALID_ARGUMENT; }
MgcStatus mgr_uploads_poll(MgcDomain* domain, uint64_t ns) {
    if(ns>100000000)return MGC_INVALID_ARGUMENT;
    return code(legacy::Engine::pollRetiredUploads(ns,domain));
}
MgcStatus mgr_result_describe(MgrResult* r, MgrResultInfo* info) {
    if (!r || !info) return MGC_INVALID_ARGUMENT;
    if (info->header.abi_major != MGC_ABI_MAJOR || info->header.struct_size < offsetof(MgrResultInfo, reserved)) return MGC_UNSUPPORTED_ABI;
    const auto size = info->header.struct_size;
    if (partial_u32_array(size,offsetof(MgrResultInfo,reserved),7) ||
        partial_u64_array(size,offsetof(MgrResultInfo,stage_ns),3) ||
        partial_field(size,offsetof(MgrResultInfo,worker_cpu_ns),sizeof(info->worker_cpu_ns)) ||
        partial_u32_array(size,offsetof(MgrResultInfo,worker_cpu_observed),2) ||
        partial_field(size,offsetof(MgrResultInfo,gpu_elapsed_ns),sizeof(info->gpu_elapsed_ns)) ||
        partial_field(size,offsetof(MgrResultInfo,gpu_batch_count),sizeof(info->gpu_batch_count)) ||
        partial_field(size,offsetof(MgrResultInfo,gpu_observed_batch_count),sizeof(info->gpu_observed_batch_count)) ||
        partial_u32_array(size,offsetof(MgrResultInfo,gpu_elapsed_observed),4)) return MGC_UNSUPPORTED_ABI;
    auto value = r->info; value.header.struct_size = size;
    std::memcpy(info, &value, std::min<size_t>(size, sizeof(value))); return MGC_SUCCESS;
}
MgcStatus mgr_result_observation(MgrResult* result, MgrJobObservation* out) {
    if (!result || !out) return MGC_INVALID_ARGUMENT;
    const uint32_t size=out->header.struct_size;
    constexpr size_t required=offsetof(MgrJobObservation,reserved);
    if(out->header.abi_major!=MGR_ABI_MAJOR || size<required ||
       (size>required && size<sizeof(MgrJobObservation) && (size-required)%sizeof(uint32_t)))
        return MGC_UNSUPPORTED_ABI;
#if defined(MGR_ENABLE_JOB_OBSERVATION)
    const auto& value=result->observation;
    if(!value.started || !value.completed) return MGC_INTERNAL_ERROR;
    MgrJobObservation observed{};observed.header=MGC_HEADER_INIT(MgrJobObservation);
    observed.header.struct_size=size;observed.request_id=result->info.request_id;
    observed.frame_kind=result->info.frame_kind;observed.effective_flow_scale=uint32_t(value.effectiveScale);
    observed.cache_lookup_scale=uint32_t(value.cacheLookupScale);
    observed.static_guard_dispatch=uint32_t(value.staticGuard);observed.motion_guard_dispatch=uint32_t(value.motionGuard);
    observed.tensor_cache_left=uint32_t(value.tensorCache[0]);observed.tensor_cache_right=uint32_t(value.tensorCache[1]);
    observed.bridge_cache_left=uint32_t(value.bridgeCache[0]);observed.bridge_cache_right=uint32_t(value.bridgeCache[1]);
    observed.tensor_cache_publish_stage_completed=value.cachePublishObserved?1u:0u;
    observed.tensor_cache_published_count=value.cachePublishedCount;
    std::memcpy(out,&observed,std::min<size_t>(size,sizeof(observed)));return MGC_SUCCESS;
#else
    (void)size;
    return MGC_UNSUPPORTED_CAPABILITY;
#endif
}
void mgr_result_release(MgrResult* r) { delete r; }
}
