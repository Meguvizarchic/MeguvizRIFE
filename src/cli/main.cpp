#include <meguviz/rife.h>
#include <meguviz/backend_ncnn_sdk.hpp>
#include "private/data.hpp"
#include "private/catalog.hpp"
#include "build_identity.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <fstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace mgr_detail;
namespace fs = std::filesystem;
namespace {
fs::path output_path(const std::string& value) {
    require(!value.empty()&&value.find('\0')==std::string::npos,4,"invalid_output_path");
    return fs::u8path(value);
}
bool exists_no_follow(const fs::path& path) {
    std::error_code ec;const auto status=fs::symlink_status(path,ec);
    require(!ec||ec==std::errc::no_such_file_or_directory,9,"output_path_status_failed");
    return fs::exists(status)||fs::is_symlink(status);
}
bool path_component_equal(const fs::path& a,const fs::path& b) {
#ifdef _WIN32
    const auto x=a.native(),y=b.native();
    return CompareStringOrdinal(x.c_str(),int(x.size()),y.c_str(),int(y.size()),TRUE)==CSTR_EQUAL;
#else
    return a==b;
#endif
}
fs::path normalized(const fs::path& path) {
    std::error_code ec;const auto absolute=fs::absolute(path,ec);require(!ec,9,"output_path_resolution_failed");
    const auto result=fs::weakly_canonical(absolute,ec);require(!ec,9,"output_path_resolution_failed");return result.lexically_normal();
}
bool contains_path(const fs::path& parent,const fs::path& candidate) {
    const auto root=normalized(parent),target=normalized(candidate);auto a=root.begin(),b=target.begin();
    for(;a!=root.end();++a,++b)if(b==target.end()||!path_component_equal(*a,*b))return false;
    return true;
}
bool same_path(const fs::path& a,const fs::path& b) {
    std::error_code ec;if(fs::equivalent(a,b,ec)&&!ec)return true;
    return contains_path(a,b)&&contains_path(b,a);
}
void check(MgcStatus s, const char* phase) { require(s == MGC_SUCCESS, 9, std::string(phase) + ":" + std::to_string(s)); }
MgcHash digest(const std::string& s) {
    require(valid_hash(s), 4, "invalid_sha256"); MgcHash h{};
    for (size_t i=0;i<32;i++) h.bytes[i] = uint8_t(std::stoul(s.substr(i*2,2), nullptr,16));
    return h;
}
std::string hex_digest(const MgcHash& hash) {
    static constexpr char digits[]="0123456789abcdef";std::string out;
    for(auto byte:hash.bytes){out+=digits[byte>>4];out+=digits[byte&15];}return out;
}
struct Image { uint32_t w=0,h=0; std::string pixels; };
Image ppm(const fs::path& path) {
    const auto bytes = read(path,128*1024*1024); size_t at=0;
    auto token = [&]() {
        for (;;) {
            while (at<bytes.size() && (bytes[at]==' '||bytes[at]=='\t'||bytes[at]=='\r'||bytes[at]=='\n')) at++;
            if (at<bytes.size() && bytes[at]=='#') { while(at<bytes.size()&&bytes[at]!='\n') at++; }
            else break;
        }
        const auto begin=at;
        while(at<bytes.size() && bytes[at]!=' '&&bytes[at]!='\t'&&bytes[at]!='\r'&&bytes[at]!='\n') at++;
        require(at>begin&&at-begin<32,4,"ppm_token"); return bytes.substr(begin,at-begin);
    };
    require(token()=="P6",4,"ppm_requires_p6");
    const auto sw=token(), sh=token();
    require(!sw.empty()&&!sh.empty()&&std::all_of(sw.begin(),sw.end(),[](char c){return c>='0'&&c<='9';})&&
        std::all_of(sh.begin(),sh.end(),[](char c){return c>='0'&&c<='9';}),4,"ppm_dimensions");
    const auto w=std::stoul(sw),h=std::stoul(sh);
    require(w>0&&h>0&&w<=4096&&h<=4096&&uint64_t(w)*h<=4096u*2160u,4,"ppm_dimensions");
    require(token()=="255"&&at<bytes.size(),4,"ppm_rgb8_required");
    if(bytes[at]=='\r'&&at+1<bytes.size()&&bytes[at+1]=='\n') at+=2; else at++;
    require(bytes.size()-at==uint64_t(w)*h*3,4,"ppm_payload_size");
    return {uint32_t(w),uint32_t(h),bytes.substr(at)};
}
std::string encode(const Image& image) { return "P6\n"+std::to_string(image.w)+" "+std::to_string(image.h)+"\n255\n"+image.pixels; }
struct Runtime {
    MgcRuntime* runtime=nullptr; MgcDomain* domain=nullptr; MgrModel* model=nullptr; MgrPlan* plan=nullptr;
    MgrSession* session=nullptr; MgcLease *left=nullptr,*right=nullptr; MgcTicket* ticket=nullptr; MgrResult* result=nullptr;
    MgcShaderpack* shaderpack=nullptr;
    ~Runtime() {
        if(session) { mgr_drain(session,UINT64_MAX); mgr_session_close(session); }
        mgr_result_release(result); mgc_ticket_release(ticket); mgc_lease_release(left);mgc_lease_release(right);
        mgr_plan_release(plan);mgc_shaderpack_release(shaderpack);mgr_model_release(model);mgc_domain_release(domain);
        if(runtime) { mgc_runtime_close(runtime);mgc_runtime_release(runtime); }
    }
};
MgcLease* upload(MgcDomain* domain,const Image& image) {
    MgcResourceDesc r{};r.header=MGC_HEADER_INIT(MgcResourceDesc);r.capacity_bytes=image.pixels.size();
    r.width=image.w;r.height=image.h;r.format=MGC_FORMAT_RGB8;r.row_stride_bytes=uint64_t(image.w)*3;
    MgcResource* resource=nullptr;check(mgc_resource_create(domain,&r,&resource),"resource");
    MgcLeaseDesc d{};d.header=MGC_HEADER_INIT(MgcLeaseDesc);d.byte_length=r.capacity_bytes;d.access=MGC_ACCESS_WRITE;d.content_id=1;
    MgcLease* lease=nullptr;auto s=mgc_lease_create(resource,&d,&lease);
    if(s!=MGC_SUCCESS){mgc_resource_release(resource);check(s,"write_lease");}
    void* ptr=nullptr;uint64_t size=0;s=mgc_lease_map(lease,&ptr,&size);
    if(s==MGC_SUCCESS){std::memcpy(ptr,image.pixels.data(),image.pixels.size());s=mgc_lease_unmap(lease);}
    mgc_lease_release(lease);lease=nullptr;
    if(s==MGC_SUCCESS){d.access=MGC_ACCESS_READ;s=mgc_lease_create(resource,&d,&lease);}
    mgc_resource_release(resource);check(s,"input_upload");return lease;
}
struct Completion { MgcStatus status=MGC_INTERNAL_ERROR; MgrResult* result=nullptr; unsigned count=0; };
void completed(void* c,uint64_t,MgcStatus status,MgrResult* result) {
    auto& value=*static_cast<Completion*>(c);value.status=status;value.result=result;value.count++;
}
J interpolate(const J& spec,bool catalogOnly=false,bool benchmark=false) {
    keys(spec,{"schema_version","model_directory","left","right","alpha","output","precision_profile",
        "flow_scale","guard_flags","execution_flags","optimization_flags","num_threads",
        "shader_request_output_directory","shader_toolchain_lock_hash","shaderpack_directory",
        "shaderpack_manifest_sha256","shader_bindings_file","warmup_seconds","run_seconds","raw_events"});
    require(spec.number("schema_version")==1,4,"run_schema");
    const auto profile=spec.text("precision_profile");
    const bool reference=profile=="CPU_FP32_REFERENCE";
    const bool cpuFp16=profile=="CPU_FP16_LEGACY",cpu=reference||cpuFp16;
    require(cpu||profile=="VULKAN_LEGACY_FP16"||profile=="VULKAN_LEGACY_FP32",3,"unsupported_precision_profile");
    size_t consumed=0;const auto alpha=std::stod(spec.at("alpha").str(),&consumed);
    require(consumed==spec.at("alpha").str().size()&&std::isfinite(alpha)&&alpha>=0&&alpha<=1,4,"alpha_range");
    if(benchmark) {
        require(alpha>0&&alpha<1,4,"benchmark_requires_inferred_alpha");
        require(spec.number("warmup_seconds")==30&&spec.number("run_seconds")==120,4,"benchmark_fixed_protocol_30s_120s");
        require(!spec.text("raw_events").empty(),4,"benchmark_raw_events_required");
    } else require(!spec.has("warmup_seconds")&&!spec.has("run_seconds")&&!spec.has("raw_events"),4,"benchmark_fields_only_for_benchmark");
    const auto a=ppm(fs::u8path(spec.at("left").str())),b=ppm(fs::u8path(spec.at("right").str()));
    require(a.w==b.w&&a.h==b.h,4,"shape_mismatch");
    const auto manifest=Parser(meguviz::rife::catalog_manifest).parse();
    const auto modelDir=fs::u8path(spec.at("model_directory").str());
    const auto graph=read(modelDir/"flownet.param",1024*1024),weights=read(modelDir/"flownet.bin",32*1024*1024);
    // The callback context must outlive Runtime's drain even on an error path.
    Completion completion;Runtime r;MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=cpu?MGC_PROVIDER_NCNN_CPU:MGC_PROVIDER_NCNN_VULKAN;
    rd.memory_budget_bytes=256u*1024u*1024u;check(mgc_runtime_create(&rd,&r.runtime),"runtime");
    MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);dd.private_sdk_fingerprint=digest(meguviz::private_sdk_fingerprint());
    check(mgc_domain_acquire(r.runtime,&dd,&r.domain),"domain");
    MgrModelDesc md{};md.header=MGC_HEADER_INIT(MgrModelDesc);
    md.manifest_utf8=reinterpret_cast<const uint8_t*>(meguviz::rife::catalog_manifest);md.manifest_size=std::strlen(meguviz::rife::catalog_manifest);
    md.graph_data=reinterpret_cast<const uint8_t*>(graph.data());md.graph_size=graph.size();
    md.weights_data=reinterpret_cast<const uint8_t*>(weights.data());md.weights_size=weights.size();
    md.graph_sha256=digest(manifest.at("graph").text("sha256"));md.weights_sha256=digest(manifest.at("weights").text("sha256"));
    md.private_sdk_fingerprint=dd.private_sdk_fingerprint;check(mgr_model_create(r.domain,&md,&r.model),"model");
    MgrPlanDesc pd{};pd.header=MGC_HEADER_INIT(MgrPlanDesc);pd.width=a.w;pd.height=a.h;pd.input_format=MGC_FORMAT_RGB8;
    pd.precision_profile=reference?MGR_PRECISION_FP32_REFERENCE:cpuFp16?MGR_PRECISION_CPU_FP16_LEGACY:
        profile=="VULKAN_LEGACY_FP16"?MGR_PRECISION_LEGACY_FP16:MGR_PRECISION_VULKAN_FP32_LEGACY;
    for(const auto* key:{"flow_scale","guard_flags","execution_flags","optimization_flags","num_threads"})
        require(spec.number(key)>=0&&uint64_t(spec.number(key))<=UINT32_MAX,4,"profile_integer_range");
    pd.flow_scale=uint32_t(spec.number("flow_scale",cpu?1:4));
    pd.guard_flags=uint32_t(spec.number("guard_flags",cpu?0:3));
    pd.execution_flags=uint32_t(spec.number("execution_flags"));pd.optimization_flags=uint32_t(spec.number("optimization_flags"));
    pd.num_threads=uint32_t(spec.number("num_threads",1));
    if(pd.execution_flags&MGR_EXECUTION_DYNAMIC_SHAPE)pd.max_pixels=uint64_t(a.w)*a.h;
    std::vector<MgrShaderBinding> bindings;
    if(spec.has("shaderpack_directory")) {
        require(!cpu,3,"shaderpack_requires_vulkan_plan");
        const auto directory=spec.text("shaderpack_directory");
        MgcShaderpackDesc pack{};pack.header=MGC_HEADER_INIT(MgcShaderpackDesc);
        pack.root_path_utf8=directory.data();pack.root_path_length=uint32_t(directory.size());pack.profile=MGC_SHADER_PROFILE_NCNN_COMPAT;
        pack.manifest_hash=digest(spec.text("shaderpack_manifest_sha256"));
        check(mgc_shaderpack_open(r.domain,&pack,&r.shaderpack),"shaderpack_open");
        const auto config=load(fs::u8path(spec.text("shader_bindings_file")));
        keys(config,{"schema_version","private_sdk_fingerprint","bindings"});
        require(config.number("schema_version")==1&&config.text("private_sdk_fingerprint")==meguviz::private_sdk_fingerprint(),4,"shader_binding_sdk");
        require(config.at("bindings").arr().size()<=9,4,"shader_binding_count");
        for(const auto& item:config.at("bindings").arr()) {
            keys(item,{"kernel","variant_key","effective_options_hash","target_facts_hash","source_sha256","spirv_sha256"});
            require(item.number("kernel")>=1&&item.number("kernel")<=9,4,"shader_kernel");
            MgrShaderBinding binding{};binding.header=MGC_HEADER_INIT(MgrShaderBinding);binding.kernel=uint32_t(item.number("kernel"));
            binding.variant_key=digest(item.text("variant_key"));binding.effective_options_hash=digest(item.text("effective_options_hash"));
            binding.target_facts_hash=digest(item.text("target_facts_hash"));binding.source_sha256=digest(item.text("source_sha256"));
            binding.spirv_sha256=digest(item.text("spirv_sha256"));binding.private_sdk_fingerprint=dd.private_sdk_fingerprint;
            bindings.push_back(binding);
        }
        pd.shaderpack=r.shaderpack;pd.shader_bindings=bindings.data();pd.shader_binding_count=uint32_t(bindings.size());
    }
    check(mgr_plan_create(r.model,&pd,&r.plan),"plan");
    MgrPlanInfo planInfo{};planInfo.header=MGC_HEADER_INIT(MgrPlanInfo);check(mgr_plan_describe(r.plan,&planInfo),"plan_describe");
    if(catalogOnly) {
        require(!cpu,3,"captured_shader_requests_require_gpu_domain");
        const auto lockHash=spec.text("shader_toolchain_lock_hash");require(valid_hash(lockHash),4,"shader_toolchain_lock_hash_required");
        const char* text=nullptr;uint64_t size=0;check(mgr_plan_shader_catalog(r.plan,&text,&size),"shader_catalog");
        auto catalog=Parser(std::string(text,size_t(size))).parse();
        require(!catalog.at("variants").arr().empty(),3,"no_owner_shader_requests_for_plan");
        const auto destination=fs::u8path(spec.text("shader_request_output_directory"));
        require(!destination.empty()&&(!fs::exists(destination)||(fs::is_directory(destination)&&fs::is_empty(destination))),9,"shader_request_generation_exists");
        fs::create_directories(destination);
        J::A exportedEntries;
        for(auto entry:catalog.at("variants").arr()) {
            auto request=entry.at("compile_request");request["toolchain_lock_hash"]=lockHash;
            const auto directory=destination/("kernel-"+std::to_string(entry.number("kernel")));
            fs::create_directories(directory);write(directory/"shader.comp",entry.text("source_utf8"));write(directory/"request.json",dump(request)+"\n");
            entry["compile_request"]=request;
            exportedEntries.push_back(std::move(entry));
        }
        catalog["variants"]=exportedEntries;
        write(destination/"catalog.json",dump(catalog)+"\n");
        return J::O{{"result","PASS"},{"scope","captured compile requests only; no GPU numerical qualification"},
            {"request_count",int64_t(catalog.at("variants").arr().size())},{"output_directory",destination.u8string()},
            {"offline_pipelines",planInfo.offline_pipelines},{"runtime_fallbacks",planInfo.runtime_fallbacks},{"variant_misses",planInfo.variant_misses}};
    }
    MgcSessionDesc sd{};sd.header=MGC_HEADER_INIT(MgcSessionDesc);sd.max_in_flight=1;check(mgr_session_create(r.plan,&sd,&r.session),"session");
    r.left=upload(r.domain,a);r.right=upload(r.domain,b);
    if(benchmark) {
        const auto rawPath=fs::u8path(spec.text("raw_events"));
        ExclusiveFile stream(rawPath);
        auto timestamp=[]()->int64_t{return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());};
        auto emit=[&](const J& event){stream.append(dump(event)+"\n");};
        uint64_t sequence=0;int64_t warmCount=0,measureCount=0;
        for(const auto& phase:{std::pair<const char*,int>{"warmup",30},{"measure",120}}) {
            const auto started=std::chrono::steady_clock::now();const auto until=started+std::chrono::seconds(phase.second);
            emit(J::O{{"kind","run_start"},{"phase",phase.first},{"timestamp_ns",timestamp()},{"time_domain","CPU_MONOTONIC"}});
            while(std::chrono::steady_clock::now()<until) {
                require(++sequence<=1000000,9,"benchmark_sample_limit");completion=Completion{};
                MgrRequest request{};request.header=MGC_HEADER_INIT(MgrRequest);request.request_id=sequence;request.alpha=alpha;
                request.left=r.left;request.right=r.right;mgc_lease_retain(r.left);mgc_lease_retain(r.right);
                request.output.header=MGC_HEADER_INIT(MgrOutputSpec);request.output.width=a.w;request.output.height=a.h;
                request.output.format=MGC_FORMAT_RGB8;request.output.row_stride_bytes=uint64_t(a.w)*3;
                const auto begin=timestamp();auto accepted=mgr_submit(r.session,&request,completed,&completion,nullptr,nullptr,&r.ticket);
                if(accepted!=MGC_ACCEPTED) {
                    mgc_lease_release(r.left);mgc_lease_release(r.right);
                    emit(J::O{{"kind","job"},{"phase",phase.first},{"request_id",int64_t(sequence)},
                        {"status",int64_t(accepted)},{"accepted",false},{"timestamp_ns",timestamp()}});
                    throw Error(9,"benchmark_submit_failed:"+std::to_string(accepted));
                }
                auto drained=mgr_drain(r.session,UINT64_MAX);const auto end=timestamp();mgc_ticket_release(r.ticket);r.ticket=nullptr;
                if(drained!=MGC_SUCCESS||completion.status!=MGC_SUCCESS||completion.count!=1||!completion.result) {
                    mgr_result_release(r.result);r.result=completion.result;
                    emit(J::O{{"kind","job"},{"phase",phase.first},{"request_id",int64_t(sequence)},
                        {"status",int64_t(completion.status)},{"drain_status",int64_t(drained)},
                        {"accepted",true},{"timestamp_ns",end}});
                    throw Error(9,"benchmark_completion_failed");
                }
                // Keep at most one completed output pinned for a phase-end hash.
                mgr_result_release(r.result);r.result=completion.result;
                MgrResultInfo info{};info.header=MGC_HEADER_INIT(MgrResultInfo);check(mgr_result_describe(r.result,&info),"benchmark_result");
                emit(J::O{{"kind","job"},{"phase",phase.first},{"request_id",int64_t(sequence)},
                    {"accepted",true},{"status",int64_t(completion.status)},{"algorithm_accepted",bool(info.algorithm_accepted)},
                    {"frame_kind",int64_t(info.frame_kind)},{"submit_to_drain_ns",int64_t(end-begin)},
                    {"timestamp_ns",end},{"worker_cpu_ns",int64_t(info.worker_cpu_ns)},
                    {"worker_cpu_observed",bool(info.worker_cpu_observed)},{"gpu_elapsed_ns",int64_t(info.gpu_elapsed_ns)},
                    {"gpu_elapsed_observed",bool(info.gpu_elapsed_observed)}});
                require(info.algorithm_accepted&&info.output&&info.frame_kind==MGR_RESULT_INFERRED,9,"benchmark_non_inferred_result");
                if(std::strcmp(phase.first,"warmup")==0)++warmCount;else ++measureCount;
            }
            require(r.result!=nullptr,9,"benchmark_no_successful_jobs");
            MgrResultInfo last{};last.header=MGC_HEADER_INIT(MgrResultInfo);check(mgr_result_describe(r.result,&last),"benchmark_last_result");
            void* pixels=nullptr;uint64_t size=0;check(mgc_lease_map(last.output,&pixels,&size),"benchmark_output_map");
            const auto outputHash=sha256(std::string(static_cast<const char*>(pixels),size_t(size)));
            check(mgc_lease_unmap(last.output),"benchmark_output_unmap");
            emit(J::O{{"kind","run_end"},{"phase",phase.first},{"timestamp_ns",timestamp()},
                {"time_domain","CPU_MONOTONIC"},{"last_output_sha256",outputHash}});
            mgr_result_release(r.result);r.result=nullptr;
        }
        require(warmCount>0&&measureCount>0,9,"benchmark_no_inferred_jobs");
        stream.close();
        return J::O{{"result","OBSERVED"},{"scope","SINGLE_PERSISTENT_SESSION_NOT_PAIRED_PERFORMANCE_QUALIFICATION"},
            {"measurement_boundary","CPU_MONOTONIC submit through drain, includes wrapper and queue; output hash outside interval"},
            {"warmup_seconds",30},{"run_seconds",120},{"warmup_successful_inferred_jobs",warmCount},{"successful_inferred_jobs",measureCount},
            {"raw_events",rawPath.u8string()},{"raw_events_sha256",sha256(read(rawPath,256*1024*1024))},
            {"input_left_sha256",sha256(a.pixels)},{"input_right_sha256",sha256(b.pixels)},
            {"model_graph_sha256",sha256(graph)},{"model_weights_sha256",sha256(weights)},
            {"effective_options_hash",hex_digest(planInfo.effective_options_hash)},{"runtime_graph_sha256",hex_digest(planInfo.runtime_graph_hash)},
            {"private_sdk_fingerprint",meguviz::private_sdk_fingerprint()},{"build_fingerprint",meguviz::rife::build_fingerprint},
            {"width",int64_t(a.w)},{"height",int64_t(a.h)},{"alpha",spec.at("alpha")},{"precision_profile",profile},
            {"requested_guard_flags",int64_t(pd.guard_flags)},{"requested_execution_flags",int64_t(pd.execution_flags)},
            {"requested_optimization_flags",int64_t(pd.optimization_flags)},{"requested_num_threads",int64_t(pd.num_threads)},
            {"quality_qualification","INCONCLUSIVE"},{"performance_qualification","INCONCLUSIVE"}};
    }
    MgrRequest request{};request.header=MGC_HEADER_INIT(MgrRequest);request.request_id=1;request.alpha=alpha;
    request.left=r.left;request.right=r.right;mgc_lease_retain(r.left);mgc_lease_retain(r.right);
    request.output.header=MGC_HEADER_INIT(MgrOutputSpec);request.output.width=a.w;request.output.height=a.h;
    request.output.format=MGC_FORMAT_RGB8;request.output.row_stride_bytes=uint64_t(a.w)*3;
    const auto began=std::chrono::steady_clock::now();
    const auto accepted=mgr_submit(r.session,&request,completed,&completion,nullptr,nullptr,&r.ticket);
    if(accepted!=MGC_ACCEPTED){mgc_lease_release(request.left);mgc_lease_release(request.right);check(accepted,"submit");}
    check(mgr_drain(r.session,UINT64_MAX),"drain");
    const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-began).count();
    r.result=completion.result;check(completion.status,"inference");require(completion.count==1&&r.result,9,"completion_or_output_missing");
    MgrResultInfo info{};info.header=MGC_HEADER_INIT(MgrResultInfo);check(mgr_result_describe(r.result,&info),"result");
    check(mgr_session_close(r.session),"close_session_with_owned_output");r.session=nullptr;
    mgr_plan_release(r.plan);r.plan=nullptr;mgr_model_release(r.model);r.model=nullptr;
    mgc_ticket_release(r.ticket);r.ticket=nullptr;
    mgc_lease_release(r.left);r.left=nullptr;mgc_lease_release(r.right);r.right=nullptr;
    mgc_domain_release(r.domain);r.domain=nullptr;
    require(mgc_runtime_close(r.runtime)==MGC_BUSY,9,"output_must_pin_domain_after_session_close");
    void* pixels=nullptr;uint64_t size=0;check(mgc_lease_map(info.output,&pixels,&size),"output_map");
    Image output{a.w,a.h,std::string(static_cast<const char*>(pixels),size_t(size))};check(mgc_lease_unmap(info.output),"output_unmap");
    const auto encoded=encode(output);write(fs::u8path(spec.at("output").str()),encoded);
    return J::O{{"result","PASS"},{"scope",reference?"CPU_FP32_REFERENCE_REAL_MODEL_ONLY":cpuFp16?"CPU_FP16_LEGACY_EXECUTION_ONLY_UNQUALIFIED":"VULKAN_EXECUTION_ONLY_UNQUALIFIED"},{"algorithm_accepted",true},
        {"frame_kind",int64_t(info.frame_kind)},{"input_left_sha256",sha256(a.pixels)},{"input_right_sha256",sha256(b.pixels)},
        {"output_sha256",sha256(encoded)},{"pixel_sha256",sha256(output.pixels)},{"output_bytes",int64_t(encoded.size())},
        {"width",int64_t(a.w)},{"height",int64_t(a.h)},{"alpha",spec.at("alpha")},{"service_wall_ns",int64_t(elapsed)},
        {"private_sdk_fingerprint",meguviz::private_sdk_fingerprint()},{"model_graph_sha256",sha256(graph)},
        {"build_fingerprint",meguviz::rife::build_fingerprint},
        {"effective_options_hash",hex_digest(planInfo.effective_options_hash)},
        {"runtime_graph_sha256",hex_digest(planInfo.runtime_graph_hash)},
        {"offline_pipelines",planInfo.offline_pipelines},{"runtime_fallbacks",planInfo.runtime_fallbacks},{"variant_misses",planInfo.variant_misses},
        {"precision_profile",profile},{"flow_scale",int64_t(info.effective_flow_scale)},
        {"requested_guard_flags",int64_t(pd.guard_flags)},{"requested_execution_flags",int64_t(pd.execution_flags)},
        {"requested_optimization_flags",int64_t(pd.optimization_flags)},{"requested_num_threads",int64_t(pd.num_threads)},
        {"unreliable_permille",int64_t(info.unreliable_permille)},{"patched_tiles",int64_t(info.patched_tiles)},
        {"histogram_permille",int64_t(info.histogram_permille)},
        {"stage_ns",J::A{J(info.stage_ns[0]),J(info.stage_ns[1]),J(info.stage_ns[2])}},
        {"worker_cpu_ns",int64_t(info.worker_cpu_ns)},{"worker_cpu_observed",bool(info.worker_cpu_observed)},
        {"gpu_elapsed_ns",int64_t(info.gpu_elapsed_ns)},{"gpu_elapsed_observed",bool(info.gpu_elapsed_observed)},
        {"gpu_batch_count",int64_t(info.gpu_batch_count)},{"gpu_observed_batch_count",int64_t(info.gpu_observed_batch_count)},
        {"gpu_timestamp_valid_bits",info.gpu_timestamp_valid_bits},{"gpu_timestamp_period_bits",info.gpu_timestamp_period_bits},
        {"batch_event_coverage_observed",bool(info.batch_event_coverage_observed)},
        {"output_survived_session_close",true},{"runtime_close_with_owned_output","BUSY"},
        {"model_weights_sha256",sha256(weights)},{"performance_qualification","INCONCLUSIVE"},
        {"vulkan_legacy_equivalence","INCONCLUSIVE"},{"quality_qualification","INCONCLUSIVE"},
        {"guards",cpu?"CPU full-graph finite; effective precision follows actual loaded graph Options":"motion/static-detail/tile/nonfinite; GPU summaries + final GPU patch"}};
}
J compare(const J& spec) {
    keys(spec,{"schema_version","baseline_report","baseline_output","candidate_report","candidate_output"});
    require(spec.number("schema_version")==1,4,"compare_schema");
    const auto a=load(fs::u8path(spec.at("baseline_report").str()));
    const auto b=load(fs::u8path(spec.at("candidate_report").str()));
    for(const auto* report:{&a,&b}) {
        require(report->text("result")=="PASS" && report->flag("algorithm_accepted"),4,"compare_run_not_successful");
        require(report->number("frame_kind")==1||report->number("frame_kind")==2,4,"compare_frame_kind");
        require(report->has("alpha")&&!report->at("alpha").str().empty()&&
                !report->text("precision_profile").empty()&&!report->text("guards").empty(),4,"compare_missing_profile");
        size_t alphaEnd=0;double alphaValue=0;
        try { alphaValue=std::stod(report->at("alpha").str(),&alphaEnd); }
        catch(const std::exception&) { throw Error(4,"compare_invalid_alpha"); }
        require(alphaEnd==report->at("alpha").str().size()&&std::isfinite(alphaValue)&&alphaValue>=0&&alphaValue<=1,4,"compare_invalid_alpha");
        require((alphaValue==0||alphaValue==1)==(report->number("frame_kind")==2),4,"compare_alpha_frame_kind_mismatch");
        require(report->has("flow_scale")&&report->number("flow_scale")>=0&&
                (report->number("frame_kind")!=1||report->number("flow_scale")>0),4,"compare_missing_flow_scale");
        for(const auto* key:{"unreliable_permille","patched_tiles","histogram_permille"})
            require(report->has(key)&&report->number(key)>=0,4,std::string("compare_missing_")+key);
        for(const auto* key:{"input_left_sha256","input_right_sha256","model_graph_sha256","model_weights_sha256",
                             "effective_options_hash","runtime_graph_sha256","output_sha256","pixel_sha256"})
            require(valid_hash(report->text(key)),4,std::string("compare_invalid_")+key);
        require(report->has("requested_guard_flags")&&report->has("requested_execution_flags")&&
                report->has("requested_optimization_flags")&&report->has("requested_num_threads"),4,"compare_missing_requested_flags");
    }
    for(const auto* key:{"input_left_sha256","input_right_sha256","model_graph_sha256","model_weights_sha256",
                         "effective_options_hash","precision_profile","guards","alpha"})
        require(a.text(key)==b.text(key),4,std::string("compare_identity_mismatch_")+key);
    for(const auto* key:{"width","height","frame_kind","flow_scale","requested_guard_flags",
                         "requested_execution_flags","requested_num_threads"})
        require(a.number(key)==b.number(key),4,std::string("compare_identity_mismatch_")+key);
    const auto left=ppm(fs::u8path(spec.at("baseline_output").str()));
    const auto right=ppm(fs::u8path(spec.at("candidate_output").str()));
    require(left.w==right.w&&left.h==right.h&&left.w==uint64_t(a.number("width"))&&left.h==uint64_t(a.number("height")),4,"compare_output_shape");
    const auto leftBytes=read(fs::u8path(spec.at("baseline_output").str()),128*1024*1024);
    const auto rightBytes=read(fs::u8path(spec.at("candidate_output").str()),128*1024*1024);
    require(sha256(leftBytes)==a.text("output_sha256")&&sha256(rightBytes)==b.text("output_sha256")&&
            sha256(left.pixels)==a.text("pixel_sha256")&&sha256(right.pixels)==b.text("pixel_sha256"),4,"compare_output_hash_mismatch");
    uint64_t different=0,sum=0,squared=0;uint32_t maximum=0;
    for(size_t i=0;i<left.pixels.size();++i) {
        const auto x=uint8_t(left.pixels[i]),y=uint8_t(right.pixels[i]);const uint32_t delta=x>y?x-y:y-x;
        different+=delta!=0;sum+=delta;squared+=uint64_t(delta)*delta;maximum=std::max(maximum,delta);
    }
    J::O guards;bool sameGuards=true;
    for(const auto* key:{"unreliable_permille","patched_tiles","histogram_permille"}) {
        const bool same=a.number(key)==b.number(key);guards[key]=same;sameGuards&=same;
    }
    return J::O{{"result","OBSERVED"},{"scope","TWO_RECORDED_CLI_OUTPUTS_ONLY"},{"quality_qualification","INCONCLUSIVE"},
        {"performance_qualification","INCONCLUSIVE"},{"actual_guard_configuration_observed",false},
        {"note","Requested guard flags match; active guard state, cache state and device identity are not verified by this CLI comparison."},
        {"frame_kind",a.number("frame_kind")},{"sample_count",int64_t(left.pixels.size())},
        {"different_samples",int64_t(different)},{"max_abs_error",int64_t(maximum)},
        {"sum_abs_error",int64_t(sum)},{"sum_squared_error",int64_t(squared)},
        {"pixel_bytes_equal",different==0},{"guard_values_equal",sameGuards},{"guard_checks",guards},
        {"baseline_requested_optimization_flags",a.number("requested_optimization_flags")},
        {"candidate_requested_optimization_flags",b.number("requested_optimization_flags")},
        {"runtime_graph_equal",a.text("runtime_graph_sha256")==b.text("runtime_graph_sha256")},
        {"baseline_report_sha256",sha256(read(fs::u8path(spec.at("baseline_report").str()),16*1024*1024))},
        {"candidate_report_sha256",sha256(read(fs::u8path(spec.at("candidate_report").str()),16*1024*1024))},
        {"baseline_output_sha256",sha256(leftBytes)},{"candidate_output_sha256",sha256(rightBytes)}};
}
}
int main(int argc,char** argv) {
    std::string report;bool reportReady=false;
    try {
        require(argc>=2,2,"usage: mgr-cli inspect-model [--manifest FILE] | interpolate|compare|benchmark --run-spec FILE --report FILE");
        const std::string command=argv[1];std::string runSpec,manifestPath;
        for(int i=2;i<argc;i+=2){require(i+1<argc,2,"missing_argument");std::string key=argv[i];
            if(key=="--run-spec") {require(runSpec.empty(),2,"duplicate_run_spec");runSpec=argv[i+1];}
            else if(key=="--report") {require(report.empty(),2,"duplicate_report");report=argv[i+1];}
            else if(key=="--manifest") {require(manifestPath.empty()&&argv[i+1][0],2,"duplicate_or_empty_manifest");manifestPath=argv[i+1];}
            else throw Error(2,"unknown_argument");}
        require(manifestPath.empty()||command=="inspect-model",2,"manifest_only_for_inspect_model");
        if(!report.empty()) {
            const auto target=output_path(report);
            require(!exists_no_follow(target),9,"report_already_exists");
            reportReady=true;
        }
        auto protectInput=[&](const fs::path& input) {
            if(reportReady&&same_path(input,output_path(report))) {reportReady=false;throw Error(4,"report_equals_input");}
        };
        if(!runSpec.empty())protectInput(output_path(runSpec));
        if(!manifestPath.empty())protectInput(output_path(manifestPath));
        auto protectSpecInputs=[&](const J& spec) {
            for(const auto* key:{"left","right","shader_bindings_file","baseline_report","candidate_report","baseline_output","candidate_output"})
                if(spec.has(key)&&!spec.at(key).str().empty())protectInput(output_path(spec.at(key).str()));
            if(spec.has("model_directory")&&!spec.at("model_directory").str().empty()) {
                const auto dir=output_path(spec.at("model_directory").str());
                protectInput(dir/"flownet.param");protectInput(dir/"flownet.bin");
            }
        };
        J output;
        if(command=="inspect-model") {
            output=Parser(meguviz::rife::catalog_manifest).parse();
            if(!manifestPath.empty()) {
                // Apply the same canonical semantic identity check and size
                // limit as mgr_model_create; this inspects metadata only.
                const auto actual=Parser(read(fs::u8path(manifestPath),65536)).parse();
                require(dump(actual)==dump(output),4,"manifest_model_mismatch");
            }
        }
        else if(command=="interpolate") {
            require(!runSpec.empty(),2,"run_spec_required");const auto spec=load(fs::u8path(runSpec));
            protectSpecInputs(spec);
            const auto target=output_path(spec.at("output").str());
            if(reportReady&&same_path(target,output_path(report))) {reportReady=false;throw Error(4,"report_equals_output");}
            require(!exists_no_follow(target),9,"output_already_exists");output=interpolate(spec);
        }
        else if(command=="compare") {require(!runSpec.empty(),2,"run_spec_required");const auto spec=load(fs::u8path(runSpec));protectSpecInputs(spec);output=compare(spec);}
        else if(command=="benchmark") {
            require(!runSpec.empty()&&!report.empty(),2,"benchmark_run_spec_and_report_required");
            const auto spec=load(fs::u8path(runSpec));
            protectSpecInputs(spec);
            const auto target=output_path(spec.text("raw_events"));
            if(reportReady&&same_path(target,output_path(report))) {reportReady=false;throw Error(4,"benchmark_report_equals_raw_events");}
            require(!exists_no_follow(target),9,"benchmark_raw_events_already_exists");
            output=interpolate(spec,false,true);
        }
        else if(command=="export-shader-requests") {
            require(!runSpec.empty(),2,"run_spec_required");const auto spec=load(fs::u8path(runSpec));
            protectSpecInputs(spec);
            const auto directory=output_path(spec.text("shader_request_output_directory"));
            if(reportReady&&contains_path(directory,output_path(report))) {reportReady=false;throw Error(4,"report_inside_shader_output");}
            output=interpolate(spec,true);
        }
        else if(command=="--version-json") output=J::O{{"version","0.1.0-dev"},{"scope","CPU reference and Compute-owned Vulkan implementation; GPU qualification pending"},{"build_fingerprint",meguviz::rife::build_fingerprint},{"private_sdk_fingerprint",meguviz::private_sdk_fingerprint()}};
        else throw Error(3,"unsupported_command");
        if(reportReady)write(output_path(report),dump(output)+"\n");std::cout<<dump(output)<<'\n';return 0;
    } catch(const Error& e) {
        J failure=J::O{{"result","FAIL"},{"cause",e.what()},{"exit_code",e.code}};
        if(reportReady)try{write(output_path(report),dump(failure)+"\n");}catch(...){}
        std::cerr<<dump(failure)<<'\n';return e.code;
    } catch(const std::exception&) {
        J failure=J::O{{"result","FAIL"},{"cause","invalid_input_or_internal_error"},{"exit_code",12}};
        if(reportReady)try{write(output_path(report),dump(failure)+"\n");}catch(...){}
        std::cerr<<dump(failure)<<'\n';return 12;
    }
}
