#define main observation_model_unused_main
#include "observation_model.cpp"
#undef main
#include <meguviz/compute_job.h>
#include <array>
#include <condition_variable>
#include <chrono>
#include <mutex>

struct Counters {
    std::atomic<int> retain{0}, release{0}, completion{0};
    bool throw_retain=false, throw_release=false;
    MgcStatus status=MGC_INTERNAL_ERROR;
    MgrResult* result=nullptr;
};
static void context_retain(void* p) { auto& c=*static_cast<Counters*>(p);++c.retain;if(c.throw_retain)throw std::runtime_error("retain side effect then throw"); }
static void context_release(void* p) { auto& c=*static_cast<Counters*>(p);++c.release;if(c.throw_release)throw std::runtime_error("release side effect then throw"); }
static void context_complete(void* p,uint64_t,MgcStatus status,MgrResult* result) {
    auto& c=*static_cast<Counters*>(p);c.status=status;c.result=result;++c.completion;
}
struct Gate {std::mutex mutex;std::condition_variable cv;bool entered=false, release=false;};
static MgcStatus blocked(void* p,MgcTicket*) {
    auto& g=*static_cast<Gate*>(p);std::unique_lock<std::mutex> lock(g.mutex);
    g.entered=true;g.cv.notify_all();g.cv.wait(lock,[&]{return g.release;});return MGC_SUCCESS;
}
static void unblock(Gate& g){std::lock_guard<std::mutex> lock(g.mutex);g.release=true;g.cv.notify_all();}
static MgrRequest request(Input& a,Input& b,uint64_t id) {
    MgrRequest d{};d.header=MGC_HEADER_INIT(MgrRequest);d.request_id=id;d.left=a.lease;d.right=b.lease;d.alpha=0;
    d.output.header=MGC_HEADER_INIT(MgrOutputSpec);d.output.width=32;d.output.height=32;d.output.format=MGC_FORMAT_RGB8;d.output.row_stride_bytes=96;return d;
}
static MgcStatus submit_owned(MgrSession* s,MgrRequest& d,MgcTicket** t,MgrCompletionFn fn,void* ctx,MgcContextFn retain,MgcContextFn release) {
    mgc_lease_retain(d.left);mgc_lease_retain(d.right);const auto status=mgr_submit(s,&d,fn,ctx,retain,release,t);
    if(status!=MGC_ACCEPTED){mgc_lease_release(d.left);mgc_lease_release(d.right);}return status;
}
template<class T> static std::array<unsigned char,sizeof(T)> raw_snapshot(const T& value) {
    std::array<unsigned char,sizeof(T)> bytes{};std::memcpy(bytes.data(),&value,sizeof(T));return bytes;
}
template<class T> static void unchanged_tail(const std::array<unsigned char,sizeof(T)>& before,const T& after,size_t offset) {
    require(offset<=sizeof(T)&&std::memcmp(before.data()+offset,
        reinterpret_cast<const unsigned char*>(&after)+offset,sizeof(T)-offset)==0,"output tail unchanged");
}
int main(int argc,char** argv){
    Gate gate;MgcRuntime* rt=nullptr;MgcDomain* domain=nullptr;MgrModel* model=nullptr;MgrPlan* p=nullptr;MgrSession* s=nullptr;
    MgcSession* blocker=nullptr;MgcTicket *block_ticket=nullptr,*ticket=nullptr;std::unique_ptr<Input> a,b,c,d;int rc=0;
    Counters throwing,normal,throw_release,cancelled,rejected;Completion plain,retry;
    try{
        require(argc==3,"modeldir catalog");auto graph=read(std::string(argv[1])+"/flownet.param",1024*1024);
        auto weights=read(std::string(argv[1])+"/flownet.bin",32*1024*1024),manifest=read(argv[2],1024*1024);
        MgcBuildInfo bi{};bi.header=MGC_HEADER_INIT(MgcBuildInfo);check(mgc_query_build_info(&bi),"build info");
        MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=MGC_PROVIDER_NCNN_CPU;rd.memory_budget_bytes=64ULL*1024*1024;
        check(mgc_runtime_create(&rd,&rt),"runtime");MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);check(mgc_domain_acquire(rt,&dd,&domain),"domain");
        for(size_t i=0;i<3;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            MgrCapabilities caps{};caps.header=MGC_HEADER_INIT(MgrCapabilities);caps.header.struct_size=uint32_t(offsetof(MgrCapabilities,reserved)+i*sizeof(uint32_t)+cut);
            require(mgr_query_capabilities(domain,&caps)==MGC_UNSUPPORTED_ABI,"partial capability field rejects");
        }
        MgrCapabilities old_caps;std::memset(&old_caps,0xa5,sizeof(old_caps));old_caps.header=MGC_HEADER_INIT(MgrCapabilities);old_caps.header.struct_size=uint32_t(offsetof(MgrCapabilities,reserved));
        const auto old_caps_before=raw_snapshot(old_caps);
        check(mgr_query_capabilities(domain,&old_caps),"complete old capabilities prefix");
        require((old_caps.precision_profiles&(1u<<MGR_PRECISION_FP32_REFERENCE))&&old_caps.flow_scales==(1u<<1)&&old_caps.gpu_input_output==0,"old capabilities values");
        unchanged_tail(old_caps_before,old_caps,offsetof(MgrCapabilities,reserved));
        MgrModelDesc md{};md.header=MGC_HEADER_INIT(MgrModelDesc);md.manifest_utf8=manifest.data();md.manifest_size=manifest.size();md.graph_data=graph.data();md.graph_size=graph.size();md.weights_data=weights.data();md.weights_size=weights.size();
        md.graph_sha256=hex("724569596bcd1e7b9fa50455c604777ebed99746d2ef40aa86e31b5725f1053c");md.weights_sha256=hex("f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958");md.private_sdk_fingerprint=bi.private_sdk_fingerprint;check(mgr_model_create(domain,&md,&model),"model");
        for(size_t i=0;i<3;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            md.header.struct_size=uint32_t(offsetof(MgrModelDesc,reserved)+i*sizeof(uint32_t)+cut);MgrModel* invalid=nullptr;
            require(mgr_model_create(domain,&md,&invalid)==MGC_UNSUPPORTED_ABI&&!invalid,"partial model reserved rejects");
        }
        md.header.struct_size=sizeof(md);
        // A caller may reuse its input buffers once model creation returns.
        // A new model must reject the changed bytes even with the old declared hash.
        weights.front() ^= 1;
        MgrModel* invalid_model=nullptr;
        require(mgr_model_create(domain,&md,&invalid_model)==MGC_MODEL_MISMATCH&&!invalid_model,
            "changed model weights reject");
        graph.front() ^= 1;
        a=input(domain,32,32,0,0);b=input(domain,32,32,2,0);c=input(domain,32,32,3,0);d=input(domain,32,32,4,0);
        // An optional ABI field is either absent or complete; partial values
        // must not silently select a different execution or resource policy.
        MgrPlanDesc pd{};pd.header=MGC_HEADER_INIT(MgrPlanDesc);
        const std::pair<size_t,size_t> plan_fields[]={
            {offsetof(MgrPlanDesc,execution_flags),sizeof(pd.execution_flags)},
            {offsetof(MgrPlanDesc,optimization_flags),sizeof(pd.optimization_flags)},
            {offsetof(MgrPlanDesc,max_pixels),sizeof(pd.max_pixels)},
            {offsetof(MgrPlanDesc,num_threads),sizeof(pd.num_threads)},
            {offsetof(MgrPlanDesc,reserved_tail),sizeof(pd.reserved_tail)},
            {offsetof(MgrPlanDesc,shaderpack),sizeof(pd.shaderpack)},
            {offsetof(MgrPlanDesc,shader_bindings),sizeof(pd.shader_bindings)},
            {offsetof(MgrPlanDesc,shader_binding_count),sizeof(pd.shader_binding_count)},
            {offsetof(MgrPlanDesc,shader_required_flags),sizeof(pd.shader_required_flags)},
            {offsetof(MgrPlanDesc,output_pool_capacity),sizeof(pd.output_pool_capacity)},
            {offsetof(MgrPlanDesc,output_pool_reserved),sizeof(pd.output_pool_reserved)}};
        for(auto field:plan_fields)for(size_t cut=1;cut<field.second;++cut) {
            pd.header.struct_size=uint32_t(field.first+cut);MgrPlan* invalid=nullptr;
            require(mgr_plan_create(nullptr,&pd,&invalid)==MGC_UNSUPPORTED_ABI&&!invalid,"partial plan field rejects");
        }
        for(size_t i=0;i<3;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            pd.header.struct_size=uint32_t(offsetof(MgrPlanDesc,reserved)+i*sizeof(uint32_t)+cut);MgrPlan* invalid=nullptr;
            require(mgr_plan_create(nullptr,&pd,&invalid)==MGC_UNSUPPORTED_ABI&&!invalid,"partial plan reserved rejects");
        }
        pd={};pd.header=MGC_HEADER_INIT(MgrPlanDesc);pd.header.struct_size=uint32_t(offsetof(MgrPlanDesc,reserved));
        pd.width=32;pd.height=32;pd.input_format=MGC_FORMAT_RGB8;pd.precision_profile=MGR_PRECISION_FP32_REFERENCE;pd.flow_scale=1;
        MgrPlan* old_plan=nullptr;check(mgr_plan_create(model,&pd,&old_plan),"complete old plan prefix");mgr_plan_release(old_plan);
        p=plan(model,MGR_PRECISION_FP32_REFERENCE,1);
        for(size_t i=0;i<4;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            MgcSessionDesc sd{};sd.header=MGC_HEADER_INIT(MgcSessionDesc);sd.max_in_flight=1;
            sd.header.struct_size=uint32_t(offsetof(MgcSessionDesc,reserved)+i*sizeof(uint32_t)+cut);MgrSession* invalid=nullptr;
            require(mgr_session_create(p,&sd,&invalid)==MGC_UNSUPPORTED_ABI&&!invalid,"partial session field rejects");
        }
        MgcSessionDesc old_sd{};old_sd.header=MGC_HEADER_INIT(MgcSessionDesc);old_sd.header.struct_size=uint32_t(offsetof(MgcSessionDesc,reserved));old_sd.max_in_flight=1;
        MgrSession* old_session=nullptr;check(mgr_session_create(p,&old_sd,&old_session),"complete old session prefix");check(mgr_session_close(old_session),"old session close");
        MgrPlanInfo old_info;std::memset(&old_info,0xa5,sizeof(old_info));old_info.header=MGC_HEADER_INIT(MgrPlanInfo);old_info.header.struct_size=uint32_t(offsetof(MgrPlanInfo,reserved));
        const auto old_info_before=raw_snapshot(old_info);
        check(mgr_plan_describe(p,&old_info),"complete old plan info prefix");
        require(old_info.precision_profile==MGR_PRECISION_FP32_REFERENCE&&old_info.flow_scale==1,"old plan info values");
        unchanged_tail(old_info_before,old_info,offsetof(MgrPlanInfo,reserved));
        MgrPlanInfo full_info{};full_info.header=MGC_HEADER_INIT(MgrPlanInfo);check(mgr_plan_describe(p,&full_info),"complete current plan info");
        for(size_t i=0;i<9;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            MgrPlanInfo info{};info.header=MGC_HEADER_INIT(MgrPlanInfo);info.header.struct_size=uint32_t(offsetof(MgrPlanInfo,reserved)+i*sizeof(uint32_t)+cut);
            require(mgr_plan_describe(p,&info)==MGC_UNSUPPORTED_ABI,"partial plan info field rejects");
        }
        s=session(p);auto req=request(*a,*b,1);
        const std::pair<size_t,size_t> request_fields[]={
            {offsetof(MgrRequest,flow_scale),sizeof(req.flow_scale)},
            {offsetof(MgrRequest,reserved_tail),sizeof(req.reserved_tail)},
            {offsetof(MgrRequest,output_storage_usage),sizeof(req.output_storage_usage)},
            {offsetof(MgrRequest,output_sync_mode),sizeof(req.output_sync_mode)},
            {offsetof(MgrRequest,batch_event),sizeof(req.batch_event)}};
        for(auto field:request_fields)for(size_t cut=1;cut<field.second;++cut) {
            req.header.struct_size=uint32_t(field.first+cut);
            require(mgr_submit(s,&req,completed,&plain,nullptr,nullptr,&ticket)==MGC_UNSUPPORTED_ABI&&!ticket,"partial request field rejects");
        }
        for(size_t i=0;i<3;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            req.header.struct_size=uint32_t(offsetof(MgrRequest,reserved)+i*sizeof(uint32_t)+cut);
            require(mgr_submit(s,&req,completed,&plain,nullptr,nullptr,&ticket)==MGC_UNSUPPORTED_ABI&&!ticket,"partial request reserved rejects");
        }
        req.header.struct_size=sizeof(req);
        for(size_t i=0;i<4;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            req.output.header.struct_size=uint32_t(offsetof(MgrOutputSpec,reserved)+i*sizeof(uint32_t)+cut);
            require(mgr_submit(s,&req,completed,&plain,nullptr,nullptr,&ticket)==MGC_UNSUPPORTED_ABI&&!ticket,"partial output reserved rejects");
        }
        req.output.header.struct_size=uint32_t(offsetof(MgrOutputSpec,reserved));
        req.header.struct_size=uint32_t(offsetof(MgrRequest,reserved));
        auto status=submit_owned(s,req,&ticket,completed,&plain,nullptr,nullptr);
        require(status==MGC_ACCEPTED&&ticket,"complete old request prefix");check(mgr_drain(s,UINT64_MAX),"old request drain");
        require(plain.calls==1&&plain.result,"old request completion");
        MgrResultInfo old_result;std::memset(&old_result,0xa5,sizeof(old_result));old_result.header=MGC_HEADER_INIT(MgrResultInfo);old_result.header.struct_size=uint32_t(offsetof(MgrResultInfo,reserved));
        const auto old_result_before=raw_snapshot(old_result);
        check(mgr_result_describe(plain.result,&old_result),"complete old result prefix");
        require(old_result.request_id==1&&old_result.output&&old_result.frame_kind==MGR_RESULT_ENDPOINT,"old result values");
        unchanged_tail(old_result_before,old_result,offsetof(MgrResultInfo,reserved));
        MgrResultInfo full_result{};full_result.header=MGC_HEADER_INIT(MgrResultInfo);check(mgr_result_describe(plain.result,&full_result),"complete current result");
        const std::pair<size_t,size_t> result_fields[]={
            {offsetof(MgrResultInfo,stage_ns),sizeof(uint64_t)},
            {offsetof(MgrResultInfo,stage_ns)+sizeof(uint64_t),sizeof(uint64_t)},
            {offsetof(MgrResultInfo,stage_ns)+2*sizeof(uint64_t),sizeof(uint64_t)},
            {offsetof(MgrResultInfo,worker_cpu_ns),sizeof(old_result.worker_cpu_ns)},
            {offsetof(MgrResultInfo,gpu_elapsed_ns),sizeof(old_result.gpu_elapsed_ns)},
            {offsetof(MgrResultInfo,gpu_batch_count),sizeof(old_result.gpu_batch_count)},
            {offsetof(MgrResultInfo,gpu_observed_batch_count),sizeof(old_result.gpu_observed_batch_count)}};
        for(auto field:result_fields)for(size_t cut=1;cut<field.second;++cut) {
            MgrResultInfo info{};info.header=MGC_HEADER_INIT(MgrResultInfo);info.header.struct_size=uint32_t(field.first+cut);
            const auto before=raw_snapshot(info);require(mgr_result_describe(plain.result,&info)==MGC_UNSUPPORTED_ABI&&std::memcmp(before.data(),&info,sizeof(info))==0,"partial result field rejects without output");
        }
        for(size_t i=0;i<7;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
            MgrResultInfo info{};info.header=MGC_HEADER_INIT(MgrResultInfo);info.header.struct_size=uint32_t(offsetof(MgrResultInfo,reserved)+i*sizeof(uint32_t)+cut);
            require(mgr_result_describe(plain.result,&info)==MGC_UNSUPPORTED_ABI,"partial result u32 rejects");
        }
        for(auto group: {std::make_pair(offsetof(MgrResultInfo,worker_cpu_observed),size_t(2)),
                         std::make_pair(offsetof(MgrResultInfo,gpu_elapsed_observed),size_t(4))})
            for(size_t i=0;i<group.second;++i)for(size_t cut=1;cut<sizeof(uint32_t);++cut) {
                MgrResultInfo info{};info.header=MGC_HEADER_INIT(MgrResultInfo);info.header.struct_size=uint32_t(group.first+i*sizeof(uint32_t)+cut);
                require(mgr_result_describe(plain.result,&info)==MGC_UNSUPPORTED_ABI,"partial result tail u32 rejects");
            }
        mgr_result_release(plain.result);plain.result=nullptr;plain.calls=0;
        mgc_ticket_release(ticket);ticket=nullptr;req=request(*a,*b,1);throwing.throw_retain=true;
        status=submit_owned(s,req,&ticket,context_complete,&throwing,context_retain,context_release);
        require(status==MGC_OUT_OF_MEMORY&&!ticket&&throwing.retain==1&&throwing.release==0&&throwing.completion==0,"retain throw rejects without external release");
        req.request_id=2;status=submit_owned(s,req,&ticket,completed,&plain,nullptr,nullptr);
        require(status==MGC_ACCEPTED&&ticket,"capacity available after throwing retain");check(mgr_drain(s,UINT64_MAX),"plain drain");
        require(plain.calls==1&&plain.status==MGC_SUCCESS&&plain.result,"plain completion");mgr_result_release(plain.result);plain.result=nullptr;mgc_ticket_release(ticket);ticket=nullptr;
        req.request_id=3;status=submit_owned(s,req,&ticket,context_complete,&normal,context_retain,context_release);
        require(status==MGC_ACCEPTED,"normal accepted");check(mgr_drain(s,UINT64_MAX),"normal drain");
        require(normal.retain==1&&normal.release==1&&normal.completion==1&&normal.status==MGC_SUCCESS&&normal.result,"normal callback pairing");mgr_result_release(normal.result);normal.result=nullptr;mgc_ticket_release(ticket);ticket=nullptr;
        throw_release.throw_release=true;req.request_id=4;status=submit_owned(s,req,&ticket,context_complete,&throw_release,context_retain,context_release);
        require(status==MGC_ACCEPTED,"release-throw accepted");check(mgr_drain(s,UINT64_MAX),"release-throw drain");
        require(throw_release.retain==1&&throw_release.release==1&&throw_release.completion==1&&throw_release.result,"release-throw callback invoked once");mgr_result_release(throw_release.result);throw_release.result=nullptr;mgc_ticket_release(ticket);ticket=nullptr;
        check(mgr_session_close(s),"first session close");s=nullptr;mgr_plan_release(p);p=nullptr;
        p=plan(model,MGR_PRECISION_FP32_REFERENCE,2);s=session(p);
        MgcSessionDesc sd{};sd.header=MGC_HEADER_INIT(MgcSessionDesc);sd.max_in_flight=1;check(mgc_session_create(domain,&sd,&blocker),"blocker session");
        MgcHostJobDesc hold{};hold.header=MGC_HEADER_INIT(MgcHostJobDesc);hold.request_id=99;hold.execute=blocked;hold.context=&gate;
        require(mgc_private_submit_host_job(blocker,&hold,&block_ticket)==MGC_ACCEPTED,"block accepted");
        {std::unique_lock<std::mutex> lock(gate.mutex);require(gate.cv.wait_for(lock,std::chrono::seconds(10),[&]{return gate.entered;}),"block entered");}
        req=request(*a,*b,5);status=submit_owned(s,req,&ticket,context_complete,&cancelled,context_retain,context_release);require(status==MGC_ACCEPTED,"queued accepted");
        rejected.throw_release=true;auto req2=request(*c,*d,6);MgcTicket* refused=nullptr;
        status=submit_owned(s,req2,&refused,context_complete,&rejected,context_retain,context_release);
        require(status==MGC_BACKPRESSURE&&!refused&&rejected.retain==1&&rejected.release==1&&rejected.completion==0,"rejected retained context released even when throwing");
        require(mgr_cancel(nullptr,ticket)==MGC_INVALID_ARGUMENT,"null session must not cancel ticket");
        auto* foreign=session(p);
        require(mgr_cancel(foreign,ticket)==MGC_INVALID_ARGUMENT,"foreign session must not cancel ticket");
        check(mgr_session_close(foreign),"foreign session close");
        MgcCompletion queued{};queued.header=MGC_HEADER_INIT(MgcCompletion);
        require(mgc_ticket_query(ticket,&queued)==MGC_BUSY&&!queued.result_discarded,"invalid cancellation leaves queued ticket unchanged");
        check(mgr_cancel(s,ticket),"cancel");unblock(gate);check(mgc_session_drain(blocker,UINT64_MAX),"blocker drain");check(mgr_drain(s,UINT64_MAX),"cancel drain");
        require(cancelled.retain==1&&cancelled.release==1&&cancelled.completion==1&&cancelled.status==MGC_CANCELLED&&!cancelled.result,"cancel callback pairing");
        mgc_ticket_release(ticket);ticket=nullptr;req2.request_id=7;status=submit_owned(s,req2,&ticket,completed,&retry,nullptr,nullptr);
        require(status==MGC_ACCEPTED,"capacity after rejection and cancel");check(mgr_drain(s,UINT64_MAX),"retry drain");require(retry.calls==1&&retry.result,"retry success");mgr_result_release(retry.result);retry.result=nullptr;mgc_ticket_release(ticket);ticket=nullptr;
        std::cout<<"PASS retain exception, null callbacks, paired callbacks, release exception, rejection, cancel, capacity\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';rc=1;}
    unblock(gate);if(blocker)mgc_session_drain(blocker,UINT64_MAX);if(s)mgr_drain(s,UINT64_MAX);
    mgr_result_release(plain.result);mgr_result_release(retry.result);mgr_result_release(normal.result);mgr_result_release(throw_release.result);mgr_result_release(cancelled.result);mgr_result_release(rejected.result);
    if(ticket)mgc_ticket_release(ticket);if(block_ticket)mgc_ticket_release(block_ticket);
    if(s)mgr_session_close(s);if(blocker)mgc_session_close(blocker);if(p)mgr_plan_release(p);
    a.reset();b.reset();c.reset();d.reset();if(model)mgr_model_release(model);if(domain)mgc_domain_release(domain);
    if(rt){auto result=mgc_runtime_close(rt);std::cout<<"runtime_close="<<result<<'\n';if(result!=MGC_SUCCESS)rc=1;mgc_runtime_release(rt);}
    return rc;
}
