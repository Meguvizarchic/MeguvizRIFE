#define main observation_model_unused_main
#include "observation_model.cpp"
#undef main
#include <cmath>
#include <limits>
#if defined(_M_X64) || defined(__x86_64__)
#include <xmmintrin.h>
struct ModeCompletion { Completion completion; std::atomic<int> retains{0},releases{0}; unsigned mxcsr=0; bool set=false; bool restore=false; };
static void mode_retain(void* data){++static_cast<ModeCompletion*>(data)->retains;}
static void mode_release(void* data){++static_cast<ModeCompletion*>(data)->releases;}
static void mode_completed(void* data,uint64_t id,MgcStatus status,MgrResult* result){
    auto& c=*static_cast<ModeCompletion*>(data);c.mxcsr=_mm_getcsr();
    if(c.set) _mm_setcsr(c.mxcsr|0x8040u);
    if(c.restore) _mm_setcsr(c.mxcsr&~0x8040u);
    completed(&c.completion,id,status,result);
}
static double from_bits(uint64_t bits){double value;std::memcpy(&value,&bits,sizeof(value));return value;}
#endif

int main(int argc,char** argv){try{
    require(argc==3,"modeldir catalog");
    auto graph=read(std::string(argv[1])+"/flownet.param",1024*1024);
    auto weights=read(std::string(argv[1])+"/flownet.bin",32*1024*1024);
    auto manifest=read(argv[2],1024*1024);
    MgcBuildInfo bi{};bi.header=MGC_HEADER_INIT(MgcBuildInfo);check(mgc_query_build_info(&bi),"build info");
    MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=MGC_PROVIDER_NCNN_CPU;rd.memory_budget_bytes=64ULL*1024*1024;
    MgcRuntime* rt=nullptr;check(mgc_runtime_create(&rd,&rt),"runtime");
    MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);MgcDomain* domain=nullptr;check(mgc_domain_acquire(rt,&dd,&domain),"domain");
    MgrModelDesc md{};md.header=MGC_HEADER_INIT(MgrModelDesc);md.manifest_utf8=manifest.data();md.manifest_size=manifest.size();
    md.graph_data=graph.data();md.graph_size=graph.size();md.weights_data=weights.data();md.weights_size=weights.size();
    md.graph_sha256=hex("724569596bcd1e7b9fa50455c604777ebed99746d2ef40aa86e31b5725f1053c");
    md.weights_sha256=hex("f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958");md.private_sdk_fingerprint=bi.private_sdk_fingerprint;
    MgrModel* model=nullptr;check(mgr_model_create(domain,&md,&model),"model");
    auto left=input(domain,32,32,0,3),right=input(domain,32,32,3,7);
    MgrPlan* p=plan(model,MGR_PRECISION_FP32_REFERENCE,1);MgrSession* s=session(p);
    auto run=[&](double alpha,uint32_t expected,const char* name){
        auto output=submit(s,*left,*right,alpha,1);
        std::cout<<name<<" kind="<<output->info.frame_kind<<" inference_ns="<<output->info.stage_ns[1]<<'\n';
        require(output->info.frame_kind==expected,"requested alpha determines endpoint/inferred kind");
        if(expected==MGR_RESULT_ENDPOINT){
            require(output->tight==pixels(32,32,alpha==0.0?0:3),"endpoint exact bytes");
            require(output->info.stage_ns[1]==0,"endpoint did not run inference");
        }else require(output->info.stage_ns[1]>0,"interior executes CPU model");
    };
    run(0.0,MGR_RESULT_ENDPOINT,"zero");run(1.0,MGR_RESULT_ENDPOINT,"one");
    run(std::nextafter(0.0,1.0),MGR_RESULT_INFERRED,"positive-double-subnormal");
    run(1e-50,MGR_RESULT_INFERRED,"below-float-range");
    run(std::nextafter(1.0,0.0),MGR_RESULT_INFERRED,"below-one-double");
    run(1.0-1e-8,MGR_RESULT_INFERRED,"rounds-to-one-float");
    run(0.5,MGR_RESULT_INFERRED,"midpoint");
    for(double alpha:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::denorm_min(),std::nextafter(1.0,2.0)}){
        MgrRequest d{};d.header=MGC_HEADER_INIT(MgrRequest);d.left=left->lease;d.right=right->lease;d.alpha=alpha;
        d.output.header=MGC_HEADER_INIT(MgrOutputSpec);d.output.width=32;d.output.height=32;d.output.format=MGC_FORMAT_RGB8;d.output.row_stride_bytes=96;
        MgcTicket* ticket=nullptr;Completion c;require(mgr_submit(s,&d,completed,&c,nullptr,nullptr,&ticket)==MGC_INVALID_ARGUMENT&&!ticket&&c.calls==0,"invalid alpha rejected without ownership transfer");
    }
#if defined(_M_X64) || defined(__x86_64__)
    // The previous callback may change the compute worker's FP mode independently
    // of the submitter; compare exact binary64 input under both environments.
    const unsigned original=_mm_getcsr();
    auto mode_run=[&](uint64_t bits,bool valid,uint32_t kind,bool setWorker,bool restoreWorker,const char* label){
        MgrRequest d{};d.header=MGC_HEADER_INIT(MgrRequest);d.left=left->lease;d.right=right->lease;d.alpha=from_bits(bits);
        d.output.header=MGC_HEADER_INIT(MgrOutputSpec);d.output.width=32;d.output.height=32;d.output.format=MGC_FORMAT_RGB8;d.output.row_stride_bytes=96;
        ModeCompletion c;c.set=setWorker;c.restore=restoreWorker;MgcTicket* ticket=nullptr;
        mgc_lease_retain(d.left);mgc_lease_retain(d.right);
        auto status=mgr_submit(s,&d,mode_completed,&c,mode_retain,mode_release,&ticket);
        if(!valid){
            require(status==MGC_INVALID_ARGUMENT&&!ticket&&c.completion.calls==0&&c.retains==0&&c.releases==0,"DAZ invalid rejected synchronously without context transfer");
            mgc_lease_release(d.left);mgc_lease_release(d.right);
        }else{
            require(status==MGC_ACCEPTED&&ticket,"DAZ valid accepted");check(mgr_drain(s,UINT64_MAX),"DAZ drain");
            require(c.completion.calls==1&&c.completion.status==MGC_SUCCESS&&c.completion.result&&c.retains==1&&c.releases==1,"DAZ exactly one callback and balanced context transfer");
            if(!setWorker) require((c.mxcsr&0x8040u)==0x8040u,"worker DAZ and FTZ actually enabled");
            MgrResultInfo info{};info.header=MGC_HEADER_INIT(MgrResultInfo);check(mgr_result_describe(c.completion.result,&info),"DAZ result");
            require(info.frame_kind==kind,"DAZ exact kind");
            require(kind==MGR_RESULT_ENDPOINT?info.stage_ns[1]==0:info.stage_ns[1]>0,"DAZ model dispatch");
            std::cout<<label<<" worker_mxcsr=0x"<<std::hex<<c.mxcsr<<std::dec<<" kind="<<info.frame_kind<<" inference_ns="<<info.stage_ns[1]<<'\n';
            mgr_result_release(c.completion.result);mgc_ticket_release(ticket);
        }
    };
    _mm_setcsr(original|0x8040u);
    mode_run(0x8000000000000001ULL,false,0,false,false,"negative subnormal caller DAZ");
    mode_run(0x7ff8000000000001ULL,false,0,false,false,"NaN caller DAZ");
    mode_run(0x7ff0000000000000ULL,false,0,false,false,"inf caller DAZ");
    mode_run(0x3ff0000000000001ULL,false,0,false,false,"above one caller DAZ");
    _mm_setcsr(original&~0x8040u);
    mode_run(0,true,MGR_RESULT_ENDPOINT,true,false,"set worker DAZ");
    mode_run(0x8000000000000000ULL,true,MGR_RESULT_ENDPOINT,false,false,"negative zero worker DAZ");
    mode_run(0x3ff0000000000000ULL,true,MGR_RESULT_ENDPOINT,false,false,"one worker DAZ");
    _mm_setcsr(original|0x8040u);
    mode_run(1,true,MGR_RESULT_INFERRED,false,false,"positive subnormal caller and worker DAZ");
    _mm_setcsr(original&~0x8040u);
    mode_run(0x3fefffffffffffffULL,true,MGR_RESULT_INFERRED,false,true,"below one worker DAZ");
    _mm_setcsr(original);
#endif
    check(mgr_session_close(s),"session close");mgr_plan_release(p);left.reset();right.reset();mgr_model_release(model);mgc_domain_release(domain);check(mgc_runtime_close(rt),"runtime close");mgc_runtime_release(rt);
    std::cout<<"PASS exact endpoints, representable double interiors, invalid alpha; CPU only\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
