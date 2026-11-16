#define main observation_model_unused_main
#include "observation_model.cpp"
#undef main

struct Context {MgrSession* second=nullptr;Input* left=nullptr;Input* right=nullptr;Completion nested;
    MgcTicket* nested_ticket=nullptr;MgcStatus nested_status=MGC_INTERNAL_ERROR;int outer_calls=0;};
static void outer_complete(void* ptr,uint64_t,MgcStatus status,MgrResult* result){
    auto& c=*static_cast<Context*>(ptr);++c.outer_calls;
    if(status!=MGC_SUCCESS || !result){mgr_result_release(result);return;}
    mgr_result_release(result);
    MgrRequest d{};d.header=MGC_HEADER_INIT(MgrRequest);d.request_id=2;d.left=c.left->lease;d.right=c.right->lease;d.alpha=0;
    d.output.header=MGC_HEADER_INIT(MgrOutputSpec);d.output.width=32;d.output.height=32;d.output.format=MGC_FORMAT_RGB8;d.output.row_stride_bytes=96;
    mgc_lease_retain(d.left);mgc_lease_retain(d.right);
    c.nested_status=mgr_submit(c.second,&d,completed,&c.nested,nullptr,nullptr,&c.nested_ticket);
    if(c.nested_status!=MGC_ACCEPTED){mgc_lease_release(d.left);mgc_lease_release(d.right);}
}
int main(int argc,char** argv){try{
    require(argc==3,"modeldir catalog");auto graph=read(std::string(argv[1])+"/flownet.param",1024*1024),weights=read(std::string(argv[1])+"/flownet.bin",32*1024*1024),manifest=read(argv[2],1024*1024);
    MgcBuildInfo bi{};bi.header=MGC_HEADER_INIT(MgcBuildInfo);check(mgc_query_build_info(&bi),"build");
    MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=MGC_PROVIDER_NCNN_CPU;rd.memory_budget_bytes=64ULL*1024*1024;MgcRuntime* rt=nullptr;check(mgc_runtime_create(&rd,&rt),"runtime");
    MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);MgcDomain* domain=nullptr;check(mgc_domain_acquire(rt,&dd,&domain),"domain");
    MgrModelDesc md{};md.header=MGC_HEADER_INIT(MgrModelDesc);md.manifest_utf8=manifest.data();md.manifest_size=manifest.size();md.graph_data=graph.data();md.graph_size=graph.size();md.weights_data=weights.data();md.weights_size=weights.size();md.graph_sha256=hex("724569596bcd1e7b9fa50455c604777ebed99746d2ef40aa86e31b5725f1053c");md.weights_sha256=hex("f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958");md.private_sdk_fingerprint=bi.private_sdk_fingerprint;MgrModel* model=nullptr;check(mgr_model_create(domain,&md,&model),"model");
    auto a=input(domain,32,32,0,0),b=input(domain,32,32,1,0),c=input(domain,32,32,2,0),d=input(domain,32,32,3,0);
    MgrPlan* p=plan(model,MGR_PRECISION_FP32_REFERENCE,1);MgrSession* s1=session(p);MgrSession* s2=session(p);Context ctx;ctx.second=s2;ctx.left=c.get();ctx.right=d.get();
    MgrRequest req{};req.header=MGC_HEADER_INIT(MgrRequest);req.request_id=1;req.left=a->lease;req.right=b->lease;req.alpha=0;req.output.header=MGC_HEADER_INIT(MgrOutputSpec);req.output.width=32;req.output.height=32;req.output.format=MGC_FORMAT_RGB8;req.output.row_stride_bytes=96;
    mgc_lease_retain(req.left);mgc_lease_retain(req.right);MgcTicket* t=nullptr;auto status=mgr_submit(s1,&req,outer_complete,&ctx,nullptr,nullptr,&t);
    if(status!=MGC_ACCEPTED){mgc_lease_release(req.left);mgc_lease_release(req.right);}require(status==MGC_ACCEPTED&&t,"outer accepted");
    check(mgr_drain(s1,UINT64_MAX),"outer drain");check(mgr_drain(s2,UINT64_MAX),"inner drain");
    std::cout<<"outer_calls="<<ctx.outer_calls<<" nested_status="<<ctx.nested_status<<" nested_ticket="<<(ctx.nested_ticket!=nullptr)<<" nested_calls="<<ctx.nested.calls<<'\n';
    require(ctx.outer_calls==1&&ctx.nested_status==MGC_ACCEPTED&&ctx.nested_ticket&&ctx.nested.calls==1,"released result must allow callback reentry");
    mgr_result_release(ctx.nested.result);mgc_ticket_release(ctx.nested_ticket);mgc_ticket_release(t);check(mgr_session_close(s1),"s1 close");check(mgr_session_close(s2),"s2 close");mgr_plan_release(p);
    a.reset();b.reset();c.reset();d.reset();mgr_model_release(model);mgc_domain_release(domain);check(mgc_runtime_close(rt),"runtime close");mgc_runtime_release(rt);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
