#include <meguviz/rife.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

static void require(bool value,const char* detail){if(!value)throw std::runtime_error(detail);}
static void check(MgcStatus status,const char* detail){if(status!=MGC_SUCCESS)throw std::runtime_error(std::string(detail)+":"+std::to_string(status));}
static std::vector<uint8_t> read(const std::string& file,size_t maximum){std::ifstream s(file,std::ios::binary|std::ios::ate);require(bool(s),"read open");auto size=s.tellg();require(size>=0&&uint64_t(size)<=maximum,"read bound");std::vector<uint8_t> bytes(size_t(size),0);s.seekg(0);if(!bytes.empty())require(bool(s.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()))),"read bytes");return bytes;}
static void write(const std::string& file,const std::vector<uint8_t>& bytes){std::ofstream s(file,std::ios::binary);require(bool(s.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()))),"write bytes");}
static MgcHash hex(const char* text){MgcHash out{};for(size_t i=0;i<32;++i){auto digit=[](char c){return c<='9'?c-'0':c-'a'+10;};out.bytes[i]=uint8_t(digit(text[i*2])*16+digit(text[i*2+1]));}return out;}
static std::vector<uint8_t> pixels(int width,int height,int shift){std::vector<uint8_t> result;result.reserve(size_t(width)*height*3);for(int y=0;y<height;++y)for(int x=0;x<width;++x){int sx=std::clamp(x-shift,0,width-1);result.push_back(uint8_t(sx*255/width));result.push_back(uint8_t(y*255/height));result.push_back(uint8_t((sx/8+y/8)%2?220:30));}return result;}
struct Input {
    MgcResource* resource=nullptr;MgcLease* lease=nullptr;int width=0,height=0,stride=0;uint64_t content=0;
    ~Input(){mgc_lease_release(lease);mgc_resource_release(resource);}
    void update(const std::vector<uint8_t>& tight){
        require(tight.size()==size_t(width)*height*3,"tight input size");mgc_lease_release(lease);lease=nullptr;
        MgcLeaseDesc d{};d.header=MGC_HEADER_INIT(MgcLeaseDesc);d.content_id=++content;d.byte_length=uint64_t(stride)*height;d.access=MGC_ACCESS_WRITE;
        MgcLease* writing=nullptr;check(mgc_lease_create(resource,&d,&writing),"exclusive input update");
        void* data=nullptr;uint64_t size=0;check(mgc_lease_map(writing,&data,&size),"input map");std::memset(data,0xa5,size_t(size));
        for(int y=0;y<height;++y)std::memcpy(static_cast<uint8_t*>(data)+size_t(y)*stride,tight.data()+size_t(y)*width*3,size_t(width)*3);
        check(mgc_lease_unmap(writing),"input unmap");check(mgc_lease_publish_read(writing,&lease),"publish input");mgc_lease_release(writing);
    }
};
static std::unique_ptr<Input> input(MgcDomain* domain,int width,int height,int shift,int padding){auto out=std::make_unique<Input>();out->width=width;out->height=height;out->stride=width*3+padding;
    MgcResourceDesc d{};d.header=MGC_HEADER_INIT(MgcResourceDesc);d.width=width;d.height=height;d.format=MGC_FORMAT_RGB8;d.row_stride_bytes=out->stride;d.capacity_bytes=uint64_t(out->stride)*height;
    check(mgc_resource_create(domain,&d,&out->resource),"input resource");out->update(pixels(width,height,shift));return out;}
struct Completion{std::atomic<int> calls{0};MgcStatus status=MGC_INTERNAL_ERROR;MgrResult* result=nullptr;};
static void completed(void* p,uint64_t,MgcStatus status,MgrResult* result){auto& c=*static_cast<Completion*>(p);c.status=status;c.result=result;++c.calls;}
struct Result {
    MgrResult* value=nullptr;MgrResultInfo info{};MgcLeaseInfo storage{};std::vector<uint8_t> tight;
    ~Result(){mgr_result_release(value);}
    void refresh(){info={};info.header=MGC_HEADER_INIT(MgrResultInfo);check(mgr_result_describe(value,&info),"result describe");require(info.algorithm_accepted!=0&&info.output,"accepted output");
        MgrJobObservation observed{};observed.header=MGC_HEADER_INIT(MgrJobObservation);
        const auto observation_status=mgr_result_observation(value,&observed);
#if defined(MGR_ENABLE_JOB_OBSERVATION)
        require(observation_status==MGC_SUCCESS,"completed observation is available");
        require(observed.request_id==info.request_id&&observed.frame_kind==info.frame_kind,"observation bound to exact result");
        require(observed.tensor_cache_left==MGR_CACHE_OBSERVATION_NOT_APPLICABLE&&observed.tensor_cache_right==MGR_CACHE_OBSERVATION_NOT_APPLICABLE,"CPU tensor cache not applicable");
        require(observed.bridge_cache_left==MGR_CACHE_OBSERVATION_NOT_APPLICABLE&&observed.bridge_cache_right==MGR_CACHE_OBSERVATION_NOT_APPLICABLE,"CPU bridge cache not applicable");
        require(observed.static_guard_dispatch==MGR_GUARD_OBSERVATION_NOT_APPLICABLE&&observed.motion_guard_dispatch==MGR_GUARD_OBSERVATION_NOT_APPLICABLE,"CPU guards not applicable");
        require(observed.effective_flow_scale==(info.frame_kind==MGR_RESULT_ENDPOINT?0u:info.effective_flow_scale),"effective scale bound to result");
        MgrJobObservation partial{};partial.header=MGC_HEADER_INIT(MgrJobObservation);partial.header.struct_size=uint32_t(offsetof(MgrJobObservation,reserved));
        require(mgr_result_observation(value,&partial)==MGC_SUCCESS&&partial.request_id==info.request_id,"required prefix is supported");
#else
        require(observation_status==MGC_UNSUPPORTED_CAPABILITY,"disabled observation is unavailable");
#endif
        storage={};storage.header=MGC_HEADER_INIT(MgcLeaseInfo);check(mgc_lease_info(info.output,&storage),"output info");
        void* data=nullptr;uint64_t size=0;check(mgc_lease_map(info.output,&data,&size),"output map");require(size>=storage.row_stride_bytes*(storage.height-1)+uint64_t(storage.width)*3,"output span");
        tight.resize(size_t(storage.width)*storage.height*3);for(uint32_t y=0;y<storage.height;++y)std::memcpy(tight.data()+size_t(y)*storage.width*3,static_cast<uint8_t*>(data)+y*storage.row_stride_bytes,size_t(storage.width)*3);check(mgc_lease_unmap(info.output),"output unmap");}
};
static std::unique_ptr<Result> submit(MgrSession* session,Input& a,Input& b,double alpha,uint64_t id,MgcStatus expected=MGC_ACCEPTED){
    MgrRequest d{};d.header=MGC_HEADER_INIT(MgrRequest);d.request_id=id;d.left=a.lease;d.right=b.lease;d.alpha=alpha;d.output.header=MGC_HEADER_INIT(MgrOutputSpec);d.output.width=a.width;d.output.height=a.height;d.output.format=MGC_FORMAT_RGB8;d.output.row_stride_bytes=uint64_t(a.width)*3+11;
    mgc_lease_retain(a.lease);mgc_lease_retain(b.lease);Completion c;MgcTicket* ticket=nullptr;const auto status=mgr_submit(session,&d,completed,&c,nullptr,nullptr,&ticket);
    if(status!=MGC_ACCEPTED){mgc_lease_release(a.lease);mgc_lease_release(b.lease);require(status==expected&&ticket==nullptr&&c.calls==0,"rejection preserves ownership and callback");return {};}
    require(expected==MGC_ACCEPTED&&ticket,"accepted ticket");check(mgr_drain(session,UINT64_MAX),"drain");mgc_ticket_release(ticket);require(c.calls==1&&c.status==MGC_SUCCESS&&c.result,"exactly one successful completion");auto result=std::make_unique<Result>();result->value=c.result;result->refresh();return result;
}
static MgrPlan* plan(MgrModel* model,uint32_t profile,uint32_t capacity){MgrPlanDesc d{};d.header=MGC_HEADER_INIT(MgrPlanDesc);d.input_format=MGC_FORMAT_RGB8;d.precision_profile=profile;d.flow_scale=1;d.execution_flags=MGR_EXECUTION_DYNAMIC_SHAPE;d.max_pixels=128*128;d.num_threads=1;d.output_pool_capacity=capacity;MgrPlan* result=nullptr;check(mgr_plan_create(model,&d,&result),"plan");return result;}
static MgrSession* session(MgrPlan* plan){MgcSessionDesc d{};d.header=MGC_HEADER_INIT(MgcSessionDesc);d.max_in_flight=1;MgrSession* result=nullptr;check(mgr_session_create(plan,&d,&result),"session");return result;}
static void identity(MgrPlan* plan,const std::string& file){const char* text=nullptr;uint64_t size=0;check(mgr_plan_execution_info(plan,&text,&size),"actual execution identity");write(file,std::vector<uint8_t>(text,text+size));}
int main(int argc,char** argv){try{
    require(argc==4,"modeldir catalog outputdir");const std::string model_dir=argv[1],catalog=argv[2],out=argv[3];
    require(std::filesystem::is_directory(out)&&std::filesystem::is_empty(out),"output directory must be new and empty");
    auto graph=read(model_dir+"/flownet.param",1024*1024),weights=read(model_dir+"/flownet.bin",32*1024*1024),manifest=read(catalog,1024*1024);
    MgcBuildInfo build{};build.header=MGC_HEADER_INIT(MgcBuildInfo);check(mgc_query_build_info(&build),"build identity");
    MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=MGC_PROVIDER_NCNN_CPU;rd.memory_budget_bytes=64ULL*1024*1024;MgcRuntime* runtime=nullptr;check(mgc_runtime_create(&rd,&runtime),"runtime");
    MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);MgcDomain* domain=nullptr;check(mgc_domain_acquire(runtime,&dd,&domain),"domain");
    MgrModelDesc md{};md.header=MGC_HEADER_INIT(MgrModelDesc);md.manifest_utf8=manifest.data();md.manifest_size=manifest.size();md.graph_data=graph.data();md.graph_size=graph.size();md.weights_data=weights.data();md.weights_size=weights.size();md.graph_sha256=hex("724569596bcd1e7b9fa50455c604777ebed99746d2ef40aa86e31b5725f1053c");md.weights_sha256=hex("f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958");md.private_sdk_fingerprint=build.private_sdk_fingerprint;MgrModel* model=nullptr;check(mgr_model_create(domain,&md,&model),"fixed authenticated model");
    MgrCapabilities caps{};caps.header=MGC_HEADER_INIT(MgrCapabilities);check(mgr_query_capabilities(domain,&caps),"capabilities");
    {
        std::ofstream state(out+"/cpu-capabilities.json");
        state<<"{\"precision_profiles\":"<<caps.precision_profiles<<",\"flow_scales\":"<<caps.flow_scales
             <<",\"input_formats\":"<<caps.input_formats<<",\"guard_flags\":"<<caps.guard_flags
             <<",\"gpu_input_output\":"<<caps.gpu_input_output<<",\"workload_qualified\":"<<caps.workload_qualified<<"}\n";
        require(bool(state),"capability output");
    }
    // Only caller-owned RAM is changed. Plans must not depend on the borrowed
    // graph/weight/manifest buffers after successful model creation.
    std::fill(graph.begin(),graph.end(),uint8_t(0x55));std::fill(weights.begin(),weights.end(),uint8_t(0xaa));std::fill(manifest.begin(),manifest.end(),uint8_t('x'));
    std::unique_ptr<Result> late;
    for(uint32_t profile:{uint32_t(MGR_PRECISION_FP32_REFERENCE),uint32_t(MGR_PRECISION_CPU_FP16_LEGACY)}){
        require((caps.precision_profiles&(1u<<profile))!=0,"CPU compatibility profile is advertised");const std::string tag=profile==1?"fp32":"cpu-fp16-legacy";
        auto* p=plan(model,profile,2);auto* first=session(p);auto* second=session(p);identity(p,out+"/"+tag+"-execution.json");
        auto a=input(domain,32,32,0,7),b=input(domain,32,32,2,9);auto a_tight=pixels(32,32,0),b_tight=pixels(32,32,2);
        write(out+"/"+tag+"-left-32.rgb",a_tight);write(out+"/"+tag+"-right-32.rgb",b_tight);
        auto one=submit(first,*a,*b,.5,1),two=submit(second,*a,*b,.5,2);
        require(one->info.frame_kind==MGR_RESULT_INFERRED&&one->tight==two->tight,"warm repeated input equivalence");require(one->storage.allocation_id!=two->storage.allocation_id,"held outputs own different allocations");
        write(out+"/"+tag+"-output-32.rgb",one->tight);const auto first_allocation=one->storage.allocation_id,first_content=one->storage.content_id;
        submit(first,*a,*b,.5,3,MGC_BACKPRESSURE);MgcLease* held=one->info.output;mgc_lease_retain(held);one.reset();submit(first,*a,*b,.5,4,MGC_BACKPRESSURE);mgc_lease_release(held);
        auto reused=submit(first,*a,*b,.5,5);require(reused->storage.allocation_id==first_allocation&&reused->storage.content_id>first_content,"free pool slot reused with newer content");require(reused->tight==two->tight,"reused output equals held output");
        auto odd_a=input(domain,65,35,0,5),odd_b=input(domain,65,35,2,13);submit(first,*odd_a,*odd_b,.5,6,MGC_BACKPRESSURE);auto retained_pixels=reused->tight;two.reset();auto odd=submit(second,*odd_a,*odd_b,.5,7);reused->refresh();require(reused->tight==retained_pixels,"shape change does not overwrite held previous output");
        write(out+"/"+tag+"-left-odd.rgb",pixels(65,35,0));write(out+"/"+tag+"-right-odd.rgb",pixels(65,35,2));write(out+"/"+tag+"-output-odd.rgb",odd->tight);
        MgrPlanInfo state{};state.header=MGC_HEADER_INIT(MgrPlanInfo);check(mgr_plan_describe(p,&state),"pool state");require(state.output_pool_capacity==2&&state.output_pool_allocated_slots==2,"physical pool remains bounded");
        odd.reset();reused.reset();a->update(pixels(32,32,-3));auto changed=submit(first,*a,*b,.5,8);write(out+"/"+tag+"-left-changed.rgb",pixels(32,32,-3));write(out+"/"+tag+"-output-changed.rgb",changed->tight);
        auto* cold_plan=plan(model,profile,1);
        MgrPlanInfo main_state{},cold_state{};main_state.header=MGC_HEADER_INIT(MgrPlanInfo);cold_state.header=MGC_HEADER_INIT(MgrPlanInfo);
        check(mgr_plan_describe(p,&main_state),"main immutable plan");check(mgr_plan_describe(cold_plan,&cold_state),"cold immutable plan");
        require(main_state.precision_profile==profile&&main_state.output_pool_capacity==2&&cold_state.output_pool_capacity==1,"plans preserve independent copied configuration");
        auto* cold_session=session(cold_plan);auto cold=submit(cold_session,*a,*b,.5,9);require(cold->tight==changed->tight,"content mutation cache matches independent cold plan");
        check(mgr_session_close(cold_session),"cold close");mgr_plan_release(cold_plan);cold.reset();changed.reset();
        auto endpoint=submit(first,*a,*b,0,10);require(endpoint->info.frame_kind==MGR_RESULT_ENDPOINT&&endpoint->tight==pixels(32,32,-3),"endpoint is original bytes not NN");
        check(mgr_session_close(first),"first close with consumer result");check(mgr_session_close(second),"second close with consumer result");mgr_plan_release(p);endpoint->refresh();require(endpoint->tight==pixels(32,32,-3),"owned output survives both sessions and plan");
        late=std::move(endpoint);a.reset();b.reset();odd_a.reset();odd_b.reset();
        std::cout<<"{\"result\":\"PASS\",\"profile\":"<<profile<<",\"scope\":\"CPU pool/repeated-content/shape/stride and owned-output lifecycle\"}\n";
    }
    mgr_model_release(model);mgc_domain_release(domain);require(mgc_runtime_close(runtime)==MGC_BUSY,"consumer output retains domain after model/plan/session close");late->refresh();require(late->tight==pixels(32,32,-3),"late output still readable");late.reset();check(mgc_runtime_close(runtime),"all owned resources returned");mgc_runtime_release(runtime);
    std::cout<<"{\"result\":\"PASS\",\"scope\":\"fixed-model CPU host lifecycle only; no GPU/performance qualification\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
