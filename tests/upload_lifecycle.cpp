#include <meguviz/rife.h>
#include <meguviz/backend_ncnn_sdk.hpp>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
static PFN_vkWaitForFences original_wait = nullptr;
static unsigned held_waits = 0;
static VKAPI_ATTR VkResult VKAPI_CALL held(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t timeout) {
    if(timeout>100000000)std::terminate();
    ++held_waits;return VK_TIMEOUT;
}
static std::string read(const std::string& path) {
    std::ifstream file(path,std::ios::binary);REQUIRE(file.good());
    return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
static MgcHash hash(const char* text) {
    MgcHash result{};auto digit=[](char c){return c<='9'?c-'0':c-'a'+10;};
    for(size_t i=0;i<32;++i)result.bytes[i]=uint8_t(digit(text[2*i])*16+digit(text[2*i+1]));return result;
}
int main(int argc,char** argv) {
    MgcRuntime* runtime=nullptr;MgcDomain* domain=nullptr;MgrModel* model=nullptr;MgrPlan* plan=nullptr;
    try {
        REQUIRE(argc==3);
        auto graph=read(std::string(argv[1])+"/flownet.param"),weights=read(std::string(argv[1])+"/flownet.bin"),manifest=read(argv[2]);
        MgcRuntimeDesc rd{};rd.header=MGC_HEADER_INIT(MgcRuntimeDesc);rd.provider=MGC_PROVIDER_NCNN_VULKAN;rd.memory_budget_bytes=256*1024*1024;
        REQUIRE(mgc_runtime_create(&rd,&runtime)==MGC_SUCCESS);
        MgcDomainDesc dd{};dd.header=MGC_HEADER_INIT(MgcDomainDesc);const auto available=mgc_domain_acquire(runtime,&dd,&domain);
        if(available==MGC_UNSUPPORTED_CAPABILITY){mgc_runtime_release(runtime);return 77;}
        REQUIRE(available==MGC_SUCCESS);
        MgcBuildInfo bi{};bi.header=MGC_HEADER_INIT(MgcBuildInfo);REQUIRE(mgc_query_build_info(&bi)==MGC_SUCCESS);
        MgrModelDesc md{};md.header=MGC_HEADER_INIT(MgrModelDesc);md.manifest_utf8=reinterpret_cast<const uint8_t*>(manifest.data());md.manifest_size=manifest.size();
        md.graph_data=reinterpret_cast<const uint8_t*>(graph.data());md.graph_size=graph.size();md.weights_data=reinterpret_cast<const uint8_t*>(weights.data());md.weights_size=weights.size();md.private_sdk_fingerprint=bi.private_sdk_fingerprint;
        md.graph_sha256=hash("724569596bcd1e7b9fa50455c604777ebed99746d2ef40aa86e31b5725f1053c");
        md.weights_sha256=hash("f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958");
        REQUIRE(mgr_model_create(domain,&md,&model)==MGC_SUCCESS);
        MgrPlanDesc pd{};pd.header=MGC_HEADER_INIT(MgrPlanDesc);pd.width=32;pd.height=32;pd.input_format=MGC_FORMAT_RGB8;
        pd.precision_profile=MGR_PRECISION_VULKAN_FP32_LEGACY;pd.flow_scale=4;pd.execution_flags=MGR_EXECUTION_GPU_IO_DISABLED|MGR_EXECUTION_ADAPTIVE_FLOW;
        pd.num_threads=1;pd.max_pixels=4096;
        REQUIRE(mgr_plan_create(model,&pd,&plan)==MGC_SUCCESS&&plan);
        const char* info=nullptr;uint64_t size=0;REQUIRE(mgr_plan_execution_info(plan,&info,&size)==MGC_SUCCESS);
        const std::string execution(info,size);REQUIRE(execution.find("downscale_6")!=std::string::npos&&execution.find("upscale_8")!=std::string::npos);
        mgr_plan_release(plan);plan=nullptr;
        REQUIRE(mgr_uploads_poll(domain,0)==MGC_SUCCESS);
        original_wait=ncnn::vkWaitForFences;ncnn::vkWaitForFences=held;
        REQUIRE(mgr_plan_create(model,&pd,&plan)==MGC_BUSY&&!plan&&held_waits);
        REQUIRE(mgr_uploads_poll(domain,0)==MGC_BUSY);
        REQUIRE(mgr_uploads_poll(domain,100000001)==MGC_INVALID_ARGUMENT);
        mgr_model_release(model);model=nullptr;mgc_domain_release(domain);domain=nullptr;
        REQUIRE(mgc_runtime_close(runtime)==MGC_BUSY);
        ncnn::vkWaitForFences=original_wait;
        MgcStatus result=MGC_BUSY;
        for(unsigned i=0;i<100&&result==MGC_BUSY;++i)result=mgr_uploads_poll(nullptr,1000000);
        REQUIRE(result==MGC_SUCCESS);
        REQUIRE(mgc_runtime_close(runtime)==MGC_SUCCESS);mgc_runtime_release(runtime);runtime=nullptr;
        std::puts("PASS formal SDK public plan upload, seven networks, failed creation orphan, explicit poll and runtime lifetime");
        return 0;
    }catch(const std::exception& error){
        if(original_wait)ncnn::vkWaitForFences=original_wait;
        std::fprintf(stderr,"FAIL %s\n",error.what());return 1;
    }
}
