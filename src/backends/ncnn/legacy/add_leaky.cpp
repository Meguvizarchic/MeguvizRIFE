#include "add_leaky.h"
#include <memory>
#include <new>
#if NCNN_VULKAN
#include <gpu.h>
#include <pipeline.h>
#include <command.h>
#include "frame_add_leaky.comp.hex.h"
#endif
namespace meguviz::rife::legacy::detail {
class AddLeaky final : public ncnn::Layer {
public:
    AddLeaky() {
#if NCNN_VULKAN
        support_vulkan=true;support_vulkan_packing=true;
#endif
    }
    int load_param(const ncnn::ParamDict& p) override {slope_=p.get(0,0.2f);return 0;}
#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& options) override {
        if(!options.use_vulkan_compute||!vkdev)return -1;
        for(int pack=0;pack<2;pack++) {
            std::string source=frame_add_leaky_comp_data;
            source.insert(source.find('\n')+1,pack?"#define MGR_PACK4 1\n":"#define MGR_PACK4 0\n");
            pipelines_[pack]=std::make_unique<ncnn::Pipeline>(vkdev);pipelines_[pack]->set_optimal_local_size_xyz(64,1,1);
            std::vector<uint32_t> spirv;
            if(ncnn::compile_spirv_module_with_local_size(source.c_str(),int(source.size()),options,
                pipelines_[pack]->local_size_x(),pipelines_[pack]->local_size_y(),pipelines_[pack]->local_size_z(),spirv)||spirv.empty())return -1;
            if(pipelines_[pack]->create(spirv.data(),spirv.size()*sizeof(uint32_t),{}))return -1;
        }
        return 0;
    }
    int destroy_pipeline(const ncnn::Option&) override {for(auto& p:pipelines_)p.reset();return 0;}
    int forward(const std::vector<ncnn::VkMat>& inputs,std::vector<ncnn::VkMat>& outputs,
                ncnn::VkCompute& command,const ncnn::Option& options) const override {
        if(inputs.size()!=2||outputs.size()!=1)return -1;
        const auto& a=inputs[0];const auto& b=inputs[1];const int slot=a.elempack==1?0:a.elempack==4?1:-1;
        if(a.empty()||b.empty()||a.dims!=b.dims||a.w!=b.w||a.h!=b.h||a.c!=b.c||a.d!=b.d||
            a.elemsize!=b.elemsize||a.elempack!=b.elempack||a.cstep!=b.cstep||slot<0||!pipelines_[slot])return -1;
        outputs[0].create_like(a,options.blob_vkallocator);if(outputs[0].empty())return -100;
        const auto count=a.cstep*a.c;if(count>INT32_MAX)return -1;
        std::vector<ncnn::vk_constant_type> c(2);c[0].i=int(count);c[1].f=slope_;
        ncnn::VkMat dispatch;dispatch.dims=1;dispatch.w=int(count);dispatch.h=dispatch.c=1;
        command.record_pipeline(pipelines_[slot].get(),{a,b,outputs[0]},c,dispatch);return 0;
    }
private:
    std::unique_ptr<ncnn::Pipeline> pipelines_[2];
#endif
    float slope_=0.2f;
};
ncnn::Layer* createAddLeaky(void*) {return new(std::nothrow) AddLeaky();}
}
