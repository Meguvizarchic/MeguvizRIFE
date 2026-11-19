#include "frame_engine.h"
#if defined(MGR_ENABLE_JOB_OBSERVATION)
#define MGR_OBSERVE_JOB(...) __VA_ARGS__
#else
#define MGR_OBSERVE_JOB(...)
#endif

#include "upstream/rife_warp.h"
#include "add_leaky.h"
#include "private/data.hpp"
#include "private/alpha.hpp"
#include <meguviz/backend_ncnn_sdk.hpp>
#if NCNN_VULKAN
#if defined(__ANDROID__)
#include <meguviz/compute_android.h>
#endif
#include "frame_preproc.comp.hex.h"
#include "frame_postproc.comp.hex.h"
#include "frame_fused.comp.hex.h"
#include "frame_fused_rgb8.comp.hex.h"
#include "frame_reduce.comp.hex.h"
#include "frame_gpu_unpack.comp.hex.h"
#include "frame_gpu_image_read.comp.hex.h"
#include "frame_gpu_statistics.comp.hex.h"
#include "frame_gpu_patch.comp.hex.h"
#include "frame_gpu_image_write.comp.hex.h"
#include <pipeline.h>
#endif
#include <command.h>
#include <datareader.h>
#include <gpu.h>
#include <mat.h>
#include <net.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <optional>
#include <tuple>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <functional>

namespace meguviz::rife::legacy {
namespace {
constexpr std::size_t MaxParamBytes = 1024u * 1024u;
constexpr std::size_t MaxWeightBytes = 32u * 1024u * 1024u;
constexpr std::size_t MaxPixels = 4096u * 2160u;
constexpr int MaxDimension = 4096;

class BoundedReader final : public ncnn::DataReader {
public:
    BoundedReader(const std::uint8_t* bytes, std::size_t size) : bytes_(bytes), size_(size) {}
    std::size_t read(void* destination, std::size_t size) const override {
        if (size > size_ - offset_ || (size != 0 && destination == nullptr)) {
            truncated_ = true;
            return 0;
        }
        if (size != 0) std::memcpy(destination, bytes_ + offset_, size);
        offset_ += size;
        return size;
    }
    // Never expose borrowed weights as a retained model pointer; ncnn must copy through read.
    std::size_t reference(std::size_t, const void**) const override { return 0; }
    bool truncated() const noexcept { return truncated_; }
private:
    const std::uint8_t* bytes_;
    std::size_t size_;
    mutable std::size_t offset_ = 0;
    mutable bool truncated_ = false;
};

bool layoutValid(const void* data, int width, int height, std::size_t stride,
                 std::size_t capacity, std::size_t maxPixels) noexcept {
    if (data == nullptr || width <= 0 || height <= 0 || width > MaxDimension || height > MaxDimension)
        return false;
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (w > maxPixels / h) return false;
    const auto rowBytes = w * 3u;
    if (stride < rowBytes || (h > 1 && stride > (std::numeric_limits<std::size_t>::max() - rowBytes) / (h - 1)))
        return false;
    return (h - 1) * stride + rowBytes <= capacity;
}

bool modelShapeValid(const ncnn::Net& net, bool reducedOnly = false) {
    const auto& inputs = net.input_names();
    const auto& outputs = net.output_names();
    if (inputs.size() != 3) return false;
    if (!reducedOnly && (outputs.size() != 1 || std::strcmp(outputs[0], "out0") != 0)) return false;
    if (reducedOnly) {
        bool flow=false,mask=false;
        for(const auto* name:outputs) {flow|=std::strcmp(name,"327")==0;mask|=std::strcmp(name,"332")==0;}
        if(!flow||!mask) return false;
    }
    for (const char* required : {"in0", "in1", "in2"}) {
        bool found = false;
        for (const char* input : inputs) found = found || std::strcmp(required, input) == 0;
        if (!found) return false;
    }
    return true;
}

// The approved v4.6 graph is hash checked by the public model loader. Only
// weightless descendants are removed, so the original weight stream is exact.
std::string pruneTail(const std::string& param) {
    struct Node {std::string text,type;std::vector<std::string> inputs,outputs;};
    std::istringstream source(param);int magic=0,layers=0,blobs=0;source>>magic>>layers>>blobs;
    std::string line;std::getline(source,line);std::vector<Node> nodes;std::map<std::string,size_t> producers;
    while(std::getline(source,line)) {
        if(line.empty())continue;std::istringstream row(line);Node node;node.text=line;std::string name;int in=0,out=0;
        if(!(row>>node.type>>name>>in>>out)||in<0||out<0||in>32||out>32)throw std::runtime_error("graph_shape");
        std::string blob;
        for(int i=0;i<in;i++){if(!(row>>blob))throw std::runtime_error("graph_input");node.inputs.push_back(blob);}
        for(int i=0;i<out;i++){if(!(row>>blob)||!producers.emplace(blob,nodes.size()).second)throw std::runtime_error("graph_output");node.outputs.push_back(blob);}
        nodes.push_back(std::move(node));
    }
    if(int(nodes.size())!=layers)throw std::runtime_error("graph_layers");
    std::set<size_t> keep;std::function<void(const std::string&)> visit=[&](const std::string& blob) {
        const auto found=producers.find(blob);if(found==producers.end())throw std::runtime_error("graph_dependency");
        if(!keep.insert(found->second).second)return;for(const auto& input:nodes[found->second].inputs)visit(input);
    };
    visit("327");visit("332");std::set<std::string> outputBlobs;
    std::string body;
    for(size_t i=0;i<nodes.size();i++) {
        if(keep.count(i)){body+=nodes[i].text+"\n";outputBlobs.insert(nodes[i].outputs.begin(),nodes[i].outputs.end());}
        else if(nodes[i].type!="Split"&&nodes[i].type!="BinaryOp"&&nodes[i].type!="Crop"&&nodes[i].type!="rife.Warp")throw std::runtime_error("weighted_tail");
    }
    return "7767517\n"+std::to_string(keep.size())+" "+std::to_string(outputBlobs.size())+"\n"+body;
}

std::string fuseAddLeaky(const std::string& param) {
    struct Node {std::string type,name,text;std::vector<std::string> inputs,outputs,params;};
    std::istringstream source(param);int magic=0,layers=0,blobs=0;source>>magic>>layers>>blobs;
    std::string line;std::getline(source,line);std::vector<Node> nodes;std::map<std::string,int> consumers;
    while(std::getline(source,line)) {
        if(line.empty())continue;std::istringstream row(line);Node n;n.text=line;int in=0,out=0;
        if(!(row>>n.type>>n.name>>in>>out)||in<0||out<0||in>32||out>32)throw std::runtime_error("fusion_graph");
        std::string token;for(int i=0;i<in;i++){if(!(row>>token))throw std::runtime_error("fusion_input");n.inputs.push_back(token);consumers[token]++;}
        for(int i=0;i<out;i++){if(!(row>>token))throw std::runtime_error("fusion_output");n.outputs.push_back(token);}
        while(row>>token)n.params.push_back(token);nodes.push_back(std::move(n));
    }
    if(int(nodes.size())!=layers)throw std::runtime_error("fusion_layers");
    std::string body;std::set<std::string> outputs;int kept=0;
    for(size_t i=0;i<nodes.size();i++) {
        auto& a=nodes[i];bool fused=false;
        if(i+1<nodes.size()&&a.type=="BinaryOp"&&a.inputs.size()==2&&a.outputs.size()==1&&
            consumers[a.outputs[0]]==1&&(a.params.empty()||(a.params.size()==1&&a.params[0]=="0=0"))) {
            auto& b=nodes[i+1];
            if(b.type=="ReLU"&&b.inputs.size()==1&&b.outputs.size()==1&&b.inputs[0]==a.outputs[0]&&
                b.params.size()==1&&b.params[0].rfind("0=",0)==0) {
                const float slope=std::stof(b.params[0].substr(2));
                if(std::isfinite(slope)&&slope>0.f) {
                    body+="rife.AddLeaky "+b.name+" 2 1 "+a.inputs[0]+" "+a.inputs[1]+" "+b.outputs[0]+" "+b.params[0]+"\n";
                    outputs.insert(b.outputs[0]);kept++;i++;fused=true;
                }
            }
        }
        if(!fused){body+=a.text+"\n";outputs.insert(a.outputs.begin(),a.outputs.end());kept++;}
    }
    return "7767517\n"+std::to_string(kept)+" "+std::to_string(outputs.size())+"\n"+body;
}

Status gpuReady(const ncnn::VulkanDevice* device) {
#if NCNN_VULKAN
    if (device == nullptr) return {Code::GpuUnavailable};
    return {};
#else
    (void)device;
    return {Code::UnsupportedBackend};
#endif
}

ncnn::Mat inputTensor(ImageView image, int paddedWidth, int paddedHeight) {
    // NEON 路径的打包与归一化；补零到模型要求的倍数，与上游 v4 无 TTA 流程一致。
    ncnn::Mat packed = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, image.width, image.height,
                                              static_cast<int>(image.strideBytes));
    if (packed.empty()) return packed;
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    packed.substract_mean_normalize(nullptr, norm);
    if (paddedWidth == image.width && paddedHeight == image.height) return packed;
    ncnn::Mat padded;
    ncnn::copy_make_border(packed, padded, 0, paddedHeight - image.height, 0, paddedWidth - image.width,
                           ncnn::BORDER_CONSTANT, 0.f);
    return padded;
}

// 低分辨率光流（scale = 2 或 4）：GPU 下采样输入，flownet 只在 1/scale 分辨率上估计光流与融合遮罩，
// 再把光流（数值 ×scale）与遮罩上采样到全分辨率，用全分辨率原帧做 Warp 与融合。
// 末段与 flownet.param 的 warp_6/warp_7/mul_76/mul_77/add_78 同构：blob 327 为 4 通道光流，332 为遮罩。
// 运动置信度分块：与 frame_fused*.comp 的 GUARD_TILES_X/Y 一致。
constexpr int GuardTilesX = 16, GuardTilesY = 9, GuardTiles = GuardTilesX * GuardTilesY;
#if NCNN_VULKAN
struct GuardSampleCounts {
    int width = 0, height = 0;
    int64_t total = 0;
    std::array<int64_t, GuardTiles> tiles{};

    void update(int frameWidth, int frameHeight) noexcept {
        if (width == frameWidth && height == frameHeight) return;
        width = frameWidth;
        height = frameHeight;
        total = 0;
        tiles.fill(0);
        // Each shader invocation covers four pixels. Every eighth invocation
        // attributes its whole group to the tile containing the first pixel.
        const int64_t pixels = int64_t(width) * height;
        for (int64_t first = 0; first < pixels; first += 32) {
            const int64_t y = first / width, x = first - y * width;
            const int64_t count = std::min<int64_t>(4, pixels - first);
            tiles[size_t((y * GuardTilesY / height) * GuardTilesX + x * GuardTilesX / width)] += count;
            total += count;
        }
    }
};
#endif
// 分块不可靠像素千分比：达到前者直接换成最近原帧；达到后者且与已换分块相邻时一并替换。
constexpr int TilePatchPermille = 300, TileNeighbourPermille = 50;

// 静止分块：抽样平均每通道绝对差不超过此值（0..255）。
constexpr double StaticTileDiff = 3.0;

// 每个 16×9 分块在两帧间的抽样平均通道绝对差（约 320×180 网格）。
void tileMotion(const ImageView& a, const ImageView& b, std::array<double, GuardTiles>& out) {
    const int stepX = std::max(1, a.width / 320), stepY = std::max(1, a.height / 180);
    std::array<std::uint64_t, GuardTiles> sum{}, count{};
    for (int y = 0; y < a.height; y += stepY) {
        const auto* ra = a.data + static_cast<std::size_t>(y) * a.strideBytes;
        const auto* rb = b.data + static_cast<std::size_t>(y) * b.strideBytes;
        const int ty = y * GuardTilesY / a.height;
        for (int x = 0; x < a.width; x += stepX) {
            const int t = ty * GuardTilesX + x * GuardTilesX / a.width;
            for (int c = 0; c < 3; ++c)
                sum[t] += static_cast<std::uint64_t>(std::abs(static_cast<int>(ra[x * 3 + c]) - rb[x * 3 + c]));
            count[t] += 3;
        }
    }
    for (int t = 0; t < GuardTiles; ++t) out[t] = count[t] ? static_cast<double>(sum[t]) / count[t] : 0.0;
}

// 两帧颜色分布的差异（16 级 × 3 通道直方图的 L1 距离，0..1000）：切镜明显改变颜色构成，摇镜基本不变。
int histogramDistancePermille(const ImageView& a, const ImageView& b) {
    const int stepX = std::max(1, a.width / 320), stepY = std::max(1, a.height / 180);
    std::array<std::int64_t, 48> bins{};
    std::int64_t samples = 0;
    for (int y = 0; y < a.height; y += stepY) {
        const auto* ra = a.data + static_cast<std::size_t>(y) * a.strideBytes;
        const auto* rb = b.data + static_cast<std::size_t>(y) * b.strideBytes;
        for (int x = 0; x < a.width; x += stepX) {
            for (int c = 0; c < 3; ++c) { ++bins[c * 16 + (ra[x * 3 + c] >> 4)]; --bins[c * 16 + (rb[x * 3 + c] >> 4)]; }
            ++samples;
        }
    }
    std::int64_t l1 = 0;
    for (const auto v : bins) l1 += v < 0 ? -v : v;
    return samples ? static_cast<int>(l1 * 1000 / (6 * samples)) : 0;
}

// 约 320×180 网格的通道均值（0..255）；只读原图，成本与分辨率无关。
void sampleMeans(const ImageView& image, double out[3]) {
    const int stepX = std::max(1, image.width / 320), stepY = std::max(1, image.height / 180);
    std::uint64_t sum[3] = {0, 0, 0}, count = 0;
    for (int y = 0; y < image.height; y += stepY) {
        const auto* row = image.data + static_cast<std::size_t>(y) * image.strideBytes;
        for (int x = 0; x < image.width; x += stepX) {
            for (int c = 0; c < 3; ++c) sum[c] += row[static_cast<std::size_t>(x) * 3 + c];
            ++count;
        }
    }
    for (int c = 0; c < 3; ++c) out[c] = count ? static_cast<double>(sum[c]) / count : 0.0;
}

// adaptiveFlow：像素数超过此值的帧用 1/8 光流（1080p 及以下保持 1/4）。
constexpr std::int64_t AdaptiveFlowPixels = 1920LL * 1080LL;

std::string downscaleParam(int scale) {
    const std::string factor = std::to_string(1.0 / scale);
    return "7767517\n4 4\nInput in0 0 1 in0\nInput in1 0 1 in1\n"
           "Interp down0 1 1 in0 half0 0=2 1=" + factor + " 2=" + factor + "\n"
           "Interp down1 1 1 in1 half1 0=2 1=" + factor + " 2=" + factor + "\n";
}

std::string upscaleParam(int scale) {
    const std::string factor = std::to_string(static_cast<double>(scale));
    return "7767517\n17 19\n"
           "Input in0 0 1 in0\nInput in1 0 1 in1\nInput flow 0 1 flow\nInput mask 0 1 mask\n"
           "Interp upflow 1 1 flow flowup 0=2 1=" + factor + " 2=" + factor + "\n"
           "BinaryOp scaleflow 1 1 flowup flow2 0=2 1=1 2=" + factor + "\n"
           "Split splitflow 1 2 flow2 fa fb\n"
           "Crop flow0 1 1 fa f0 -23309=1,0 -23310=1,2 -23311=1,0\n"
           "Crop flow1 1 1 fb f1 -23309=1,2 -23310=1,4 -23311=1,0\n"
           "Interp upmask 1 1 mask maskup 0=2 1=" + factor + " 2=" + factor + "\n"
           "Split splitmask 1 2 maskup m0 m1\n"
           "BinaryOp invmask 1 1 m1 minv 0=7 1=1 2=1.0\n"
           "rife.Warp warp0 2 1 in0 f0 w0\nrife.Warp warp1 2 1 in1 f1 w1\n"
           "BinaryOp blend0 2 1 w0 m0 a 0=2\nBinaryOp blend1 2 1 w1 minv b 0=2\n"
           "BinaryOp merge 2 1 a b out0\n";
}

constexpr const char* FlowBlob = "327";
constexpr const char* MaskBlob = "332";

std::int64_t nowNs() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
mgr_detail::J graphOptions(const ncnn::Option& o) {
    return mgr_detail::Parser(meguviz_options_detail::execution_options_json(o)).parse();
}
mgr_detail::J executionGraphOptions(const ncnn::Option& o) {
    return graphOptions(o);
}
} // namespace

class Engine::Impl {
public:
    explicit Impl(Options value) : options(value) { mgc_shaderpack_retain(options.shaderpack); }
    ~Impl() {
        // Vulkan instances reach this destructor only after closeOwned() has
        // destroyed their GPU state under the actual Compute owner scope.
        releaseNets(); releaseAllocators();
        mgc_shaderpack_release(options.shaderpack);
    }
    Options options;
    std::mutex mutex;
    // Protects the owner handle across scope return and release. Every entry
    // uses try_lock, so a device-scope caller never waits on an inverse edge.
    std::mutex lifecycleMutex;
    // A continuation owns this token until all of its command-dependent
    // objects have been retired and released. No mutex remains locked while
    // the accepted job is waiting for Compute to resume it.
    const void* activeInference = nullptr;
    std::atomic<bool> loaded{false};
    bool closed = false;
    bool fp16 = false;
    int flowScale = 1;
    bool gpuIo = false;
    bool protectStaticDetails = false;
    bool fusedWarp = false;
    bool rawGpuInput = false;
    bool adaptiveFlow = false;
    // 运动置信度保护（融合路径）：逐像素回退到未 warp 的时间加权混合，并统计不可靠像素比例。
#if defined(MGR_ENABLE_JOB_OBSERVATION)
    std::atomic<bool> motionGuard{true};
    JobObservation diagnosticLast{};
#else
    bool motionGuard = true;
#endif
    std::atomic<int> unreliablePermille{0};
    std::atomic<int> patchedTiles{0};
    std::atomic<int> histogramPermille{0};
    std::unique_ptr<ncnn::Net> network;
    std::unique_ptr<ncnn::Net> downscale;
    std::unique_ptr<ncnn::Net> upscale;
    // adaptiveFlow 时另备 1/6 与 1/8 光流的缩放网络（无权重），按 [scale] 索引；可逐次选择。
    std::array<std::unique_ptr<ncnn::Net>, 9> extraDown;
    std::array<std::unique_ptr<ncnn::Net>, 9> extraUp;
    std::vector<std::uint8_t> staging;
    std::string graphText;
    std::string catalogText;
    std::string executionText;
    uint32_t shaderCounters[3]{};
    ncnn::Option requestedNetworkOptions;
    std::atomic<std::int64_t> timings[3] = {};
#if NCNN_VULKAN
    static constexpr std::uint64_t UploadWaitLimitNs = 100000000;
    meguviz::NcnnDomainOwner* uploadOwner = nullptr;
    std::atomic<bool> uploadsPending{false};
    std::array<std::unique_ptr<ncnn::VkTransfer>,7> uploads;
    std::array<std::string,7> uploadParams;
    std::vector<std::uint8_t> uploadWeights;
    struct RetirementSlot { Impl* impl=nullptr; bool abandoned=false,polling=false; std::shared_ptr<Impl> orphan; };
    struct RetirementRegistry { std::mutex mutex; std::array<RetirementSlot,16> slots; };
    static RetirementRegistry& registry() { static auto* value=new RetirementRegistry; return *value; }
    std::size_t retirementSlot = 16;

    MgcStatus initializeOwner() {
        if(options.backend!=Backend::Vulkan)return MGC_SUCCESS;
        if(!options.computeDomain)return MGC_INVALID_ARGUMENT;
        auto& r=registry();
        {
            std::lock_guard<std::mutex> lock(r.mutex);
            for(std::size_t i=0;i<r.slots.size();++i)if(!r.slots[i].impl) {
                r.slots[i].impl=this;retirementSlot=i;break;
            }
        }
        if(retirementSlot==r.slots.size())return MGC_BACKPRESSURE;
        auto status=meguviz::ncnn_domain_owner_create(options.computeDomain,&uploadOwner);
        if(status!=MGC_SUCCESS){unregisterOwner();return status;}
        // Reject a borrowed device that does not belong to the retained domain.
        // The validation scope records no work and explicitly clears its mark.
        status=meguviz::with_ncnn_domain_owner(uploadOwner,[](void* p,const ncnn::VulkanDevice* device) -> MgcStatus {
            auto* self=static_cast<Impl*>(p);
            const auto cleared=meguviz::ncnn_domain_owner_pending(self->uploadOwner,false);
            return cleared!=MGC_SUCCESS?cleared:device==self->options.domainDevice?MGC_SUCCESS:MGC_INCOMPATIBLE_DOMAIN;
        },this);
        return status;
    }
    void unregisterOwner() noexcept {
        if(retirementSlot==16)return;
        auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);
        if(r.slots[retirementSlot].impl==this)r.slots[retirementSlot]=RetirementSlot{};
        retirementSlot=16;
    }
    bool refreshPending() noexcept {
        bool pending=false;
        for(const auto& transfer:uploads)pending|=transfer&&!transfer->transfer_batches_retired();
        uploadsPending.store(pending);
        return pending;
    }
    bool observeUploadLoss() noexcept {
        for(const auto& transfer:uploads)if(transfer&&
            (transfer->last_submit_result()==VK_ERROR_DEVICE_LOST||transfer->last_wait_result()==VK_ERROR_DEVICE_LOST))
            meguviz::ncnn_report_device_lost(options.computeDomain);
        const bool lost=meguviz::ncnn_domain_is_lost(options.computeDomain);
        if(lost)loaded.store(false);
        return lost;
    }
    MgcStatus retireUploads(std::uint64_t waitNs) noexcept {
        if(observeUploadLoss()) {
            if(!meguviz::ncnn_domain_loss_retired(options.computeDomain))return MGC_BUSY;
            for(auto& transfer:uploads)if(transfer)transfer->retire_device_lost();
            uploadsPending.store(false);
            return MGC_DEVICE_LOST;
        }
        for(auto& transfer:uploads)
            if(transfer&&!transfer->transfer_batches_retired())transfer->wait_transfer_retirement(waitNs);
        if(observeUploadLoss()) {
            if(!meguviz::ncnn_domain_loss_retired(options.computeDomain))return MGC_BUSY;
            for(auto& transfer:uploads)if(transfer)transfer->retire_device_lost();
            uploadsPending.store(false);
            return MGC_DEVICE_LOST;
        }
        return refreshPending()?MGC_BUSY:MGC_SUCCESS;
    }
    int loadNetwork(ncnn::Net& net,const ncnn::DataReader& reader,std::size_t slot) {
        if(options.backend!=Backend::Vulkan)return net.load_model(reader);
        if(meguviz::ncnn_domain_is_lost(options.computeDomain))return -1;
        // The Impl and its fixed registry slot already own Net/model/allocator.
        // Install the command owner before load_model can perform its first batch.
        uploads[slot]=std::make_unique<ncnn::VkTransfer>(options.domainDevice);
        if(!uploads[slot]->is_valid())return -1;
        return net.load_model_with_transfer(reader,uploads[slot].get(),UploadWaitLimitNs);
    }
    MgcStatus closeOwned(std::uint64_t waitNs) noexcept {
        if(activeInference)return MGC_BUSY;
        const auto retirement=retireUploads(waitNs);
        if(retirement!=MGC_SUCCESS&&retirement!=MGC_DEVICE_LOST)return retirement;
        loaded.store(false);closed=true;
        releaseNets();releaseAllocators();
        mgc_shaderpack_release(options.shaderpack);options.shaderpack=nullptr;
        return meguviz::ncnn_domain_owner_pending(uploadOwner,false);
    }
    MgcStatus closeWithOwner(std::uint64_t waitNs) noexcept {
        if(!uploadOwner)return MGC_SUCCESS;
        if(observeUploadLoss()) {
            mgc_domain_service(options.computeDomain,0);
            if(!meguviz::ncnn_domain_loss_retired(options.computeDomain))return MGC_BUSY;
        }
        struct Call { Impl* self;std::uint64_t wait; } call{this,waitNs};
        const auto enter=meguviz::ncnn_domain_is_lost(options.computeDomain)?
            meguviz::try_with_ncnn_domain_owner_retire:meguviz::try_with_ncnn_domain_owner;
        const auto status=enter(uploadOwner,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
            auto& call=*static_cast<Call*>(p);
            std::unique_lock<std::mutex> lock(call.self->mutex,std::try_to_lock);
            if(!lock.owns_lock())return MGC_BUSY;
            return call.self->closeOwned(call.wait);
        },&call);
        if(status!=MGC_SUCCESS)return status;
        const auto released=meguviz::ncnn_domain_owner_release(uploadOwner);
        if(released==MGC_SUCCESS){uploadOwner=nullptr;unregisterOwner();}
        return released;
    }
    static void abandon(std::shared_ptr<Impl>& impl) noexcept {
        auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);
        auto* raw=impl.get();
        if(raw->retirementSlot<r.slots.size()&&r.slots[raw->retirementSlot].impl==raw) {
            r.slots[raw->retirementSlot].abandoned=true;r.slots[raw->retirementSlot].orphan=std::move(impl);
        }
    }
    // 常驻显存池：避免每次推理重新申请/释放 VkBuffer；网络必须先于池释放。
    const ncnn::VulkanDevice* device = nullptr;
    ncnn::VkAllocator* blobAllocator = nullptr;
    ncnn::VkAllocator* stagingAllocator = nullptr;
    // 每个源帧会在相邻两次插帧中各作一次输入（后一帧 → 下一次的前一帧）。按帧 ID 缓存其显存张量，
    // 省去重复的上传、解包与下采样；帧 ID 由所有权账本保证唯一且永不复用，内容在其存续期内不变。
    struct GpuFrame {
        std::int64_t id = -1;
        int width = 0, height = 0;
        // Only the explicit rawGpuInput experiment retains RGB8; the default retains planar input.
        ncnn::VkMat full, reduced;
        int sourceWidth = 0, sourceHeight = 0;
        std::size_t sourceStride = 0;
        // 下采样输入的光流比例；不同比例的 reduced 尺寸不同，缓存命中必须一致。
        int scale = 0;
    };
    GpuFrame frames[3];
    int nextFrame = 0;
    void clearFrames() { for (auto& frame : frames) frame = GpuFrame{}; nextFrame = 0; }
    // RGB8 <-> 平面浮点的 GPU 转换管线；与网络同一 Option 编译，fp16 存储保持一致。
    struct PipelineSlot {
        std::unique_ptr<ncnn::Pipeline> owned;
        ncnn::Pipeline* borrowed=nullptr;
        ncnn::Pipeline* get() const {return borrowed?borrowed:owned.get();}
        void reset(){owned.reset();borrowed=nullptr;}
        PipelineSlot& operator=(std::unique_ptr<ncnn::Pipeline> p){owned=std::move(p);borrowed=nullptr;return *this;}
    };
    PipelineSlot preproc,postproc,fused,reduce,gpuUnpack,gpuImageRead,gpuStatistics,gpuPatch,gpuImageWrite;
    mgr_detail::J::A shaderRecords;
    MgcStatus makePipeline(uint32_t kernel,const char* logical,const char* source,int size,const ncnn::Option& effective,
                           int x,int y,int z,const std::vector<int>& bindings,int pushCount,PipelineSlot& slot);
    struct GpuLeaseFrame { int64_t id = -1; int width=0,height=0,scale=0; ncnn::VkMat raw,full,reduced; };
    GpuLeaseFrame leaseFrames[3];
    int nextLeaseFrame = 0;
    GuardSampleCounts guardSampleCounts; // Accessed under the inference mutex.
#endif

    void releaseNets() noexcept {
        catalogText.clear();executionText.clear();for(auto& count:shaderCounters)count=0;
#if NCNN_VULKAN
        for(auto& transfer:uploads)transfer.reset();
        shaderRecords.clear();
        clearFrames();
        for (auto& frame : leaseFrames) frame = GpuLeaseFrame{};
        gpuUnpack.reset(); gpuImageRead.reset(); gpuStatistics.reset(); gpuPatch.reset(); gpuImageWrite.reset();
        reduce.reset(); fused.reset(); postproc.reset(); preproc.reset();
#endif
        for (auto& net : extraUp) net.reset();
        for (auto& net : extraDown) net.reset();
        upscale.reset(); downscale.reset(); network.reset();
#if NCNN_VULKAN
        // Model bytes outlive all Nets, including their destruction paths.
        for(auto& param:uploadParams)param.clear();
        uploadWeights.clear();
#endif
    }
    void releaseAllocators() noexcept {
#if NCNN_VULKAN
        if (device && blobAllocator) device->reclaim_blob_allocator(blobAllocator);
        if (device && stagingAllocator) device->reclaim_staging_allocator(stagingAllocator);
        blobAllocator = stagingAllocator = nullptr;
#endif
    }
    void configureNet(ncnn::Net& net, bool fp16Arithmetic = true) const {
        net.opt.num_threads = options.threads;
        net.opt.use_vulkan_compute = options.backend == Backend::Vulkan;
        // fp16 仅在显式开启时启用，并由 ncnn 按设备能力再与运算；默认保持 fp32 参考路径。
        net.opt.use_fp16_packed = fp16;
        net.opt.use_fp16_storage = fp16;
        net.opt.use_fp16_arithmetic = fp16 && fp16Arithmetic;
        net.opt.use_bf16_storage = false;
        net.opt.use_int8_inference = false;
#if NCNN_VULKAN
        if (options.backend == Backend::Vulkan) {
            net.set_vulkan_device(options.domainDevice);
#if defined(__ANDROID__)
            if (options.domainDevice) {
                const auto& gpu = options.domainDevice->info;
                // This Adreno 840 driver stalls in convolution_gemm_cm pipeline creation.
                if (gpu.vendor_id() == 0x5143 && gpu.device_id() == 0x44050a31 &&
                    gpu.driver_version() == 0x8034a034)
                    net.opt.use_cooperative_matrix = false;
            }
#endif
            net.opt.blob_vkallocator = blobAllocator;
            net.opt.workspace_vkallocator = blobAllocator;
            net.opt.staging_vkallocator = stagingAllocator;
        }
#endif
    }
};
#if NCNN_VULKAN
#include "shader_pipeline.inc"
#endif

const char* Status::name() const noexcept {
    switch (code) {
        case Code::Ok: return "ok";
        case Code::InvalidArgument: return "invalid_argument";
        case Code::UnsupportedBackend: return "unsupported_backend";
        case Code::GpuUnavailable: return "gpu_unavailable";
        case Code::ModelTooLarge: return "model_too_large";
        case Code::InvalidModel: return "invalid_model";
        case Code::ModelLoadFailed: return "model_load_failed";
        case Code::NotLoaded: return "not_loaded";
        case Code::Busy: return "busy";
        case Code::AllocationFailed: return "allocation_failed";
        case Code::InferenceFailed: return "inference_failed";
        case Code::InvalidOutput: return "invalid_output";
        case Code::ArtifactMismatch: return "artifact_mismatch";
        case Code::IncompatibleDomain: return "incompatible_domain";
        case Code::DeviceLost: return "device_lost";
    }
    return "unknown";
}

Engine::Engine(std::shared_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Engine::~Engine() {
#if NCNN_VULKAN
    if(impl_&&impl_->uploadOwner&& !close().ok())Impl::abandon(impl_);
#endif
}

Status Engine::pollUploads(std::uint64_t waitNs) noexcept {
    std::unique_lock<std::mutex> lifetime(impl_->lifecycleMutex,std::try_to_lock);
    if(!lifetime.owns_lock())return {Code::Busy};
#if NCNN_VULKAN
    if(waitNs>Impl::UploadWaitLimitNs)return {Code::InvalidArgument};
    if(impl_->uploadOwner) {
        const bool lost=impl_->observeUploadLoss();
        if(lost){mgc_domain_service(impl_->options.computeDomain,0);if(!meguviz::ncnn_domain_loss_retired(impl_->options.computeDomain))return {Code::Busy};}
        struct Call { Impl* impl;std::uint64_t wait; } call{impl_.get(),waitNs};
        const auto enter=lost?meguviz::try_with_ncnn_domain_owner_retire:
            waitNs?meguviz::with_ncnn_domain_owner:meguviz::try_with_ncnn_domain_owner;
        const auto status=enter(impl_->uploadOwner,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
            auto& c=*static_cast<Call*>(p);
            std::unique_lock<std::mutex> lock(c.impl->mutex,std::try_to_lock);
            if(!lock.owns_lock())return MGC_BUSY;
            const auto retirement=c.impl->retireUploads(c.wait);
            if(retirement!=MGC_SUCCESS&&retirement!=MGC_DEVICE_LOST)return retirement;
            return meguviz::ncnn_domain_owner_pending(c.impl->uploadOwner,false);
        },&call);
        return status==MGC_SUCCESS?Status{impl_->observeUploadLoss()?Code::DeviceLost:Code::Ok}:
            Status{status==MGC_DEVICE_LOST?Code::DeviceLost:Code::Busy};
    }
#else
    (void)waitNs;
#endif
    return {};
}

Status Engine::close() noexcept {
    std::unique_lock<std::mutex> lifetime(impl_->lifecycleMutex,std::try_to_lock);
    if(!lifetime.owns_lock())return {Code::Busy};
    std::unique_lock<std::mutex> lock(impl_->mutex,std::try_to_lock);
    if(!lock.owns_lock())return {Code::Busy};
#if NCNN_VULKAN
    // closeWithOwner uses the same device serialization as load, but must not
    // acquire it after the Engine mutex: load owns device then Engine. Let the
    // owner callback acquire Engine instead. The caller keeps this Engine live.
    if(impl_->uploadOwner) {
        lock.unlock();
        if(impl_->observeUploadLoss()){mgc_domain_service(impl_->options.computeDomain,0);if(!meguviz::ncnn_domain_loss_retired(impl_->options.computeDomain))return {Code::Busy};}
        struct Call { Impl* impl; } call{impl_.get()};
        const auto enter=meguviz::ncnn_domain_is_lost(impl_->options.computeDomain)?
            meguviz::try_with_ncnn_domain_owner_retire:meguviz::try_with_ncnn_domain_owner;
        const auto status=enter(impl_->uploadOwner,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
            auto* impl=static_cast<Call*>(p)->impl;
            std::unique_lock<std::mutex> guard(impl->mutex,std::try_to_lock);
            if(!guard.owns_lock())return MGC_BUSY;
            return impl->closeOwned(0);
        },&call);
        if(status!=MGC_SUCCESS)return {Code::Busy};
        const auto released=meguviz::ncnn_domain_owner_release(impl_->uploadOwner);
        if(released!=MGC_SUCCESS)return {Code::Busy};
        impl_->uploadOwner=nullptr;impl_->unregisterOwner();
        return {};
    }
#endif
    impl_->loaded.store(false);impl_->closed=true;impl_->releaseNets();impl_->releaseAllocators();
    mgc_shaderpack_release(impl_->options.shaderpack);impl_->options.shaderpack=nullptr;
    return {};
}

Status Engine::pollRetiredUploads(std::uint64_t waitNs, MgcDomain* domain) noexcept {
#if NCNN_VULKAN
    if(waitNs>Impl::UploadWaitLimitNs)return {Code::InvalidArgument};
    bool waiting=false;
    auto& registry=Impl::registry();
    for(std::size_t i=0;i<registry.slots.size();++i) {
        std::shared_ptr<Impl> impl;
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            auto& slot=registry.slots[i];
            if(slot.impl&&slot.abandoned&&(!domain||slot.impl->options.computeDomain==domain)) {
                if(slot.polling){waiting=true;continue;}
                slot.polling=true;impl=slot.orphan;
            }
        }
        if(!impl)continue;
        // Only this claimant may touch/free this abandoned implementation.
        // It has no live Engine caller and all access uses the device owner.
        const auto status=impl->closeWithOwner(waitNs);
        if(status!=MGC_SUCCESS) {
            waiting=true;
            std::lock_guard<std::mutex> lock(registry.mutex);
            if(registry.slots[i].impl==impl.get())registry.slots[i].polling=false;
        }
    }
    return waiting?Status{Code::Busy}:Status{};
#else
    (void)waitNs;(void)domain;return {};
#endif
}

std::unique_ptr<Engine> Engine::create(const Options& options, Status* status) noexcept {
    const auto fail = [status](Code code) -> std::unique_ptr<Engine> {
        if (status != nullptr) *status = {code};
        return nullptr;
    };
    try {
        if (options.threads < 1 || options.threads > 8 || options.maxPixels == 0 || options.maxPixels > MaxPixels)
            return fail(Code::InvalidArgument);
        if (options.backend != Backend::Cpu && options.backend != Backend::Vulkan) return fail(Code::UnsupportedBackend);
        if (options.backend == Backend::Vulkan) {
#if NCNN_VULKAN
            if(meguviz::ncnn_domain_is_lost(options.computeDomain))return fail(Code::DeviceLost);
#endif
            const Status gpu = gpuReady(options.domainDevice);
            if (!gpu.ok()) return fail(gpu.code);
        }
        auto engine = std::unique_ptr<Engine>(new Engine(std::make_shared<Impl>(options)));
#if NCNN_VULKAN
        const auto ownership=engine->impl_->initializeOwner();
        if(ownership!=MGC_SUCCESS)return fail(ownership==MGC_OUT_OF_MEMORY?Code::AllocationFailed:
            ownership==MGC_DEVICE_LOST?Code::DeviceLost:ownership==MGC_BACKPRESSURE||ownership==MGC_BUSY?Code::Busy:Code::IncompatibleDomain);
#endif
        if (status != nullptr) *status = {};
        return engine;
    } catch (const std::bad_alloc&) { return fail(Code::AllocationFailed); }
    catch (...) { return fail(Code::InferenceFailed); }
}

Status Engine::configure(bool fp16, int flowScale, bool gpuIo, bool protectStaticDetails, bool fusedWarp, bool rawGpuInput,
                         bool adaptiveFlow) noexcept {
    std::unique_lock<std::mutex> lifetime(impl_->lifecycleMutex,std::try_to_lock);
    if(!lifetime.owns_lock())return {Code::Busy};
#if NCNN_VULKAN
    if(impl_->options.backend==Backend::Vulkan) {
        if(meguviz::ncnn_domain_is_lost(impl_->options.computeDomain))return {Code::DeviceLost};
        if(!impl_->uploadOwner||impl_->closed||impl_->uploadsPending.load())return {Code::Busy};
        struct Call { Engine* self;bool fp16;int scale;bool gpu,detail,fused,raw,adaptive;Status result; }
            call{this,fp16,flowScale,gpuIo,protectStaticDetails,fusedWarp,rawGpuInput,adaptiveFlow,{Code::Busy}};
        const auto status=meguviz::with_ncnn_domain_owner(impl_->uploadOwner,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
            auto& c=*static_cast<Call*>(p);
            c.result=c.self->configureOwned(c.fp16,c.scale,c.gpu,c.detail,c.fused,c.raw,c.adaptive);
            if(!c.self->impl_->refreshPending())return meguviz::ncnn_domain_owner_pending(c.self->impl_->uploadOwner,false);
            return MGC_BUSY;
        },&call);
        return status==MGC_SUCCESS?call.result:Status{status==MGC_DEVICE_LOST?Code::DeviceLost:status==MGC_OUT_OF_MEMORY?Code::AllocationFailed:Code::Busy};
    }
#endif
    return configureOwned(fp16,flowScale,gpuIo,protectStaticDetails,fusedWarp,rawGpuInput,adaptiveFlow);
}

Status Engine::configureOwned(bool fp16, int flowScale, bool gpuIo, bool protectStaticDetails, bool fusedWarp, bool rawGpuInput,
                              bool adaptiveFlow) noexcept {
    std::unique_lock<std::mutex> guard(impl_->mutex, std::try_to_lock);
    if (!guard.owns_lock() || impl_->activeInference) return {Code::Busy};
#if NCNN_VULKAN
    if(impl_->options.backend==Backend::Vulkan&&meguviz::ncnn_domain_is_lost(impl_->options.computeDomain))return {Code::DeviceLost};
#endif
    if(impl_->closed)return {Code::Busy};
    if (flowScale != 1 && flowScale != 2 && flowScale != 4 && flowScale != 6 && flowScale != 8) return {Code::InvalidArgument};
    if (protectStaticDetails && !gpuIo) return {Code::InvalidArgument};
    if (fusedWarp && (!gpuIo || flowScale == 1)) return {Code::InvalidArgument};
    if (rawGpuInput && !fusedWarp) return {Code::InvalidArgument};
    if (adaptiveFlow && flowScale != 4) return {Code::InvalidArgument};
    if ((flowScale != 1 || gpuIo) && impl_->options.backend != Backend::Vulkan) return {Code::UnsupportedBackend};
    // 精度在 load 时固化进管线：改配置即撤销已加载模型，须重新 load。
    impl_->loaded.store(false);
    impl_->releaseNets();
    impl_->fp16 = fp16;
    impl_->flowScale = flowScale;
    impl_->gpuIo = gpuIo;
    impl_->protectStaticDetails = protectStaticDetails;
    impl_->fusedWarp = fusedWarp;
    impl_->rawGpuInput = rawGpuInput;
    impl_->adaptiveFlow = adaptiveFlow;
    return {};
}

int Engine::unreliablePermille() const noexcept { return impl_->unreliablePermille.load(); }

int Engine::patchedTiles() const noexcept { return impl_->patchedTiles.load(); }

int Engine::histogramPermille() const noexcept { return impl_->histogramPermille.load(); }

void Engine::setMotionGuard(bool enabled) noexcept { impl_->motionGuard = enabled; }

void Engine::timings(std::int64_t out[3]) const noexcept {
    for (int i = 0; i < 3; ++i) out[i] = impl_->timings[i].load();
}

Status Engine::load(ModelView model) noexcept {
    std::unique_lock<std::mutex> lifetime(impl_->lifecycleMutex,std::try_to_lock);
    if(!lifetime.owns_lock())return {Code::Busy};
#if NCNN_VULKAN
    if(impl_->options.backend==Backend::Vulkan) {
        if(meguviz::ncnn_domain_is_lost(impl_->options.computeDomain))return {Code::DeviceLost};
        if(!impl_->uploadOwner||impl_->closed||impl_->uploadsPending.load())return {Code::Busy};
        struct Call { Engine* self;ModelView model;Status result; } call{this,model,{Code::Busy}};
        const auto status=meguviz::with_ncnn_domain_owner(impl_->uploadOwner,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
            auto& c=*static_cast<Call*>(p);c.result=c.self->loadOwned(c.model);
            if(c.self->impl_->observeUploadLoss())return MGC_DEVICE_LOST;
            if(!c.self->impl_->refreshPending())return meguviz::ncnn_domain_owner_pending(c.self->impl_->uploadOwner,false);
            return MGC_BUSY;
        },&call);
        return status==MGC_SUCCESS?call.result:Status{status==MGC_DEVICE_LOST?Code::DeviceLost:status==MGC_OUT_OF_MEMORY?Code::AllocationFailed:Code::Busy};
    }
#endif
    return loadOwned(model);
}

Status Engine::loadOwned(ModelView model) noexcept {
    try {
        std::unique_lock<std::mutex> guard(impl_->mutex, std::try_to_lock);
        if (!guard.owns_lock() || impl_->activeInference) return {Code::Busy};
#if NCNN_VULKAN
        if(impl_->options.backend==Backend::Vulkan&&meguviz::ncnn_domain_is_lost(impl_->options.computeDomain))return {Code::DeviceLost};
#endif
        if(impl_->closed)return {Code::Busy};
        impl_->loaded.store(false);
        impl_->releaseNets();
        if (model.param == nullptr || model.weights == nullptr || model.paramSize == 0 || model.weightsSize == 0)
            return {Code::InvalidArgument};
        if (model.paramSize > MaxParamBytes || model.weightsSize > MaxWeightBytes) return {Code::ModelTooLarge};
        if (std::memchr(model.param, 0, model.paramSize) != nullptr) return {Code::InvalidModel};
        // load_param_mem needs a NUL terminator; the caller's buffer need not provide one.
        std::string param(reinterpret_cast<const char*>(model.param), model.paramSize);
        int magic = 0, layers = 0, blobs = 0;
        std::istringstream header(param);
        if (!(header >> magic >> layers >> blobs) || magic != 7767517 || layers <= 0 || layers > 1024 || blobs <= 0 || blobs > 8192)
            return {Code::InvalidModel};
        if(impl_->options.pruneUnusedTail) {
            if(impl_->flowScale==1)return {Code::InvalidArgument};
            param=pruneTail(param);
        }
        if(impl_->options.fuseAddLeaky)param=fuseAddLeaky(param);
#if NCNN_VULKAN
        if (impl_->options.backend == Backend::Vulkan && impl_->blobAllocator == nullptr) {
            impl_->device = impl_->options.domainDevice;
            if (impl_->device == nullptr) return {Code::GpuUnavailable};
            impl_->blobAllocator = impl_->device->acquire_blob_allocator();
            impl_->stagingAllocator = impl_->device->acquire_staging_allocator();
            if (impl_->blobAllocator == nullptr || impl_->stagingAllocator == nullptr) return {Code::AllocationFailed};
        }
#endif
        auto localNetwork = std::make_unique<ncnn::Net>();
        auto* network=localNetwork.get();
#if NCNN_VULKAN
        if(impl_->options.backend==Backend::Vulkan) {
            impl_->uploadParams[0]=param;
            impl_->uploadWeights.assign(model.weights,model.weights+model.weightsSize);
            impl_->network=std::move(localNetwork);
        }
#endif
        impl_->configureNet(*network);
        impl_->requestedNetworkOptions=network->opt;
        if (network->register_custom_layer("rife.Warp", detail::createWarp) != 0 ||
            network->register_custom_layer("rife.AddLeaky", detail::createAddLeaky) != 0 ||
            network->load_param_mem(param.c_str()) != 0 || !modelShapeValid(*network,impl_->options.pruneUnusedTail)) return {Code::InvalidModel};
        const auto* weightData=model.weights;
#if NCNN_VULKAN
        if(impl_->options.backend==Backend::Vulkan)weightData=impl_->uploadWeights.data();
#endif
        BoundedReader weights(weightData, model.weightsSize);
#if NCNN_VULKAN
        const int loaded = impl_->loadNetwork(*network,weights,0);
#else
        const int loaded = network->load_model(weights);
#endif
        if (loaded != 0 || weights.truncated()) return {Code::ModelLoadFailed};
#if NCNN_VULKAN
        // Legacy fused kernels express storage rounding with scalar/vector
        // half types. The packed-without-storage emulation is a distinct ABI;
        // reject it explicitly instead of interpreting uint packing as colour.
        if(impl_->gpuIo&&network->opt.use_fp16_packed&&!network->opt.use_fp16_storage)return {Code::UnsupportedBackend};
        if (impl_->options.backend == Backend::Vulkan) {
            if (!network->opt.use_vulkan_compute || network->vulkan_device() == nullptr) return {Code::GpuUnavailable};
            for (const auto* layer : network->layers()) {
                if (layer == nullptr || (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split"))
                    return {Code::UnsupportedBackend};
            }
        }
#endif
        for (const int scale : {impl_->flowScale, impl_->adaptiveFlow ? 6 : 1, impl_->adaptiveFlow ? 8 : 1}) {
            if (scale == 1) continue;
            auto localDown = std::make_unique<ncnn::Net>();
            auto localUp = std::make_unique<ncnn::Net>();
            auto* downscale=localDown.get();auto* upscale=localUp.get();
            // 下采样只做 Interp：fp16 运算下源坐标超过 2048 时间距为 2 像素，4K 输入会错位取样；改用 fp32 运算
            // （存储仍为 fp16），代价可忽略。
            impl_->configureNet(*downscale, false);
            impl_->configureNet(*upscale);
            const BoundedReader empty(nullptr, 0);
            const std::string downParam = downscaleParam(scale);
            const std::string upParam = upscaleParam(scale);
#if NCNN_VULKAN
            const std::size_t downSlot=scale==impl_->flowScale?1:scale==6?3:5,upSlot=downSlot+1;
            impl_->uploadParams[downSlot]=downParam;impl_->uploadParams[upSlot]=upParam;
            // The fixed parent owns both networks before either load may submit.
            (scale == impl_->flowScale ? impl_->downscale : impl_->extraDown[scale]) = std::move(localDown);
            (scale == impl_->flowScale ? impl_->upscale : impl_->extraUp[scale]) = std::move(localUp);
            if (downscale->load_param_mem(downParam.c_str()) != 0 || impl_->loadNetwork(*downscale,empty,downSlot) != 0 ||
                upscale->register_custom_layer("rife.Warp", detail::createWarp) != 0 ||
                upscale->load_param_mem(upParam.c_str()) != 0 || impl_->loadNetwork(*upscale,empty,upSlot) != 0)
                return {Code::ModelLoadFailed};
#else
            if (downscale->load_param_mem(downParam.c_str()) != 0 || downscale->load_model(empty) != 0 ||
                upscale->register_custom_layer("rife.Warp", detail::createWarp) != 0 ||
                upscale->load_param_mem(upParam.c_str()) != 0 || upscale->load_model(empty) != 0)
                return {Code::ModelLoadFailed};
            (scale == impl_->flowScale ? impl_->downscale : impl_->extraDown[scale]) = std::move(localDown);
            (scale == impl_->flowScale ? impl_->upscale : impl_->extraUp[scale]) = std::move(localUp);
#endif
        }
#if NCNN_VULKAN
        if (impl_->gpuIo) {
            struct Source {const char* data;int size;Impl::PipelineSlot* slot;int x,y,z;uint32_t kernel;const char* logical;std::vector<int> bindings;int pushCount;};
            const Source sources[] = {
                {frame_preproc_comp_data,sizeof(frame_preproc_comp_data)-1,&impl_->preproc,8,8,1,MGR_SHADER_PREPROC,"rife/preproc",{1,1},6},
                {frame_postproc_comp_data,sizeof(frame_postproc_comp_data)-1,&impl_->postproc,64,1,1,MGR_SHADER_POSTPROC,"rife/postproc",{1,1,1,1,1},7},
                {frame_reduce_comp_data,sizeof(frame_reduce_comp_data)-1,&impl_->reduce,8,8,1,MGR_SHADER_REDUCE,"rife/reduce",{1,1},7}};
            auto pipelineStatus=[](MgcStatus status)->Status {
                if(status==MGC_ARTIFACT_MISMATCH)return {Code::ArtifactMismatch};
                if(status==MGC_INCOMPATIBLE_DOMAIN)return {Code::IncompatibleDomain};
                if(status==MGC_OUT_OF_MEMORY)return {Code::AllocationFailed};
                if(status==MGC_DEVICE_LOST)return {Code::DeviceLost};
                return {Code::UnsupportedBackend};
            };
            for (const auto& source : sources) {
                if (source.slot == &impl_->reduce && !impl_->rawGpuInput) continue;
                auto compileOptions=network->opt;
                if(source.slot==&impl_->reduce && impl_->options.directReducedInput)compileOptions.use_fp16_arithmetic=false;
                const auto status=impl_->makePipeline(source.kernel,source.logical,source.data,source.size,compileOptions,
                    source.x,source.y,source.z,source.bindings,source.pushCount,*source.slot);
                if(status!=MGC_SUCCESS)return pipelineStatus(status);
            }
            const Source gpuSources[] = {
                {frame_gpu_unpack_comp_data,sizeof(frame_gpu_unpack_comp_data)-1,&impl_->gpuUnpack,64,1,1,MGR_SHADER_GPU_UNPACK,"rife/gpu-unpack",{1,1},6},
                {frame_gpu_image_read_comp_data,sizeof(frame_gpu_image_read_comp_data)-1,&impl_->gpuImageRead,64,1,1,MGR_SHADER_GPU_IMAGE_READ,"rife/gpu-image-read",{2,1},3},
                {frame_gpu_statistics_comp_data,sizeof(frame_gpu_statistics_comp_data)-1,&impl_->gpuStatistics,8,8,1,MGR_SHADER_GPU_STATISTICS,"rife/gpu-statistics",{1,1,1},6},
                {frame_gpu_patch_comp_data,sizeof(frame_gpu_patch_comp_data)-1,&impl_->gpuPatch,64,1,1,MGR_SHADER_GPU_PATCH,"rife/gpu-patch",{1,1,1,1},7},
                {frame_gpu_image_write_comp_data,sizeof(frame_gpu_image_write_comp_data)-1,&impl_->gpuImageWrite,8,8,1,MGR_SHADER_GPU_IMAGE_WRITE,"rife/gpu-image-write",{1,1,1,2},3}};
            for (const auto& source : gpuSources) {
                const auto status=impl_->makePipeline(source.kernel,source.logical,source.data,source.size,network->opt,
                    source.x,source.y,source.z,source.bindings,source.pushCount,*source.slot);
                if(status!=MGC_SUCCESS)return pipelineStatus(status);
            }
            if (impl_->fusedWarp) {
                const char* source = impl_->rawGpuInput ? frame_fused_rgb8_comp_data : frame_fused_comp_data;
                const int size = impl_->rawGpuInput ? sizeof(frame_fused_rgb8_comp_data)-1 : sizeof(frame_fused_comp_data)-1;
                const auto status=impl_->makePipeline(MGR_SHADER_FUSED,impl_->rawGpuInput?"rife/fused-rgb8":"rife/fused",source,size,
                    network->opt,64,1,1,{1,1,1,1,1,1},impl_->rawGpuInput?21:20,impl_->fused);
                if(status!=MGC_SUCCESS)return pipelineStatus(status);
            }
            for(const auto& binding:impl_->options.shaderBindings) {
                bool used=false;for(const auto& record:impl_->shaderRecords)used|=record.number("kernel")==binding.kernel;
                if(!used)return {Code::ArtifactMismatch};
            }
            impl_->catalogText=mgr_detail::dump(mgr_detail::J::O{{"schema_version",1},{"scope","RIFE owner kernels; builtin ncnn and custom Warp/AddLeaky remain runtime compiled"},
                {"target_facts_scope","captured owner-used device macros only"},{"device_qualified",false},
                {"runtime_compiler_compatibility",true},{"private_sdk_fingerprint",meguviz::private_sdk_fingerprint()},
                {"variants",impl_->shaderRecords}});
        }
#endif
        if(localNetwork)impl_->network=std::move(localNetwork);
        impl_->graphText = std::move(param);
        using mgr_detail::J;
        J catalog=impl_->catalogText.empty()?J(J::O{{"schema_version",1},
            {"scope","execution graphs; GPU target facts exist only for actual Vulkan owner kernels"},
            {"device_qualified",false},{"runtime_compiler_compatibility",true},
            {"private_sdk_fingerprint",meguviz::private_sdk_fingerprint()},{"variants",J::A{}}}):
            mgr_detail::Parser(impl_->catalogText).parse();
        J::A graphs;
        J executionGraphs=J::O{{"main",J()},{"downscale",J()},{"upscale",J()},
            {"downscale_6",J()},{"upscale_6",J()},{"downscale_8",J()},{"upscale_8",J()}};
        auto graph=[&](const char* name,int scale,const ncnn::Net* net,bool down) {
            if(!net)return;auto requested=impl_->requestedNetworkOptions;if(down)requested.use_fp16_arithmetic=false;
            const J effective=graphOptions(net->opt);
            graphs.push_back(J::O{{"name",name},{"flow_scale",scale},{"requested_options",graphOptions(requested)},
                {"effective_options",effective},{"effective_options_hash",mgr_detail::sha256(mgr_detail::dump(effective))},
                {"readback_provenance","Net.opt after pinned ncnn load_param/load_model sanitization"}});
            const std::string key=scale!=impl_->flowScale?std::string(name)+"_"+std::to_string(scale):name;
            executionGraphs[key]=executionGraphOptions(net->opt);
        };
        graph("main",impl_->flowScale,impl_->network.get(),false);
        graph("downscale",impl_->flowScale,impl_->downscale.get(),true);
        graph("upscale",impl_->flowScale,impl_->upscale.get(),false);
        for(int scale=1;scale<=8;scale++){graph("downscale",scale,impl_->extraDown[scale].get(),true);graph("upscale",scale,impl_->extraUp[scale].get(),false);}
        catalog["graphs"]=graphs;catalog["execution_graphs"]=executionGraphs;impl_->catalogText=mgr_detail::dump(catalog);
        J::O execution={{"schema_version",1},{"effective_options_observed",true},{"graphs",executionGraphs},
            {"frame_kind",0},{"stage_ns",J::A{0,0,0}}};
#if defined(MGR_ENABLE_JOB_OBSERVATION)
        const J activeConfiguration=J::O{{"backend",impl_->options.backend==Backend::Vulkan?"vulkan":"cpu"},
            {"flow_scale",impl_->flowScale},{"gpu_io",impl_->gpuIo},{"fused_warp",impl_->fusedWarp},
            {"raw_gpu_input",impl_->rawGpuInput},{"adaptive_flow",impl_->adaptiveFlow},
            {"static_detail_guard_configured",impl_->protectStaticDetails},{"motion_guard_configured",bool(impl_->motionGuard)}};
        execution["active_configuration_observed"]=true;execution["active_configuration"]=activeConfiguration;
#endif
        impl_->executionText=mgr_detail::dump(execution);
        impl_->loaded.store(true);
        return {};
    } catch (const std::bad_alloc&) { return {Code::AllocationFailed}; }
    catch (...) { return {Code::ModelLoadFailed}; }
}

Status Engine::processRgb8(ImageView left, ImageView right, double timestep, MutableImageView output,
                           std::int64_t leftId, std::int64_t rightId, int flowScale, std::shared_ptr<void>* continuation) noexcept {
#if NCNN_VULKAN
    if(impl_->options.backend==Backend::Vulkan&&meguviz::ncnn_domain_is_lost(impl_->options.computeDomain))return {Code::DeviceLost};
    if(impl_->options.backend==Backend::Vulkan&&!mgr_detail::alpha_endpoint(mgr_detail::classify_alpha(timestep)))
        return processRgb8GpuOwned(left,right,timestep,output,leftId,rightId,flowScale,continuation);
#else
    (void)continuation;
#endif
    try {
        std::unique_lock<std::mutex> guard(impl_->mutex, std::try_to_lock);
        if (!guard.owns_lock() || impl_->activeInference) return {Code::Busy};
        MGR_OBSERVE_JOB(impl_->diagnosticLast=JobObservation{};impl_->diagnosticLast.started=true;)
        if (!impl_->loaded.load() || !impl_->network) return {Code::NotLoaded};
        const auto alphaClass = mgr_detail::classify_alpha(timestep);
        if (alphaClass == mgr_detail::Alpha::Invalid ||
            left.width != right.width || left.height != right.height || left.width != output.width || left.height != output.height ||
            !layoutValid(left.data, left.width, left.height, left.strideBytes, left.byteCapacity, impl_->options.maxPixels) ||
            !layoutValid(right.data, right.width, right.height, right.strideBytes, right.byteCapacity, impl_->options.maxPixels) ||
            !layoutValid(output.data, output.width, output.height, output.strideBytes, output.byteCapacity, impl_->options.maxPixels))
            return {Code::InvalidArgument};
        // The public API accepts a double. Only exact double endpoints may
        // bypass inference; model arithmetic remains the legacy FP32 value.
        const float modelTimestep=static_cast<float>(timestep);
        const auto rowBytes = static_cast<std::size_t>(left.width) * 3u;
        MGR_OBSERVE_JOB(impl_->diagnosticLast.endpoint=mgr_detail::alpha_endpoint(alphaClass);
        if(impl_->diagnosticLast.endpoint||!impl_->gpuIo) {
            impl_->diagnosticLast.tensorCache[0]=CacheObservation::NotApplicable;
            impl_->diagnosticLast.tensorCache[1]=CacheObservation::NotApplicable;
            impl_->diagnosticLast.staticGuard=GuardObservation::NotApplicable;
            impl_->diagnosticLast.motionGuard=GuardObservation::NotApplicable;
        })
        impl_->unreliablePermille.store(0); impl_->patchedTiles.store(0); impl_->histogramPermille.store(0);
        for (auto& timing : impl_->timings) timing.store(0);
        if (mgr_detail::alpha_endpoint(alphaClass) || !impl_->gpuIo)
            impl_->staging.resize(rowBytes * static_cast<std::size_t>(left.height));
        if (mgr_detail::alpha_endpoint(alphaClass)) {
            const auto image = alphaClass == mgr_detail::Alpha::Zero ? left : right;
            for (int y = 0; y < image.height; ++y)
                std::memcpy(impl_->staging.data() + static_cast<std::size_t>(y) * rowBytes,
                            image.data + static_cast<std::size_t>(y) * image.strideBytes, rowBytes);
        } else {
            const auto began = nowNs();
            impl_->unreliablePermille.store(0);
            impl_->patchedTiles.store(0);
            impl_->histogramPermille.store(0);
            // 全分辨率补到 32×scale 的倍数，低分辨率恰为 32 的倍数，上下采样严格 scale 倍。
            // 光流比例：调用方显式指定（仅自适应模式，须为已载入的 4/6/8），否则按像素数自动选择。
            int scale = impl_->flowScale;
            if (impl_->adaptiveFlow)
                scale = flowScale > 0 ? flowScale
                    : static_cast<std::int64_t>(left.width) * left.height > AdaptiveFlowPixels ? 8 : impl_->flowScale;
            else if (flowScale > 0 && flowScale != impl_->flowScale) return {Code::InvalidArgument};
            MGR_OBSERVE_JOB(impl_->diagnosticLast.effectiveScale=scale;impl_->diagnosticLast.cacheLookupScale=impl_->gpuIo?scale:0;)
            const bool extra = scale != impl_->flowScale;
            if (extra && (scale < 0 || scale > 8 || !impl_->extraDown[scale])) return {Code::InvalidArgument};
            const int align = 32 * scale;
            const int paddedWidth = (left.width + align - 1) / align * align;
            const int paddedHeight = (left.height + align - 1) / align * align;
            ncnn::Mat time(paddedWidth / scale, paddedHeight / scale, 1);
            if (time.empty()) return {Code::AllocationFailed};
            time.fill(modelTimestep);
            ncnn::Mat first, second;
            if (!impl_->gpuIo) {
                first = inputTensor(left, paddedWidth, paddedHeight);
                second = inputTensor(right, paddedWidth, paddedHeight);
                if (first.empty() || second.empty()) return {Code::AllocationFailed};
            }
            auto prepared = nowNs();
            auto inferred = prepared;
            ncnn::Mat result;
                ncnn::Extractor extractor = impl_->network->create_extractor();
                if (extractor.input("in0", first) != 0 || extractor.input("in1", second) != 0 ||
                    extractor.input("in2", time) != 0 || extractor.extract("out0", result) != 0)
                    return {Code::InferenceFailed};
            if (!impl_->gpuIo) {
                inferred = nowNs();
                if (result.empty() || result.dims != 3 || result.w != paddedWidth || result.h != paddedHeight ||
                    result.c != 3 || result.elempack != 1 || result.elemsize != sizeof(float)) return {Code::InvalidOutput};
                // 裁掉补边后整平面检查有限值，再用 ncnn 的 NEON 路径缩放、饱和取整并打包为 RGB8。
                ncnn::Mat visible = result;
                if (paddedWidth != left.width || paddedHeight != left.height)
                    ncnn::copy_cut_border(result, visible, 0, paddedHeight - left.height, 0, paddedWidth - left.width);
                if (visible.empty() || visible.w != left.width || visible.h != left.height) return {Code::InvalidOutput};
                const auto planeSize = static_cast<std::size_t>(left.width) * static_cast<std::size_t>(left.height);
                for (int c = 0; c < 3; ++c) {
                    const float* plane = visible.channel(c);
                    bool finite = true;
                    for (std::size_t i = 0; i < planeSize; ++i) finite &= std::isfinite(plane[i]);
                    if (!finite) return {Code::InvalidOutput};
                }
                const float factor[3] = {255.f, 255.f, 255.f};
                visible.substract_mean_normalize(nullptr, factor);
                visible.to_pixels(impl_->staging.data(), ncnn::Mat::PIXEL_RGB, static_cast<int>(rowBytes));
            }
            impl_->timings[0].store(prepared - began);
            impl_->timings[1].store(inferred - prepared);
            impl_->timings[2].store(nowNs() - inferred);
        }
        // Commit only a complete valid result; error paths preserve output, including aliased inputs.
        for (int y = 0; y < output.height; ++y)
            std::memcpy(output.data + static_cast<std::size_t>(y) * output.strideBytes,
                        impl_->staging.data() + static_cast<std::size_t>(y) * rowBytes, rowBytes);
        MGR_OBSERVE_JOB(impl_->diagnosticLast.completed=true;)
        return {};
    } catch (const std::bad_alloc&) { return {Code::AllocationFailed}; }
    catch (...) { return {Code::InferenceFailed}; }
}

Backend Engine::backend() const noexcept { return impl_->options.backend; }
bool Engine::ready() const noexcept {
#if NCNN_VULKAN
    if(impl_->options.backend==Backend::Vulkan&&meguviz::ncnn_domain_is_lost(impl_->options.computeDomain))return false;
#endif
    return impl_->loaded.load();
}
std::string Engine::effectiveOptions() const {
    if(!impl_->network)return {};
    const auto& o=impl_->network->opt;
    return std::string("ncnn=7e36d25cb153092ba87a60530455c7376fedf452;vulkan=")+std::to_string(o.use_vulkan_compute)+
        ";threads="+std::to_string(o.num_threads)+";fp16_packed="+std::to_string(o.use_fp16_packed)+
        ";fp16_storage="+std::to_string(o.use_fp16_storage)+";fp16_arithmetic="+std::to_string(o.use_fp16_arithmetic)+
        ";bf16_storage="+std::to_string(o.use_bf16_storage)+";int8_inference="+std::to_string(o.use_int8_inference)+
        ";packing="+std::to_string(o.use_packing_layout)+";downscale_fp16_arithmetic=0;direct_reduce_fp16_arithmetic="+
        std::to_string(impl_->options.directReducedInput?false:o.use_fp16_arithmetic)+
        ";gpu_io="+std::to_string(impl_->gpuIo)+";fused="+std::to_string(impl_->fusedWarp)+
        ";raw="+std::to_string(impl_->rawGpuInput)+";adaptive="+std::to_string(impl_->adaptiveFlow)+
        ";static="+std::to_string(impl_->protectStaticDetails)+";motion="+std::to_string(impl_->motionGuard)+
        ";flow_scale="+std::to_string(impl_->flowScale)+";final_coordinates=fp32;remaining_options=pinned_ncnn_defaults;shader_catalog_sha256="+
        mgr_detail::sha256(impl_->catalogText);
}
const std::string& Engine::runtimeGraph() const noexcept {return impl_->graphText;}
const std::string& Engine::shaderCatalog() const noexcept {return impl_->catalogText;}
const std::string& Engine::executionInfo() const noexcept {return impl_->executionText;}
#if defined(MGR_ENABLE_JOB_OBSERVATION)
JobObservation Engine::diagnosticJob() const {std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->diagnosticLast;}
#endif
void Engine::shaderCounts(uint32_t out[3]) const noexcept {std::copy(impl_->shaderCounters,impl_->shaderCounters+3,out);}
#include "frame_gpu.inc"
#include "frame_rgb_owned.inc"
#undef MGR_OBSERVE_JOB
} // namespace meguviz::rife::legacy
