// Actual compatibility recorder with task-owned temporaries. Numeric operations
// and Options match processRgb8's old GPU path; only submission/lifetime changes.
Status Engine::processRgb8GpuOwned(ImageView inputLeft,ImageView inputRight,double timestep,MutableImageView target,
                                  int64_t inputLeftId,int64_t inputRightId,int requestedScale,std::shared_ptr<void>* continuation) noexcept {
#if !NCNN_VULKAN
    (void)inputLeft;(void)inputRight;(void)timestep;(void)target;(void)inputLeftId;(void)inputRightId;(void)requestedScale;(void)continuation;
    return {Code::UnsupportedBackend};
#else
    if(meguviz::ncnn_domain_is_lost(impl_->options.computeDomain))return {Code::DeviceLost};
    struct Work {
        std::shared_ptr<Impl> owner;Impl* engine=nullptr;
        int stage=0;bool initialized=false,callbackEntered=false,motionGuard=false;
        Code recordError=Code::InferenceFailed;
        ImageView left{},right{};MutableImageView output{};
        int64_t leftId=-1,rightId=-1,began=0,prepared=0,inferred=0;
        int scale=1,paddedWidth=0,paddedHeight=0;size_t rowBytes=0;float modelTimestep=0;
        bool commandPath=false;
        ncnn::Net* downNet=nullptr; ncnn::Net* upNet=nullptr;
        ncnn::Mat time,first,second,result,bytes,flagged,downloaded,unpacked;
        ncnn::VkMat full0,full1,flowInput0,flowInput1,merged,flow,mask,planar,packed,flag;
        std::array<ncnn::VkMat,2> rawGpuKeep;
        std::vector<ncnn::Mat> keepAlive;
        std::vector<ncnn::Extractor> downs;
        std::optional<ncnn::Extractor> up,estimate;
        std::vector<Impl::GpuFrame> fresh;
        MGR_OBSERVE_JOB(JobObservation observation{};)
        ~Work(){
            if(!owner)return;
            meguviz::with_ncnn_domain_cleanup(owner->options.computeDomain,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
                auto& w=*static_cast<Work*>(p);std::lock_guard<std::mutex> lock(w.owner->mutex);
                w.estimate.reset();w.up.reset();w.downs.clear();w.fresh.clear();
                w.full0.release();w.full1.release();w.flowInput0.release();w.flowInput1.release();w.merged.release();w.flow.release();w.mask.release();
                w.planar.release();w.packed.release();w.flag.release();for(auto& mat:w.rawGpuKeep)mat.release();
                w.time.release();w.first.release();w.second.release();w.result.release();w.bytes.release();w.flagged.release();w.downloaded.release();w.unpacked.release();w.keepAlive.clear();
                if(w.owner->activeInference==&w)w.owner->activeInference=nullptr;
                return MGC_SUCCESS;
            },this);
        }
    };
    std::shared_ptr<Work> state;
    try {
        std::unique_lock<std::mutex> lock(impl_->mutex,std::try_to_lock);if(!lock.owns_lock())return {Code::Busy};
        state=continuation&&*continuation?std::static_pointer_cast<Work>(*continuation):std::make_shared<Work>();
        auto& w=*state;
        if(w.owner&&w.owner!=impl_)return {Code::IncompatibleDomain};
        if(impl_->activeInference&&impl_->activeInference!=&w)return {Code::Busy};
        w.owner=impl_;w.engine=impl_.get();impl_->activeInference=&w;if(continuation)*continuation=state;
        if(w.stage==2)return {};
        if(!w.initialized){
            if(!impl_->loaded.load()||!impl_->network)return {Code::NotLoaded};
            if(mgr_detail::classify_alpha(timestep)!=mgr_detail::Alpha::Interior||
               inputLeft.width!=inputRight.width||inputLeft.height!=inputRight.height||inputLeft.width!=target.width||inputLeft.height!=target.height||
               !layoutValid(inputLeft.data,inputLeft.width,inputLeft.height,inputLeft.strideBytes,inputLeft.byteCapacity,impl_->options.maxPixels)||
               !layoutValid(inputRight.data,inputRight.width,inputRight.height,inputRight.strideBytes,inputRight.byteCapacity,impl_->options.maxPixels)||
               !layoutValid(target.data,target.width,target.height,target.strideBytes,target.byteCapacity,impl_->options.maxPixels))return {Code::InvalidArgument};
            w.left=inputLeft;w.right=inputRight;w.output=target;w.leftId=inputLeftId;w.rightId=inputRightId;w.modelTimestep=static_cast<float>(timestep);
            w.rowBytes=size_t(inputLeft.width)*3;w.motionGuard=impl_->motionGuard;
            if(!impl_->gpuIo)impl_->staging.resize(w.rowBytes*size_t(inputLeft.height));
            w.began=nowNs();w.scale=impl_->flowScale;
            if(impl_->adaptiveFlow)w.scale=requestedScale>0?requestedScale:int64_t(inputLeft.width)*inputLeft.height>AdaptiveFlowPixels?8:impl_->flowScale;
            else if(requestedScale>0&&requestedScale!=w.scale)return {Code::InvalidArgument};
            const bool extra=w.scale!=impl_->flowScale;
            if(extra&&(w.scale<0||w.scale>8||!impl_->extraDown[w.scale]))return {Code::InvalidArgument};
            w.downNet=extra?impl_->extraDown[w.scale].get():impl_->downscale.get();w.upNet=extra?impl_->extraUp[w.scale].get():impl_->upscale.get();
            const int align=32*w.scale;w.paddedWidth=(inputLeft.width+align-1)/align*align;w.paddedHeight=(inputLeft.height+align-1)/align*align;
            w.time.create(w.paddedWidth/w.scale,w.paddedHeight/w.scale,1);if(w.time.empty())return {Code::AllocationFailed};w.time.fill(w.modelTimestep);
            w.commandPath=w.scale!=1||impl_->gpuIo;
            if(!impl_->gpuIo){w.first=inputTensor(inputLeft,w.paddedWidth,w.paddedHeight);w.second=inputTensor(inputRight,w.paddedWidth,w.paddedHeight);if(w.first.empty()||w.second.empty())return {Code::AllocationFailed};}
            MGR_OBSERVE_JOB(w.observation=JobObservation{};w.observation.started=true;w.observation.effectiveScale=w.scale;w.observation.cacheLookupScale=impl_->gpuIo?w.scale:0;
                if(!impl_->gpuIo){w.observation.tensorCache[0]=CacheObservation::NotApplicable;w.observation.tensorCache[1]=CacheObservation::NotApplicable;w.observation.staticGuard=GuardObservation::NotApplicable;w.observation.motionGuard=GuardObservation::NotApplicable;})
            impl_->unreliablePermille.store(0);impl_->patchedTiles.store(0);impl_->histogramPermille.store(0);for(auto& value:impl_->timings)value.store(0);
            w.prepared=nowNs();w.initialized=true;
        }
        auto* impl_=w.engine;
        auto& left=w.left;auto& right=w.right;auto& output=w.output;
        const auto leftId=w.leftId,rightId=w.rightId;const auto scale=w.scale;
        const auto paddedWidth=w.paddedWidth,paddedHeight=w.paddedHeight;
        const auto modelTimestep=w.modelTimestep;const auto rowBytes=w.rowBytes;
        auto& first=w.first;auto& second=w.second;auto& time=w.time;auto& result=w.result;
        auto& full0=w.full0;auto& full1=w.full1;auto& flowInput0=w.flowInput0;auto& flowInput1=w.flowInput1;
        auto& merged=w.merged;auto& flow=w.flow;auto& mask=w.mask;auto& planar=w.planar;auto& packed=w.packed;auto& flag=w.flag;
        auto& rawGpuKeep=w.rawGpuKeep;auto& keepAlive=w.keepAlive;auto& downs=w.downs;auto& up=w.up;auto& fresh=w.fresh;
        auto& bytes=w.bytes;auto& flagged=w.flagged;auto& downloaded=w.downloaded;auto& unpacked=w.unpacked;
        auto& began=w.began;auto& prepared=w.prepared;auto& inferred=w.inferred;
        const auto& opt=impl_->network->opt;
        if(w.stage==0){
            w.stage=1;w.callbackEntered=false;
            lock.unlock();
            const auto status=meguviz::with_ncnn_gpu_owned(impl_->options.computeDomain,nullptr,0,state,[](void* context,meguviz::NcnnGpuScope& scope) -> MgcStatus {
                auto& w=*static_cast<Work*>(context);std::lock_guard<std::mutex> guard(w.engine->mutex);
                if(w.engine->activeInference!=&w)return MGC_BUSY;
                w.callbackEntered=true;
        auto* impl_=w.engine;
        auto& left=w.left;auto& right=w.right;auto& output=w.output;
        const auto leftId=w.leftId,rightId=w.rightId;const auto scale=w.scale;
        const auto paddedWidth=w.paddedWidth,paddedHeight=w.paddedHeight;
        const auto modelTimestep=w.modelTimestep;const auto rowBytes=w.rowBytes;
        auto& first=w.first;auto& second=w.second;auto& time=w.time;auto& result=w.result;
        auto& full0=w.full0;auto& full1=w.full1;auto& flowInput0=w.flowInput0;auto& flowInput1=w.flowInput1;
        auto& merged=w.merged;auto& flow=w.flow;auto& mask=w.mask;auto& planar=w.planar;auto& packed=w.packed;auto& flag=w.flag;
        auto& rawGpuKeep=w.rawGpuKeep;auto& keepAlive=w.keepAlive;auto& downs=w.downs;auto& up=w.up;auto& fresh=w.fresh;
        auto& bytes=w.bytes;auto& flagged=w.flagged;auto& downloaded=w.downloaded;auto& unpacked=w.unpacked;
        auto& began=w.began;auto& prepared=w.prepared;auto& inferred=w.inferred;
                const auto& opt=impl_->network->opt;auto& command=*scope.command;
                auto* downNet=w.downNet;auto* upNet=w.upNet;
                if(!w.commandPath){
                    w.estimate.emplace(impl_->network->create_extractor());auto& estimate=*w.estimate;
                    if(estimate.input("in0",first)||estimate.input("in1",second)||estimate.input("in2",time)||estimate.record_extract_host("out0",result,command))return MGC_INTERNAL_ERROR;
                    return MGC_SUCCESS;
                }
                if (impl_->gpuIo) {
                    // 只上传原始 RGB8 字节；解包、归一化与补零在 GPU 完成。缓存命中的帧完全不上传。
                    const std::size_t elem = opt.use_fp16_storage ? 2u : 4u;
                    for (const auto& [image, id, target, reduced] : {std::tuple{left, leftId, &full0, &flowInput0},
                                                                     std::tuple{right, rightId, &full1, &flowInput1}}) {
                        const Impl::GpuFrame* hit = nullptr;
                        for (const auto& frame : impl_->frames)
                            if (id >= 0 && frame.id == id && frame.scale == scale && frame.width == paddedWidth && frame.height == paddedHeight &&
                                frame.sourceWidth == image.width && frame.sourceHeight == image.height &&
                                frame.sourceStride == image.strideBytes) hit = &frame;
                        MGR_OBSERVE_JOB(const int observedInput=target==&full0?0:1;
                        w.observation.tensorCache[observedInput]=hit?CacheObservation::Hit:
                            (id<0?CacheObservation::Disabled:CacheObservation::Miss);)
                        if (hit != nullptr) { *target = hit->full; *reduced = hit->reduced; continue; }
                        const std::size_t bytes = (static_cast<std::size_t>(image.height) - 1) * image.strideBytes +
                                                  static_cast<std::size_t>(image.width) * 3u;
                        const int words = static_cast<int>((bytes + 3) / 4);
                        ncnn::Mat raw;
                        if (bytes % 4 == 0 && reinterpret_cast<std::uintptr_t>(image.data) % 4 == 0) {
                            raw = ncnn::Mat(words, const_cast<std::uint8_t*>(image.data), static_cast<std::size_t>(4u));
                        } else {
                            raw.create(words, static_cast<std::size_t>(4u));
                            if (raw.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                            static_cast<std::uint32_t*>(raw.data)[words - 1] = 0;
                            std::memcpy(raw.data, image.data, bytes);
                        }
                        keepAlive.push_back(raw);
                        ncnn::VkMat rawGpu;
                        command.record_clone(raw, rawGpu, opt);
                        rawGpuKeep[target == &full0 ? 0 : 1] = rawGpu;
                        if (impl_->rawGpuInput) {
                            *target = rawGpu;
                            reduced->create(paddedWidth/scale, paddedHeight/scale, 3, elem, 1, impl_->blobAllocator);
                            if (rawGpu.empty() || reduced->empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                            std::vector<ncnn::vk_constant_type> constants(7);
                            constants[0].i=image.width; constants[1].i=image.height;
                            constants[2].i=static_cast<int>(image.strideBytes);
                            constants[3].i=reduced->w; constants[4].i=reduced->h;
                            constants[5].i=static_cast<int>(reduced->cstep); constants[6].i=scale;
                            command.record_pipeline(impl_->reduce.get(), {rawGpu,*reduced}, constants, *reduced);
                        } else {
                            target->create(paddedWidth, paddedHeight, 3, elem, 1, impl_->blobAllocator);
                            if (rawGpu.empty() || target->empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                            std::vector<ncnn::vk_constant_type> constants(6);
                            constants[0].i = image.width;
                            constants[1].i = image.height;
                            constants[2].i = static_cast<int>(image.strideBytes);
                            constants[3].i = target->w;
                            constants[4].i = target->h;
                            constants[5].i = static_cast<int>(target->cstep);
                            command.record_pipeline(impl_->preproc.get(), {rawGpu, *target}, constants, *target);
                            *reduced = *target;
                            if (scale != 1) {
                                downs.push_back(downNet->create_extractor());
                                downs.back().set_light_mode(false);
                                if (downs.back().input("in0", *target) != 0 || downs.back().extract("half0", *reduced, command) != 0)
                                    {w.recordError=Code::InferenceFailed;return MGC_INTERNAL_ERROR;}
                            }
                        }
                        if (id >= 0) fresh.push_back({id, paddedWidth, paddedHeight, *target, *reduced,
                            image.width, image.height, image.strideBytes, scale});
                    }
                } else {
                    command.record_upload(first, full0, opt);
                    if (full0.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                    command.record_upload(second, full1, opt);
                    if (full1.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                    flowInput0 = full0;
                    flowInput1 = full1;
                    if (scale != 1) {
                        downs.push_back(downNet->create_extractor());
                        auto& down = downs.back();
                        down.set_light_mode(false);
                        if (down.input("in0", full0) != 0 || down.input("in1", full1) != 0 ||
                            down.extract("half0", flowInput0, command) != 0 || down.extract("half1", flowInput1, command) != 0)
                            {w.recordError=Code::InferenceFailed;return MGC_INTERNAL_ERROR;}
                    }
                }
                prepared = nowNs();
                w.estimate.emplace(impl_->network->create_extractor());auto& estimate=*w.estimate;
                // Vulkan records resource lifetimes/barriers; extracted flow/mask keep their own references.
                // Reclaim dead internal activations instead of retaining the whole graph until completion.
                estimate.set_light_mode(impl_->fusedWarp);
                if (estimate.input("in0", flowInput0) != 0 || estimate.input("in1", flowInput1) != 0 ||
                    estimate.input("in2", time) != 0) {w.recordError=Code::InferenceFailed;return MGC_INTERNAL_ERROR;}
                if (scale == 1) {
                    if (estimate.extract("out0", merged, command) != 0) {w.recordError=Code::InferenceFailed;return MGC_INTERNAL_ERROR;}
                } else {
                    if (estimate.extract(FlowBlob, flow, command) != 0 || estimate.extract(MaskBlob, mask, command) != 0)
                        {w.recordError=Code::InferenceFailed;return MGC_INTERNAL_ERROR;}
                    if (!impl_->fusedWarp) {
                        up.emplace(upNet->create_extractor());
                        if (up->input("in0", full0) != 0 || up->input("in1", full1) != 0 || up->input("flow", flow) != 0 ||
                            up->input("mask", mask) != 0 || up->extract("out0", merged, command) != 0)
                            {w.recordError=Code::InferenceFailed;return MGC_INTERNAL_ERROR;}
                    }
                }
                if (!impl_->fusedWarp && (merged.dims != 3 || merged.w != paddedWidth || merged.h != paddedHeight || merged.c * merged.elempack != 3))
                    {w.recordError=Code::InvalidOutput;return MGC_INTERNAL_ERROR;}
                if (impl_->gpuIo) {
                    // GPU 上缩放、饱和取整并紧凑打包为 RGB8，同时标记任何非有限值；只下载字节与标志。
                    planar = merged;
                    if (!impl_->fusedWarp && merged.elempack != 1) impl_->device->convert_packing(merged, planar, 1, command, opt);
                    if (!impl_->fusedWarp && planar.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                    if (!impl_->fusedWarp && (planar.dims != 3 || planar.w != paddedWidth || planar.h != paddedHeight || planar.c != 3 || planar.elempack != 1))
                        {w.recordError=Code::InvalidOutput;return MGC_INTERNAL_ERROR;}
                    const int words = static_cast<int>((rowBytes * static_cast<std::size_t>(left.height) + 3) / 4);

                    packed.create(words, static_cast<std::size_t>(4u), 1, impl_->blobAllocator);
                    // [0] 非有限值标志，[1] 运动置信度不足的抽样像素数，[2..] 同一计数按 16×9 分块（仅融合路径）。
                    ncnn::Mat zero(2 + GuardTiles, static_cast<std::size_t>(4u));
                    if (packed.empty() || zero.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                    std::memset(zero.data, 0, (2 + GuardTiles) * sizeof(std::uint32_t));
                    keepAlive.push_back(zero);
                    command.record_clone(zero, flag, opt);
                    if (flag.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                    std::vector<ncnn::vk_constant_type> constants(impl_->fusedWarp ? (impl_->rawGpuInput ? 21 : 20) : 7);
                    constants[0].i = left.width;
                    constants[1].i = left.height;
                    constants[2].i = words;
                    constants[3].i = paddedWidth;
                    if (impl_->fusedWarp) {
                        if (flow.w != paddedWidth/scale || flow.h != paddedHeight/scale || flow.c*flow.elempack != 4 ||
                            mask.w != flow.w || mask.h != flow.h || mask.c*mask.elempack != 1 ||
                            (flow.elempack != 1 && flow.elempack != 4)) {w.recordError=Code::InvalidOutput;return MGC_INTERNAL_ERROR;}
                        constants[4].i = static_cast<int>(impl_->rawGpuInput ? left.strideBytes : full0.cstep);
                        constants[5].i = impl_->protectStaticDetails ? 1 : 0;
                        MGR_OBSERVE_JOB(w.observation.staticGuard=impl_->protectStaticDetails?GuardObservation::Executed:GuardObservation::Disabled;
                        w.observation.motionGuard=w.motionGuard?GuardObservation::Executed:GuardObservation::Disabled;)
                        constants[6].i = paddedHeight;
                        constants[7].i = flow.w; constants[8].i = flow.h;
                        constants[9].i = static_cast<int>(flow.cstep);
                        constants[10].i = flow.elempack; constants[11].i = scale;
                        if (impl_->rawGpuInput) constants[12].i = static_cast<int>(right.strideBytes);
                        const int extra = impl_->rawGpuInput ? 13 : 12;
                        constants[extra].f = modelTimestep;
                        constants[extra + 1].i = w.motionGuard ? 1 : 0;
                        // 整帧亮度变化（闪白、淡入淡出）：抽样通道均值的偏移与增益，供置信度比较时扣除。
                        double meanL[3] = {0, 0, 0}, meanR[3] = {0, 0, 0};
                        sampleMeans(left, meanL);
                        sampleMeans(right, meanR);
                        for (int c = 0; c < 3; ++c) {
                            constants[extra + 2 + c].f = static_cast<float>((meanR[c] - meanL[c]) / 255.0);
                            constants[extra + 5 + c].f = static_cast<float>(meanL[c] > 8.0 ? std::clamp(meanR[c] / meanL[c], 0.0, 8.0) : 1.0);
                        }
                        ncnn::VkMat dispatcher;
                        dispatcher.dims = 1; dispatcher.w = (left.width*left.height+3)/4; dispatcher.h = dispatcher.c = 1;
                        command.record_pipeline(impl_->fused.get(), {flow, packed, flag, full0, full1, mask}, constants, dispatcher);
                    } else {
                        constants[4].i = static_cast<int>(planar.cstep);
                        constants[5].i = static_cast<int>(full0.cstep);
                        constants[6].i = impl_->protectStaticDetails ? 1 : 0;
                        MGR_OBSERVE_JOB(w.observation.staticGuard=impl_->protectStaticDetails?GuardObservation::Executed:GuardObservation::Disabled;
                        w.observation.motionGuard=GuardObservation::NotApplicable;)
                        command.record_pipeline(impl_->postproc.get(), {planar, packed, flag, full0, full1}, constants, packed);
                    }

                    command.record_clone(packed, bytes, opt);
                    if (bytes.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                    command.record_clone(flag, flagged, opt);
                    if (flagged.empty()) {w.recordError=Code::AllocationFailed;return MGC_OUT_OF_MEMORY;}
                } else {

                    command.record_download(merged, downloaded, opt);
                }
                return MGC_SUCCESS;
            },&w);
            lock.lock();
            if(status!=MGC_SUCCESS){
                if(!meguviz::ncnn_gpu_scope_deferred()&&!w.callbackEntered)w.stage=0;
                if(status==MGC_BUSY)return {Code::Busy};
                if(status==MGC_DEVICE_LOST)return {Code::DeviceLost};
                return {status==MGC_OUT_OF_MEMORY?Code::AllocationFailed:w.recordError};
            }
        }
        if(!w.commandPath){
            if(!w.estimate||w.estimate->finish_extract_host("out0",result))return {Code::InferenceFailed};
        }else if(impl_->gpuIo){
                    inferred = nowNs();
                    // 只有整条命令成功后才把新帧放进缓存，失败路径不会留下未写入的显存张量。
                    for (auto& frame : fresh) {
                        impl_->frames[impl_->nextFrame] = frame;
                        impl_->nextFrame = (impl_->nextFrame + 1) % 3;
                    }
                    MGR_OBSERVE_JOB(w.observation.cachePublishObserved=true;w.observation.cachePublishedCount=uint32_t(fresh.size());)
                    if (flagged.empty() || bytes.empty() || static_cast<const std::uint32_t*>(flagged.data)[0] != 0)
                        return {Code::InvalidOutput};
                    std::array<bool, GuardTiles> patch{};
                    if (impl_->fusedWarp && w.motionGuard) {
                        // 抽样：每 8 组（每组 4 像素）计一组，按组首像素归入分块，与着色器一致。
                        const auto* counts = static_cast<const std::uint32_t*>(flagged.data);
                        std::array<std::int64_t, GuardTiles> sampledTiles{};
                        const std::int64_t pixels = static_cast<std::int64_t>(left.width) * left.height;
                        std::int64_t sampled = 0;
                        for (std::int64_t first = 0; first < pixels; first += 32) {
                            const std::int64_t ty = first / left.width, tx = first - ty * left.width;
                            const auto n = std::min<std::int64_t>(4, pixels - first);
                            sampledTiles[static_cast<std::size_t>((ty * GuardTilesY / left.height) * GuardTilesX + tx * GuardTilesX / left.width)] += n;
                            sampled += n;
                        }
                        impl_->unreliablePermille.store(static_cast<int>(std::min<std::int64_t>(1000, counts[1] * 1000 / std::max<std::int64_t>(1, sampled))));
                        // 局部转场（画中画、分屏切换、局部字幕卡）：多数像素不可靠的分块换成最近原帧；
                        // 与之相邻、也有相当比例不可靠的分块一并替换，避免接缝处残留拉扯。
                        std::array<int, GuardTiles> permille{};
                        for (int t = 0; t < GuardTiles; ++t)
                            permille[t] = sampledTiles[t] ? static_cast<int>(counts[2 + t] * 1000 / sampledTiles[t]) : 0;
                        for (int t = 0; t < GuardTiles; ++t) patch[t] = permille[t] >= TilePatchPermille;
                        std::array<bool, GuardTiles> core = patch;
                        for (int t = 0; t < GuardTiles; ++t) {
                            if (core[t] || permille[t] < TileNeighbourPermille) continue;
                            const int x = t % GuardTilesX, y = t / GuardTilesX;
                            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                                const int nx = x + dx, ny = y + dy;
                                if (nx >= 0 && ny >= 0 && nx < GuardTilesX && ny < GuardTilesY && core[ny * GuardTilesX + nx]) patch[t] = true;
                            }
                        }
                        // 分块回填只在周围静止时才不可见（分屏切换、静止画面上的画中画）。运动画面里（快速摇镜、
                        // 运动模糊）原帧方块与周围补帧对不上，形成拼贴方块并让该区域退回 30 fps——宿主真机抓帧所见。
                        // 已选分块周围的未选邻块里运动的超过三分之一时（摇镜：几乎全部在动），整帧都不回填；
                        // 局部切换（分屏、画中画）只有边界上部分变化的少数邻块在动。
                        std::array<double, GuardTiles> motion{};
                        tileMotion(left, right, motion);
                        std::array<bool, GuardTiles> ring{};
                        for (int t = 0; t < GuardTiles; ++t) {
                            if (!patch[t]) continue;
                            const int x = t % GuardTilesX, y = t / GuardTilesX;
                            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                                const int nx = x + dx, ny = y + dy;
                                if (nx < 0 || ny < 0 || nx >= GuardTilesX || ny >= GuardTilesY) continue;
                                if (!patch[ny * GuardTilesX + nx]) ring[ny * GuardTilesX + nx] = true;
                            }
                        }
                        int ringTiles = 0, movingRing = 0;
                        for (int t = 0; t < GuardTiles; ++t) if (ring[t]) { ++ringTiles; if (motion[t] > StaticTileDiff) ++movingRing; }
                        if (movingRing * 3 > ringTiles) patch.fill(false);
                        impl_->histogramPermille.store(histogramDistancePermille(left, right));
                    }
                    // No output writes until fence completion and the finite-value flag are validated.
                    // The download is already a stable tight RGB8 image; avoid an extra 4K staging copy.
                    for (int y = 0; y < output.height; ++y)
                        std::memcpy(output.data + static_cast<std::size_t>(y)*output.strideBytes,
                            static_cast<const std::uint8_t*>(bytes.data)+static_cast<std::size_t>(y)*rowBytes,rowBytes);
                    int patched = 0;
                    const ImageView& nearest = modelTimestep <= 0.5f ? left : right;
                    for (int t = 0; t < GuardTiles; ++t) {
                        if (!patch[t]) continue;
                        ++patched;
                        // 分块范围与着色器的 x*16/width 归属一致。
                        const int tx = t % GuardTilesX, ty = t / GuardTilesX;
                        const int x0 = (tx * output.width + GuardTilesX - 1) / GuardTilesX;
                        const int x1 = ((tx + 1) * output.width + GuardTilesX - 1) / GuardTilesX;
                        const int y0 = (ty * output.height + GuardTilesY - 1) / GuardTilesY;
                        const int y1 = ((ty + 1) * output.height + GuardTilesY - 1) / GuardTilesY;
                        for (int y = y0; y < y1; ++y)
                            std::memcpy(output.data + static_cast<std::size_t>(y) * output.strideBytes + static_cast<std::size_t>(x0) * 3,
                                nearest.data + static_cast<std::size_t>(y) * nearest.strideBytes + static_cast<std::size_t>(x0) * 3,
                                static_cast<std::size_t>(x1 - x0) * 3);
                    }
                    impl_->patchedTiles.store(patched);
                    impl_->timings[0].store(prepared-began);
                    impl_->timings[1].store(inferred-prepared);
                    impl_->timings[2].store(nowNs()-inferred);
                    MGR_OBSERVE_JOB(w.observation.completed=true;)
            w.stage=2;MGR_OBSERVE_JOB(impl_->diagnosticLast=w.observation;)return {};
        }else{
                    unpacked = downloaded;
                    if (downloaded.elempack != 1) ncnn::convert_packing(downloaded, unpacked, 1, opt);
                    if (unpacked.elembits() == 16) ncnn::cast_float16_to_float32(unpacked, result, opt);
                    else result = unpacked;
        }
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
        for (int y = 0; y < output.height; ++y)
            std::memcpy(output.data + static_cast<std::size_t>(y) * output.strideBytes,
                        impl_->staging.data() + static_cast<std::size_t>(y) * rowBytes, rowBytes);
        MGR_OBSERVE_JOB(w.observation.completed=true;)
        w.stage=2;MGR_OBSERVE_JOB(w.observation.completed=true;impl_->diagnosticLast=w.observation;)return {};
    }catch(const std::bad_alloc&){return {Code::AllocationFailed};}
    catch(...){return {Code::InferenceFailed};}
#endif
}
