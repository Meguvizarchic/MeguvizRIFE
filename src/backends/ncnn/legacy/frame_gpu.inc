// Included inside the legacy namespace after Engine::Impl is complete.
MgcStatus Engine::processGpu(MgcDomain* domain, MgcLease* left, MgcLease* right, MgcLease* output,
                            double timestep, int64_t leftId, int64_t rightId, int flowScale, std::shared_ptr<void>* continuation) noexcept {
#if !NCNN_VULKAN
    (void)domain; (void)left; (void)right; (void)output; (void)timestep; (void)leftId; (void)rightId; (void)flowScale; (void)continuation;
    return MGC_UNSUPPORTED_CAPABILITY;
#else
    if(meguviz::ncnn_domain_is_lost(domain))return MGC_DEVICE_LOST;
    struct Work {
        std::shared_ptr<Impl> owner; Impl* engine=nullptr;
        int stage=0; bool initialized=false,callbackEntered=false;
        MgcStatus terminal=MGC_SUCCESS;
        int64_t began=0,prepared=0,inferred=0;
        bool motionGuard=false;
            MGR_OBSERVE_JOB(JobObservation observation{};JobObservation* diagnostic=&observation;)
            int width=0,height=0,scale=0,pw=0,ph=0,words=0;
            int64_t ids[2]; float alpha;
            bool endpoint=false;
            ncnn::VkMat raw[2],full[2],reduced[2],summary,flow,mask,merged,planar,packed,flags,patch;
            ncnn::Mat summaryHost,flagsHost,patchHost,time;
            std::vector<ncnn::Mat> hostKeep;
            std::vector<ncnn::Extractor> extractors;
            std::vector<Impl::GpuLeaseFrame> fresh;
            std::array<uint32_t,GuardTiles> patchMask{};
            float offset[3]{},gain[3]{1.f,1.f,1.f};
        ~Work() {
            if(!owner)return;
            meguviz::with_ncnn_domain_cleanup(owner->options.computeDomain,[](void* p,const ncnn::VulkanDevice*) -> MgcStatus {
                auto& w=*static_cast<Work*>(p);
                std::lock_guard<std::mutex> lock(w.owner->mutex);
                w.extractors.clear();w.fresh.clear();
                for(auto& m:w.raw)m.release();for(auto& m:w.full)m.release();for(auto& m:w.reduced)m.release();
                w.summary.release();w.flow.release();w.mask.release();w.merged.release();w.planar.release();w.packed.release();w.flags.release();w.patch.release();
                w.summaryHost.release();w.flagsHost.release();w.patchHost.release();w.time.release();w.hostKeep.clear();
                if(w.owner->activeInference==&w)w.owner->activeInference=nullptr;
                return MGC_SUCCESS;
            },this);
        }
    };

    std::shared_ptr<Work> state;
    try {
        std::unique_lock<std::mutex> lock(impl_->mutex,std::try_to_lock);
        if (!lock.owns_lock()) return MGC_BUSY;
        if (!impl_->loaded.load() || !impl_->gpuIo || !domain || !left || !right || !output) return MGC_INVALID_ARGUMENT;
        state=continuation&&*continuation?std::static_pointer_cast<Work>(*continuation):std::make_shared<Work>();
        auto& w=*state;
        if(w.owner&&w.owner!=impl_)return MGC_INCOMPATIBLE_DOMAIN;
        if(impl_->activeInference&&impl_->activeInference!=&w)return MGC_BUSY;
        w.owner=impl_;w.engine=impl_.get();impl_->activeInference=&w;
        if(continuation)*continuation=state;
        if(w.terminal!=MGC_SUCCESS)return w.terminal;
        MgcStatus status=MGC_SUCCESS;
        if(!w.initialized) {
        MGR_OBSERVE_JOB(w.observation=JobObservation{};w.observation.started=true;)
        const auto alphaClass=mgr_detail::classify_alpha(timestep);
        if (alphaClass==mgr_detail::Alpha::Invalid) return MGC_INVALID_ARGUMENT;
        w.alpha=static_cast<float>(timestep); w.ids[0]=leftId; w.ids[1]=rightId;
        w.endpoint=mgr_detail::alpha_endpoint(alphaClass);
        MGR_OBSERVE_JOB(w.diagnostic->endpoint=w.endpoint;
        if(w.endpoint){w.diagnostic->staticGuard=GuardObservation::NotApplicable;w.diagnostic->motionGuard=GuardObservation::NotApplicable;})
        MgcLeaseInfo info{}; info.header=MGC_HEADER_INIT(MgcLeaseInfo);
        status=mgc_lease_info(left,&info); if(status!=MGC_SUCCESS) return status;
        w.width=int(info.width); w.height=int(info.height);
        if(w.width<=0||w.height<=0||w.width>MaxDimension||w.height>MaxDimension||
            uint64_t(w.width)*w.height>impl_->options.maxPixels) return MGC_INVALID_ARGUMENT;
        w.scale=impl_->flowScale;
        if(impl_->adaptiveFlow) w.scale=flowScale>0?flowScale:uint64_t(w.width)*w.height>AdaptiveFlowPixels?8:impl_->flowScale;
        else if(flowScale>0&&flowScale!=w.scale) return MGC_INVALID_ARGUMENT;
        MGR_OBSERVE_JOB(w.diagnostic->cacheLookupScale=w.scale;w.diagnostic->effectiveScale=w.endpoint?0:w.scale;)
        if(w.scale!=impl_->flowScale&&(w.scale<1||w.scale>8||!impl_->extraDown[w.scale])) return MGC_INVALID_ARGUMENT;
        const int align=32*w.scale;
        w.pw=(w.width+align-1)/align*align; w.ph=(w.height+align-1)/align*align;
        w.words=(w.width*w.height*3+3)/4;
        w.began=nowNs();w.motionGuard=impl_->motionGuard;
        impl_->unreliablePermille.store(0); impl_->patchedTiles.store(0); impl_->histogramPermille.store(0);
        for(auto& t:impl_->timings) t.store(0);
        w.initialized=true;
        }
        MgcLease* inputs[]={left,right};
        if(w.stage==0) {
        w.stage=1;w.callbackEntered=false;
        lock.unlock();
        status=meguviz::with_ncnn_gpu_owned(domain,inputs,2,state,[](void* context,meguviz::NcnnGpuScope& scope)->MgcStatus {
            auto& v=*static_cast<Work*>(context);std::lock_guard<std::mutex> guard(v.engine->mutex);
            if(v.engine->activeInference!=&v)return MGC_BUSY;
            v.callbackEntered=true; auto& e=*v.engine; auto& cmd=*scope.command; const auto& opt=e.network->opt;
            if(scope.view_count!=2||scope.device!=e.device) return MGC_INCOMPATIBLE_DOMAIN;
            for(int i=0;i<2;i++) {
                const auto& source=scope.views[i];
                if(source.info.width!=uint32_t(v.width)||source.info.height!=uint32_t(v.height)||
                    !(source.info.access&MGC_ACCESS_READ)) return MGC_INVALID_ARGUMENT;
                const Impl::GpuLeaseFrame* hit=nullptr;
                for(const auto& frame:e.leaseFrames)
                    if(v.ids[i]>=0&&frame.id==v.ids[i]&&frame.width==v.width&&frame.height==v.height&&frame.scale==v.scale) hit=&frame;
                if(hit) {MGR_OBSERVE_JOB(v.diagnostic->tensorCache[i]=CacheObservation::Hit;)v.raw[i]=hit->raw;v.full[i]=hit->full;v.reduced[i]=hit->reduced;continue;}
                MGR_OBSERVE_JOB(v.diagnostic->tensorCache[i]=v.ids[i]<0?CacheObservation::Disabled:CacheObservation::Miss;)
                v.raw[i].create(v.words,size_t(4u),1,e.blobAllocator);
                if(v.raw[i].empty()) return MGC_OUT_OF_MEMORY;
                if(source.image) {
                    if(source.info.format!=MGC_FORMAT_RGBA8||source.info.byte_offset) return MGC_UNSUPPORTED_CAPABILITY;
                    std::vector<ncnn::vk_constant_type> c(3);c[0].i=v.width;c[1].i=v.height;c[2].i=v.words;
                    cmd.record_pipeline(e.gpuImageRead.get(),{v.raw[i]},{*source.image},c,v.raw[i]);
                } else if(source.buffer) {
                    if(source.info.format!=MGC_FORMAT_RGB8&&source.info.format!=MGC_FORMAT_RGBA8) return MGC_UNSUPPORTED_CAPABILITY;
                    if(source.info.byte_offset>INT32_MAX||source.info.row_stride_bytes>INT32_MAX) return MGC_INVALID_ARGUMENT;
                    std::vector<ncnn::vk_constant_type> c(6);c[0].i=v.width;c[1].i=v.height;
                    c[2].i=int(source.info.row_stride_bytes);c[3].i=source.info.format==MGC_FORMAT_RGBA8?4:3;
                    c[4].i=int(source.info.byte_offset);c[5].i=v.words;
                    cmd.record_pipeline(e.gpuUnpack.get(),{*source.buffer,v.raw[i]},c,v.raw[i]);
                } else return MGC_UNSUPPORTED_CAPABILITY;
                const size_t elem=opt.use_fp16_storage?2u:4u;
                if(e.rawGpuInput) v.full[i]=v.raw[i];
                else {
                    v.full[i].create(v.pw,v.ph,3,elem,1,e.blobAllocator);
                    if(v.full[i].empty()) return MGC_OUT_OF_MEMORY;
                    std::vector<ncnn::vk_constant_type> c(6);c[0].i=v.width;c[1].i=v.height;c[2].i=v.width*3;
                    c[3].i=v.pw;c[4].i=v.ph;c[5].i=int(v.full[i].cstep);
                    cmd.record_pipeline(e.preproc.get(),{v.raw[i],v.full[i]},c,v.full[i]);
                }
                v.reduced[i]=v.full[i];
                if(v.scale!=1) {
                    if(e.rawGpuInput) {
                        v.reduced[i].create(v.pw/v.scale,v.ph/v.scale,3,elem,1,e.blobAllocator);
                        if(v.reduced[i].empty()) return MGC_OUT_OF_MEMORY;
                        std::vector<ncnn::vk_constant_type> r(7);r[0].i=v.width;r[1].i=v.height;r[2].i=v.width*3;
                        r[3].i=v.reduced[i].w;r[4].i=v.reduced[i].h;r[5].i=int(v.reduced[i].cstep);r[6].i=v.scale;
                        cmd.record_pipeline(e.reduce.get(),{v.raw[i],v.reduced[i]},r,v.reduced[i]);
                    } else {
                        auto* down=v.scale==e.flowScale?e.downscale.get():e.extraDown[v.scale].get();
                        v.extractors.push_back(down->create_extractor());auto& ex=v.extractors.back();ex.set_light_mode(false);
                        if(ex.input("in0",v.full[i])!=0||ex.extract("half0",v.reduced[i],cmd)!=0) return MGC_INTERNAL_ERROR;
                    }
                }
                if(v.ids[i]>=0) v.fresh.push_back({v.ids[i],v.width,v.height,v.scale,v.raw[i],v.full[i],v.reduced[i]});
            }
            if(v.endpoint) return MGC_SUCCESS;
            // Integer summaries use exactly the original RGB8 sample grid. No
            // planar FP16 rounding is allowed into frame brightness/tile policy.
            ncnn::Mat zeros(391,size_t(4u));if(zeros.empty()) return MGC_OUT_OF_MEMORY;
            std::memset(zeros.data,0,391*sizeof(uint32_t));v.hostKeep.push_back(zeros);
            cmd.record_clone(zeros,v.summary,opt);
            if(v.summary.empty()) return MGC_OUT_OF_MEMORY;
            const int sx=std::max(1,v.width/320),sy=std::max(1,v.height/180);
            std::vector<ncnn::vk_constant_type> c(6);c[0].i=v.width;c[1].i=v.height;c[2].i=sx;c[3].i=sy;
            c[4].i=(v.width+sx-1)/sx;c[5].i=(v.height+sy-1)/sy;
            ncnn::VkMat dispatcher;dispatcher.dims=2;dispatcher.w=c[4].i;dispatcher.h=c[5].i;dispatcher.c=1;
            cmd.record_pipeline(e.gpuStatistics.get(),{v.raw[0],v.raw[1],v.summary},c,dispatcher);
            cmd.record_clone(v.summary,v.summaryHost,opt);if(v.summaryHost.empty())return MGC_OUT_OF_MEMORY;
            return MGC_SUCCESS;
        },&w);
        lock.lock();
        if(status!=MGC_SUCCESS) {
            if(!meguviz::ncnn_gpu_scope_deferred()){if(!w.callbackEntered)w.stage=0;else w.terminal=status;}
            return status;
        }
        }
        if(w.stage==1) {
        for(auto& frame:w.fresh) {impl_->leaseFrames[impl_->nextLeaseFrame]=frame;impl_->nextLeaseFrame=(impl_->nextLeaseFrame+1)%3;}
        MGR_OBSERVE_JOB(w.diagnostic->cachePublishObserved=true;w.diagnostic->cachePublishedCount=uint32_t(w.fresh.size());)
        w.prepared=nowNs();
        if(!w.endpoint) {
            if(w.summaryHost.empty()) return MGC_INTERNAL_ERROR;
            const auto* sums=static_cast<const uint32_t*>(w.summaryHost.data);
            const double count=sums[6];if(count<=0) return MGC_INTERNAL_ERROR;
            for(int c=0;c<3;c++) {
                const double a=sums[c]/count,b=sums[3+c]/count;
                w.offset[c]=float((b-a)/255.0);w.gain[c]=float(a>8.0?std::clamp(b/a,0.0,8.0):1.0);
            }
            w.time.create(w.pw/w.scale,w.ph/w.scale,1);if(w.time.empty()) return MGC_OUT_OF_MEMORY;w.time.fill(w.alpha);
        }
        w.stage=w.endpoint?4:2;
        }
        if(w.stage==2) {
            w.stage=3;w.callbackEntered=false;
            lock.unlock();
        status=meguviz::with_ncnn_gpu_owned(domain,inputs,2,state,[](void* context,meguviz::NcnnGpuScope& scope)->MgcStatus {
                auto& v=*static_cast<Work*>(context);std::lock_guard<std::mutex> guard(v.engine->mutex);
                if(v.engine->activeInference!=&v)return MGC_BUSY;
                v.callbackEntered=true;auto& e=*v.engine;auto& cmd=*scope.command;const auto& opt=e.network->opt;
                v.extractors.push_back(e.network->create_extractor());auto& estimate=v.extractors.back();estimate.set_light_mode(e.fusedWarp);
                if(estimate.input("in0",v.reduced[0])||estimate.input("in1",v.reduced[1])||estimate.input("in2",v.time)) return MGC_INTERNAL_ERROR;
                if(v.scale==1) {if(estimate.extract("out0",v.merged,cmd)) return MGC_INTERNAL_ERROR;}
                else {
                    if(estimate.extract(FlowBlob,v.flow,cmd)||estimate.extract(MaskBlob,v.mask,cmd)) return MGC_INTERNAL_ERROR;
                    if(!e.fusedWarp) {
                        auto* up=v.scale==e.flowScale?e.upscale.get():e.extraUp[v.scale].get();
                        v.extractors.push_back(up->create_extractor());auto& ex=v.extractors.back();
                        if(ex.input("in0",v.full[0])||ex.input("in1",v.full[1])||ex.input("flow",v.flow)||ex.input("mask",v.mask)||ex.extract("out0",v.merged,cmd)) return MGC_INTERNAL_ERROR;
                    }
                }
                v.packed.create(v.words,size_t(4u),1,e.blobAllocator);
                ncnn::Mat zeros(2+GuardTiles,size_t(4u));if(v.packed.empty()||zeros.empty()) return MGC_OUT_OF_MEMORY;
                std::memset(zeros.data,0,(2+GuardTiles)*sizeof(uint32_t));v.hostKeep.push_back(zeros);cmd.record_clone(zeros,v.flags,opt);
                if(v.flags.empty())return MGC_OUT_OF_MEMORY;
                if(e.fusedWarp) {
                    if(v.flow.w!=v.pw/v.scale||v.flow.h!=v.ph/v.scale||v.flow.c*v.flow.elempack!=4||
                        v.mask.w!=v.flow.w||v.mask.h!=v.flow.h||v.mask.c*v.mask.elempack!=1||
                        (v.flow.elempack!=1&&v.flow.elempack!=4)) return MGC_MODEL_MISMATCH;
                    std::vector<ncnn::vk_constant_type> c(e.rawGpuInput?21:20);
                    c[0].i=v.width;c[1].i=v.height;c[2].i=v.words;c[3].i=v.pw;
                    c[4].i=e.rawGpuInput?v.width*3:int(v.full[0].cstep);c[5].i=e.protectStaticDetails;c[6].i=v.ph;
                    MGR_OBSERVE_JOB(v.diagnostic->staticGuard=e.protectStaticDetails?GuardObservation::Executed:GuardObservation::Disabled;
                    v.diagnostic->motionGuard=v.motionGuard?GuardObservation::Executed:GuardObservation::Disabled;)
                    c[7].i=v.flow.w;c[8].i=v.flow.h;c[9].i=int(v.flow.cstep);c[10].i=v.flow.elempack;c[11].i=v.scale;
                    if(e.rawGpuInput)c[12].i=v.width*3;const int at=e.rawGpuInput?13:12;
                    c[at].f=v.alpha;c[at+1].i=v.motionGuard;
                    for(int j=0;j<3;j++){c[at+2+j].f=v.offset[j];c[at+5+j].f=v.gain[j];}
                    ncnn::VkMat dispatcher;dispatcher.dims=1;dispatcher.w=(v.width*v.height+3)/4;dispatcher.h=dispatcher.c=1;
                    cmd.record_pipeline(e.fused.get(),{v.flow,v.packed,v.flags,e.rawGpuInput?v.raw[0]:v.full[0],e.rawGpuInput?v.raw[1]:v.full[1],v.mask},c,dispatcher);
                } else {
                    if(v.merged.dims!=3||v.merged.w!=v.pw||v.merged.h!=v.ph||v.merged.c*v.merged.elempack!=3) return MGC_MODEL_MISMATCH;
                    v.planar=v.merged;if(v.merged.elempack!=1)e.device->convert_packing(v.merged,v.planar,1,cmd,opt);
                    if(v.planar.empty())return MGC_OUT_OF_MEMORY;
                    if(v.planar.dims!=3||v.planar.w!=v.pw||v.planar.h!=v.ph||v.planar.c!=3||v.planar.elempack!=1)return MGC_MODEL_MISMATCH;
                    std::vector<ncnn::vk_constant_type> c(7);c[0].i=v.width;c[1].i=v.height;c[2].i=v.words;c[3].i=v.pw;
                    c[4].i=int(v.planar.cstep);c[5].i=int(v.full[0].cstep);c[6].i=e.protectStaticDetails;
                    MGR_OBSERVE_JOB(v.diagnostic->staticGuard=e.protectStaticDetails?GuardObservation::Executed:GuardObservation::Disabled;
                    v.diagnostic->motionGuard=GuardObservation::NotApplicable;)
                    cmd.record_pipeline(e.postproc.get(),{v.planar,v.packed,v.flags,v.full[0],v.full[1]},c,v.packed);
                }
                cmd.record_clone(v.flags,v.flagsHost,opt);if(v.flagsHost.empty())return MGC_OUT_OF_MEMORY;return MGC_SUCCESS;
            },&w);
            lock.lock();
            if(status!=MGC_SUCCESS){
                if(!meguviz::ncnn_gpu_scope_deferred()){if(!w.callbackEntered)w.stage=2;else w.terminal=status;}
                return status;
            }
        }
        if(w.stage==3) {
            const auto* sums=static_cast<const uint32_t*>(w.summaryHost.data);
            if(w.flagsHost.empty()||static_cast<const uint32_t*>(w.flagsHost.data)[0])return MGC_MODEL_MISMATCH;
            if(impl_->fusedWarp&&w.motionGuard) {
                const auto* flags=static_cast<const uint32_t*>(w.flagsHost.data);
                auto& samples=impl_->guardSampleCounts;
                samples.update(w.width,w.height);
                const auto& sampledTiles=samples.tiles;
                impl_->unreliablePermille.store(int(std::min<int64_t>(1000,int64_t(flags[1])*1000/std::max<int64_t>(1,samples.total))));
                std::array<int,GuardTiles> permille{};
                for(int t=0;t<GuardTiles;t++){permille[t]=sampledTiles[t]?int(int64_t(flags[2+t])*1000/sampledTiles[t]):0;w.patchMask[t]=permille[t]>=TilePatchPermille;}
                const auto core=w.patchMask;
                for(int t=0;t<GuardTiles;t++) {
                    if(core[t]||permille[t]<TileNeighbourPermille)continue;const int x=t%GuardTilesX,y=t/GuardTilesX;
                    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
                        const int nx=x+dx,ny=y+dy;if(nx>=0&&ny>=0&&nx<GuardTilesX&&ny<GuardTilesY&&core[ny*GuardTilesX+nx])w.patchMask[t]=1;
                    }
                }
                std::array<bool,GuardTiles> ring{};
                for(int t=0;t<GuardTiles;t++)if(w.patchMask[t]) {
                    const int x=t%GuardTilesX,y=t/GuardTilesX;
                    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
                        const int nx=x+dx,ny=y+dy;if(nx>=0&&ny>=0&&nx<GuardTilesX&&ny<GuardTilesY&&!w.patchMask[ny*GuardTilesX+nx])ring[ny*GuardTilesX+nx]=true;
                    }
                }
                int ringTiles=0,moving=0;
                for(int t=0;t<GuardTiles;t++)if(ring[t]){ringTiles++;if(sums[151+t]&&double(sums[7+t])/sums[151+t]>StaticTileDiff)moving++;}
                if(moving*3>ringTiles)w.patchMask.fill(0);
                int64_t l1=0;for(int i=0;i<48;i++)l1+=std::abs(int64_t(sums[295+i])-sums[343+i]);
                impl_->histogramPermille.store(int(l1*1000/(6*int64_t(sums[6]))));
            }
        }
        if(w.stage==3)w.stage=4;
        if(w.stage==4) {
        w.inferred=nowNs();
        w.patchHost.create(GuardTiles,size_t(4u));if(w.patchHost.empty())return MGC_OUT_OF_MEMORY;
        std::memcpy(w.patchHost.data,w.patchMask.data(),GuardTiles*sizeof(uint32_t));
        MgcLease* finalLeases[]={left,right,output};
        w.stage=5;w.callbackEntered=false;
        lock.unlock();
        status=meguviz::with_ncnn_gpu_owned(domain,finalLeases,3,state,[](void* context,meguviz::NcnnGpuScope& scope)->MgcStatus {
            auto& v=*static_cast<Work*>(context);std::lock_guard<std::mutex> guard(v.engine->mutex);
                if(v.engine->activeInference!=&v)return MGC_BUSY;
                v.callbackEntered=true;auto& e=*v.engine;auto& cmd=*scope.command;const auto& target=scope.views[2];
            if(target.info.width!=uint32_t(v.width)||target.info.height!=uint32_t(v.height)||!(target.info.access&MGC_ACCESS_WRITE))return MGC_INVALID_ARGUMENT;
            cmd.record_clone(v.patchHost,v.patch,e.network->opt);
            if(v.patch.empty())return MGC_OUT_OF_MEMORY;
            const auto& nearest=v.raw[v.alpha<=0.5f?0:1];const auto& packed=v.endpoint?nearest:v.packed;
            if(target.image) {
                if(target.info.format!=MGC_FORMAT_RGBA8||target.info.byte_offset)return MGC_UNSUPPORTED_CAPABILITY;
                std::vector<ncnn::vk_constant_type> c(3);c[0].i=v.width;c[1].i=v.height;c[2].i=v.endpoint;
                ncnn::Mat dispatcher(v.width,v.height,static_cast<void*>(nullptr));
                cmd.record_pipeline(e.gpuImageWrite.get(),{packed,nearest,v.patch},{*target.image},c,dispatcher);
            } else if(target.buffer) {
                if(target.info.byte_offset%4||target.info.byte_offset>INT32_MAX||target.info.row_stride_bytes>INT32_MAX||
                    (target.info.format!=MGC_FORMAT_RGB8&&target.info.format!=MGC_FORMAT_RGBA8))return MGC_UNSUPPORTED_CAPABILITY;
                std::vector<ncnn::vk_constant_type> c(7);c[0].i=v.width;c[1].i=v.height;c[2].i=int(target.info.row_stride_bytes);
                c[3].i=target.info.format==MGC_FORMAT_RGBA8?4:3;c[4].i=int(target.info.byte_offset);
                c[5].i=int((target.info.row_stride_bytes*v.height+3)/4);c[6].i=v.endpoint;
                ncnn::VkMat dispatcher;dispatcher.dims=1;dispatcher.w=c[5].i;dispatcher.h=dispatcher.c=1;
                cmd.record_pipeline(e.gpuPatch.get(),{packed,nearest,v.patch,*target.buffer},c,dispatcher);
            } else return MGC_UNSUPPORTED_CAPABILITY;
            return MGC_SUCCESS;
        },&w);
        lock.lock();
        if(status!=MGC_SUCCESS){
            if(!meguviz::ncnn_gpu_scope_deferred()){if(!w.callbackEntered)w.stage=4;else w.terminal=status;}
            return status;
        }
        }
        int patched=0;for(auto value:w.patchMask)patched+=value!=0;impl_->patchedTiles.store(patched);
        impl_->timings[0].store(w.prepared-w.began);impl_->timings[1].store(w.inferred-w.prepared);impl_->timings[2].store(nowNs()-w.inferred);
        w.stage=6;MGR_OBSERVE_JOB(w.diagnostic->completed=true;impl_->diagnosticLast=w.observation;) return MGC_SUCCESS;
    } catch(const std::bad_alloc&) {return MGC_OUT_OF_MEMORY;}
    catch(...) {return MGC_INTERNAL_ERROR;}
#endif
}
