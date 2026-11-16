// Runs only inside the Compute serial worker. Cached READ leases own completed
// immutable raw GPU bytes; current-call references outlive every GPU scope.
MgcStatus bridge_write(MgrPlan& plan,MgrPlan::BridgeSlot& slot,const MgrPlan::BridgeLayout& layout,
                       uint32_t width,uint32_t height,uint64_t stride,MgcLease** out) {
    *out=nullptr;
    if(plan.next_bridge_content==UINT64_MAX)return MGC_OUT_OF_MEMORY;
    const uint64_t content=plan.next_bridge_content++;
    auto lease=[&](MgcLease** result) {
        MgcLeaseDesc d{};d.header=MGC_HEADER_INIT(MgcLeaseDesc);d.content_id=content;d.byte_length=slot.length;d.access=MGC_ACCESS_WRITE;
        return mgc_lease_create(slot.resource,&d,result);
    };
    if(slot.resource) {
        MgcLease* probe=nullptr;auto status=lease(&probe);if(status)return status;
        if(slot.layout==layout){*out=probe;return MGC_SUCCESS;}
        // No shape replacement until an exclusive WRITE proves old readers and
        // physical users are gone. Drop old storage before creating replacement.
        mgc_resource_release(slot.resource);slot.resource=nullptr;slot.length=0;mgc_lease_release(probe);
    }
    MgcResourceDesc d{};d.header=MGC_HEADER_INIT(MgcResourceDesc);d.width=width;d.height=height;d.format=MGC_FORMAT_RGB8;
    d.usage=MGC_RESOURCE_GPU_BUFFER;d.row_stride_bytes=stride;d.capacity_bytes=(stride*height+3)&~uint64_t(3);
    auto status=mgc_resource_create(plan.model->domain,&d,&slot.resource);if(status)return status;
    slot.length=d.capacity_bytes;slot.layout=layout;return lease(out);
}
MgcStatus gpu_cpu_bridge(MgrPlan& plan,const MgrRequest& request,const MgcLeaseInfo& left,
                         const MgcLeaseInfo& right,const void* a,const void* b,void* out,int64_t bridge_ns[2]
#if defined(MGR_ENABLE_JOB_OBSERVATION)
                         ,legacy::CacheObservation observed[2]
#endif
                         ,std::shared_ptr<void>* continuation) {
    struct Buffers {
        MgrPlan* plan=nullptr;
        int stage=0;bool initialized=false,callbackEntered=false;MgcStatus terminal=MGC_SUCCESS;
        std::chrono::steady_clock::time_point began,downloadBegan;
        std::shared_ptr<void> inference;
        ncnn::Mat download;
        MgrPlan::BridgeSlot* frames[2]{};
        MgcLease* reads[2]{};MgcLease* writes[2]{};MgcLease* output=nullptr;
        MgrPlan::BridgeSlot* fresh[2]{};MgrPlan::CacheKey keys[2]{};
        ncnn::Mat upload[2];uint32_t uploads=0;
        ~Buffers(){
            inference.reset();
            for(auto* p:reads)mgc_lease_release(p);for(auto* p:writes)mgc_lease_release(p);mgc_lease_release(output);
            mgr_plan_release(plan);
        }
    };
    auto state=continuation&&*continuation?std::static_pointer_cast<Buffers>(*continuation):std::make_shared<Buffers>();
    auto& buffers=*state;
    if(buffers.plan&&buffers.plan!=&plan)return MGC_INVALID_ARGUMENT;
    if(!buffers.plan){buffers.plan=&plan;mgr_plan_retain(&plan);buffers.began=std::chrono::steady_clock::now();}
    if(continuation)*continuation=state;
    if(buffers.terminal)return buffers.terminal;
    if(!buffers.initialized) {
    const MgcLeaseInfo* inputs[]={&left,&right};const void* pixels[]={a,b};
    for(int i=0;i<2;i++) {
        const auto& info=*inputs[i];const auto key=input_key(info);
        if(++plan.bridge_clock==0){for(auto& slot:plan.bridge_inputs)slot.age=0;plan.bridge_clock=1;}
        for(auto& slot:plan.bridge_inputs)if(slot.read&&slot.key==key){buffers.frames[i]=&slot;slot.age=plan.bridge_clock;break;}
        if(buffers.frames[i]) {
#if defined(MGR_ENABLE_JOB_OBSERVATION)
            observed[i]=legacy::CacheObservation::Hit;
#endif
            buffers.reads[i]=buffers.frames[i]->read;mgc_lease_retain(buffers.reads[i]);continue;
        }
        // Identical left/right content shares even the first pending upload.
#if defined(MGR_ENABLE_JOB_OBSERVATION)
        if(i==1&&buffers.frames[0]&&input_key(left)==key){observed[i]=legacy::CacheObservation::SharedPending;buffers.frames[i]=buffers.frames[0];continue;}
        observed[i]=legacy::CacheObservation::Miss;
#else
        if(i==1&&buffers.frames[0]&&input_key(left)==key){buffers.frames[i]=buffers.frames[0];continue;}
#endif
        std::array<MgrPlan::BridgeSlot*,3> candidates{};size_t candidateCount=0;
        for(auto& slot:plan.bridge_inputs)if(&slot!=buffers.frames[0] && &slot!=buffers.frames[1])candidates[candidateCount++]=&slot;
        std::sort(candidates.begin(),candidates.begin()+candidateCount,[](const auto* x,const auto* y){return x->age<y->age;});
        const MgrPlan::BridgeLayout layout{info.domain_id,info.domain_generation,info.width,info.height,info.format,info.row_stride_bytes};
        MgcStatus status=MGC_BACKPRESSURE;MgrPlan::BridgeSlot* chosen=nullptr;
        for(size_t candidate=0;candidate<candidateCount;candidate++) {
            auto* slot=candidates[candidate];
            mgc_lease_release(slot->read);slot->read=nullptr;
            status=bridge_write(plan,*slot,layout,info.width,info.height,info.row_stride_bytes,&buffers.writes[buffers.uploads]);
            if(status==MGC_BUSY)continue;if(status!=MGC_SUCCESS)return status;chosen=slot;break;
        }
        if(!chosen)return MGC_BACKPRESSURE;
        buffers.frames[i]=chosen;chosen->age=plan.bridge_clock;
        const uint32_t index=buffers.uploads++;
        buffers.fresh[index]=chosen;buffers.keys[index]=key;
        const int words=int(chosen->length/4);
        if(info.byte_length>=chosen->length&&reinterpret_cast<uintptr_t>(pixels[i])%4==0) {
            buffers.upload[index]=ncnn::Mat(words,const_cast<void*>(pixels[i]),size_t(4u));
        } else {
            chosen->upload_scratch.create(words,size_t(4u));if(chosen->upload_scratch.empty())return MGC_OUT_OF_MEMORY;
            std::memset(chosen->upload_scratch.data,0,size_t(chosen->length));
            const uint64_t visible=(uint64_t(info.height)-1)*info.row_stride_bytes+uint64_t(info.width)*3;
            std::memcpy(chosen->upload_scratch.data,pixels[i],size_t(visible));buffers.upload[index]=chosen->upload_scratch;
        }
    }
    buffers.initialized=true;
    }
    MgcStatus status=MGC_SUCCESS;
    if(buffers.stage==0&&buffers.uploads) {
        buffers.stage=1;buffers.callbackEntered=false;
        status=meguviz::with_ncnn_gpu_owned(plan.model->domain,buffers.writes,buffers.uploads,state,[](void* context,meguviz::NcnnGpuScope& scope)->MgcStatus {
            auto& v=*static_cast<Buffers*>(context);v.callbackEntered=true;ncnn::Option options;
            options.blob_vkallocator=options.workspace_vkallocator=scope.blob_allocator;options.staging_vkallocator=scope.staging_allocator;
            for(uint32_t i=0;i<v.uploads;i++){if(!scope.views[i].buffer)return MGC_UNSUPPORTED_CAPABILITY;scope.command->record_clone(v.upload[i],*scope.views[i].buffer,options);}
            return MGC_SUCCESS;
        },&buffers);
        if(status){if(!meguviz::ncnn_gpu_scope_deferred()){if(!buffers.callbackEntered)buffers.stage=0;else buffers.terminal=status;}return status;}
    }
    if(buffers.stage==0)buffers.stage=1;
    if(buffers.stage==1) {
        for(uint32_t i=0;i<buffers.uploads;i++) {
            auto* slot=buffers.fresh[i];status=mgc_lease_publish_read(buffers.writes[i],&slot->read);if(status)return status;
            mgc_lease_release(buffers.writes[i]);buffers.writes[i]=nullptr;slot->key=buffers.keys[i];
        }
    for(int i=0;i<2;i++)if(!buffers.reads[i]){buffers.reads[i]=buffers.frames[i]->read;mgc_lease_retain(buffers.reads[i]);}
    const MgrPlan::BridgeLayout outputLayout{left.domain_id,left.domain_generation,request.output.width,request.output.height,
        request.output.format,request.output.row_stride_bytes};
    status=bridge_write(plan,plan.bridge_output,outputLayout,request.output.width,request.output.height,request.output.row_stride_bytes,&buffers.output);
    if(status)return status;
    bridge_ns[0]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-buffers.began).count();
    buffers.stage=2;
    }
    if(buffers.stage==2) {
    status=plan.engine->processGpu(plan.model->domain,buffers.reads[0],buffers.reads[1],buffers.output,request.alpha,
        input_cache_id(plan,left),input_cache_id(plan,right),int(request.flow_scale),&buffers.inference);
    if(status)return status;
    buffers.stage=3;
    }
    if(buffers.stage==3) {
    buffers.downloadBegan=std::chrono::steady_clock::now();
    buffers.download=plan.bridge_download;
    buffers.stage=4;buffers.callbackEntered=false;
    status=meguviz::with_ncnn_gpu_owned(plan.model->domain,&buffers.output,1,state,[](void* context,meguviz::NcnnGpuScope& scope)->MgcStatus {
        auto& v=*static_cast<Buffers*>(context);v.callbackEntered=true;if(!scope.views[0].buffer)return MGC_UNSUPPORTED_CAPABILITY;ncnn::Option options;
        options.blob_vkallocator=options.workspace_vkallocator=scope.blob_allocator;options.staging_vkallocator=scope.staging_allocator;
        scope.command->record_clone(*scope.views[0].buffer,v.download,options);return MGC_SUCCESS;
    },&buffers);
    if(status){if(!meguviz::ncnn_gpu_scope_deferred()){if(!buffers.callbackEntered)buffers.stage=3;else buffers.terminal=status;}return status;}
    }
    if(buffers.download.empty())return MGC_INTERNAL_ERROR;
    plan.bridge_download=buffers.download;
    for(uint32_t y=0;y<request.output.height;y++)std::memcpy(static_cast<uint8_t*>(out)+y*request.output.row_stride_bytes,
        static_cast<const uint8_t*>(buffers.download.data)+y*request.output.row_stride_bytes,size_t(request.output.width)*3);
    bridge_ns[1]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-buffers.downloadBegan).count();
    return MGC_SUCCESS;
}
