namespace {
mgr_detail::J shaderOptions(const ncnn::Option& o) {
    return mgr_detail::Parser(meguviz::ncnn_shader_options_json(o)).parse();
}
std::string shaderHex(const MgcHash& h) {
    static constexpr char digits[]="0123456789abcdef";std::string out;
    for(auto b:h.bytes){out+=digits[b>>4];out+=digits[b&15];}return out;
}
}
MgcStatus Engine::Impl::makePipeline(uint32_t kernel,const char* logical,const char* source,int size,
                                   const ncnn::Option& effective,int x,int y,int z,
                                   const std::vector<int>& bindings,int pushCount,PipelineSlot& slot) {
    using mgr_detail::J;
    std::string capturedFacts;
    const auto factsStatus=meguviz::ncnn_shader_target_facts(device,capturedFacts);
    if(factsStatus!=MGC_SUCCESS)return factsStatus;
    const J facts=mgr_detail::Parser(capturedFacts).parse(),effectiveJson=shaderOptions(effective);
    // ncnn may adapt a requested workgroup for the actual device subgroup.
    // Compile and fingerprint the exact size that Pipeline will dispatch.
    ncnn::Pipeline dimensions(device);dimensions.set_optimal_local_size_xyz(x,y,z);
    x=int(dimensions.local_size_x());y=int(dimensions.local_size_y());z=int(dimensions.local_size_z());
    auto requested=requestedNetworkOptions;
    if(kernel==MGR_SHADER_REDUCE&&options.directReducedInput)requested.use_fp16_arithmetic=false;
    const J requestedJson=shaderOptions(requested);
    const std::string optionsHash=mgr_detail::sha256(mgr_detail::dump(effectiveJson));
    const std::string factsHash=mgr_detail::sha256(mgr_detail::dump(facts));
    std::string offlineSource(source,size);
    const auto layout=offlineSource.find("layout");if(layout==std::string::npos)return MGC_ARTIFACT_MISMATCH;
    offlineSource.insert(layout,"layout(local_size_x="+std::to_string(x)+",local_size_y="+std::to_string(y)+
        ",local_size_z="+std::to_string(z)+") in;\n");
    const auto sourceHash=mgr_detail::sha256(offlineSource);
    const std::string target=facts.flag("maintenance4_enabled")?"vulkan1.3":
        effective.use_subgroup_ops||effective.use_cooperative_matrix?"vulkan1.1":"vulkan1.0";
    J::A bindingTypes;for(int type:bindings)bindingTypes.push_back(type);
    const J request=J::O{{"schema_version",1},{"logical_shader_id",logical},
        {"owner",J::O{{"repository","MeguvizRIFE"},{"revision","source-sha256:"+sourceHash}}},
        {"source",J::O{{"path","shader.comp"},{"sha256",sourceHash}}},{"includes",J::A{}},
        {"profile_id","ncnn-compat"},{"profile_version","20261020"},{"entry_point","main"},
        {"target_env",target},{"effective_options",effectiveJson},{"numeric_profile","rife-legacy-explicit-storage-rounding"},
        {"toolchain_lock_hash",J()},{"required_flags",J::A{}},{"optimization_passes",J::A{}},
        {"resource_recipe","ncnn-20261020"},{"ncnn_revision","7e36d25cb153092ba87a60530455c7376fedf452"},
        {"target_facts",facts}};
    J record=J::O{{"kernel",kernel},{"logical_shader_id",logical},{"source_utf8",offlineSource},
        {"source_sha256",sourceHash},{"requested_options",requestedJson},{"effective_options",effectiveJson},
        {"effective_options_hash",optionsHash},{"target_facts",facts},{"target_facts_hash",factsHash},
        {"target_env",target},{"private_sdk_fingerprint",meguviz::private_sdk_fingerprint()},
        {"expected_binding_types",bindingTypes},{"expected_push_constant_slots",pushCount},
        {"expected_local_size",J::A{x,y,z}},{"compile_request",request}};
    const MgrShaderBinding* bound=nullptr;
    for(const auto& binding:options.shaderBindings)if(binding.kernel==kernel)bound=&binding;
    if(bound) {
        // All explicit bindings are fail-closed. Only an absent binding is a
        // compatibility miss that is allowed to invoke runtime glslang.
        if(!options.shaderpack||shaderHex(bound->effective_options_hash)!=optionsHash||
            shaderHex(bound->target_facts_hash)!=factsHash||shaderHex(bound->source_sha256)!=sourceHash||
            shaderHex(bound->private_sdk_fingerprint)!=meguviz::private_sdk_fingerprint())return MGC_ARTIFACT_MISMATCH;
        MgcShaderArtifactRequest selection{};selection.header=MGC_HEADER_INIT(MgcShaderArtifactRequest);
        selection.logical_shader_id=logical;selection.logical_shader_id_length=uint32_t(std::strlen(logical));
        selection.variant_key=bound->variant_key;selection.effective_options_hash=bound->effective_options_hash;
        MgcShaderArtifactView view{};view.header=MGC_HEADER_INIT(MgcShaderArtifactView);
        auto status=mgc_shaderpack_select(options.shaderpack,&selection,&view);if(status)return status;
        if(view.profile!=MGC_SHADER_PROFILE_NCNN_COMPAT||shaderHex(view.spirv_hash)!=shaderHex(bound->spirv_sha256)||
            view.local_size[0]!=uint32_t(x)||view.local_size[1]!=uint32_t(y)||view.local_size[2]!=uint32_t(z))return MGC_ARTIFACT_MISMATCH;
        ncnn::ShaderInfo info{};
        if(ncnn::resolve_shader_info(view.spirv_words,size_t(view.spirv_size_bytes),info)||
            info.binding_count!=int(bindings.size())||info.push_constant_count!=pushCount||info.specialization_count)return MGC_ARTIFACT_MISMATCH;
        for(size_t i=0;i<bindings.size();i++)if(info.binding_types[i]!=bindings[i])return MGC_ARTIFACT_MISMATCH;
        ncnn::Pipeline* pipeline=nullptr;
        status=meguviz::ncnn_shaderpack_pipeline(options.shaderpack,&selection,effective,{},&pipeline);if(status)return status;
        if(!pipeline||pipeline->vkdev!=device)return MGC_INCOMPATIBLE_DOMAIN;
        slot.borrowed=pipeline;shaderCounters[0]++;
        record["selection"]="offline_pack";record["variant_key"]=shaderHex(bound->variant_key);
        record["spirv_sha256"]=shaderHex(bound->spirv_sha256);shaderRecords.push_back(std::move(record));return MGC_SUCCESS;
    }
    if(options.shaderpack){shaderCounters[2]++;record["selection"]="variant_miss_runtime_fallback";}
    else record["selection"]="runtime_compat_no_pack";
    // Publish the miss event before invoking the compiler. It stays in this
    // immutable plan's catalog; failures cannot publish a half-created plan.
    shaderRecords.push_back(record);shaderCounters[1]++;
    std::vector<uint32_t> spirv;
    if(ncnn::compile_spirv_module(offlineSource.data(),int(offlineSource.size()),effective,spirv)||spirv.empty())return MGC_UNSUPPORTED_CAPABILITY;
    auto pipeline=std::make_unique<ncnn::Pipeline>(device);pipeline->set_optimal_local_size_xyz(x,y,z);
    if(pipeline->create(spirv.data(),spirv.size()*sizeof(uint32_t),{}))return MGC_UNSUPPORTED_CAPABILITY;
    slot=std::move(pipeline);return MGC_SUCCESS;
}
