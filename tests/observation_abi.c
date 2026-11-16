#include <meguviz/rife.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    MgrJobObservation out;
    memset(&out,0xa5,sizeof(out));
    out.header=(MgcHeader)MGC_HEADER_INIT(MgrJobObservation);
    if(mgr_result_observation(NULL,&out)!=MGC_INVALID_ARGUMENT)return 1;
    if(mgr_result_observation((MgrResult*)(uintptr_t)1,NULL)!=MGC_INVALID_ARGUMENT)return 2;
    /* Invalid output prefixes must be rejected before accessing the result. */
    const uint32_t minimum=(uint32_t)offsetof(MgrJobObservation,reserved);
    for(uint32_t size=minimum-1;size<=minimum+3;++size){
        if(size==minimum)continue;
        memset(&out,0xa5,sizeof(out));out.header=(MgcHeader)MGC_HEADER_INIT(MgrJobObservation);out.header.struct_size=size;
        MgrJobObservation before=out;
        if(mgr_result_observation((MgrResult*)(uintptr_t)1,&out)!=MGC_UNSUPPORTED_ABI)return 3;
        if(memcmp(&out,&before,sizeof(out))!=0)return 4;
    }
    memset(&out,0xa5,sizeof(out));out.header=(MgcHeader)MGC_HEADER_INIT(MgrJobObservation);out.header.abi_major=MGR_ABI_MAJOR+1;
    if(mgr_result_observation((MgrResult*)(uintptr_t)1,&out)!=MGC_UNSUPPORTED_ABI)return 5;
    return 0;
}
