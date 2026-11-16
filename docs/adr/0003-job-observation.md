# Optional result-owned job observation

`MGR_ENABLE_JOB_OBSERVATION` defaults to `OFF`. An enabled build records a
fixed-size snapshot of branches reached by each successful request. The
snapshot is copied into that request's owned `MgrResult` before the Compute
execution slot is released, so a later request or a second session sharing the
plan cannot replace it. `mgr_result_observation` returns
`MGC_UNSUPPORTED_CAPABILITY` when the build has observation disabled. Failed
requests do not publish a result and therefore do not manufacture a completed
observation.

The versioned `MgrJobObservation` carries the result request ID, frame kind,
effective flow scale, and separate left/right observations for RIFE's GPU
tensor cache and its GPU buffer bridge cache. `UNKNOWN`, `NOT_APPLICABLE`,
`DISABLED`, `MISS`, `HIT`, and `SHARED_PENDING` have distinct meanings. A
same-job pending upload is not an existing cache hit. The publication stage and
number of published tensor entries are separate. A failed job can have modified
internal caches; no completed observation is implied by failure.

Guard values describe the successfully completed dispatch and whether its
guard bit was enabled. They do not assert that any individual pixel was
changed. Endpoint and CPU paths report the guard as not applicable when that
guard path was not used. The JSON execution identity adds an
`active_configuration` only in an enabled build, taken from the loaded Engine's
actual configuration. It does not replace the separately captured sanitized
ncnn `graphs` options or their hash.

The Android ByteBuffer JNI adapter additionally tracks its own host-lease cache
and binds its snapshot to a monotonically advanced generation, the native result
request ID, and the caller's two source IDs. Failed, BUSY, and reconfigured
requests invalidate prior snapshots. The legacy engine has no native request ID;
its adapter explicitly reports that field as unobserved. The Media RIFE adapter
can retain the result-owned snapshot on the corresponding output frame, and
the Android AHB session exposes an explicit frame lookup. Missing snapshots
remain unavailable. This implementation is not by itself device or workload
qualification.

Enabled observation changes the RIFE build fingerprint and must be included in
the composition lock, package receipt, and measurement identity. Collection has
nonzero potential dispatch/JNI/serialization cost and must be enabled on both
sides of any comparison, with overhead measured separately. No output-quality,
device, or performance qualification follows from enabling this feature.
