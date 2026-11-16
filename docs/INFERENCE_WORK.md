# Retained inference work

The official accepted-job path stores a strong continuation in the job and
passes it through GPU IO, GPU/CPU bridge and the compatibility recorder. Work
keeps the Engine implementation, extractors, GPU/host Mats, patch data and
delayed host targets alive. The internal output WRITE lease is also a Compute
submission reference, not merely a field of the job.
After physical retirement and Work cleanup, that internal submission reference
is relinquished before the owned READ result is published. The job's writer
protects the handoff window; afterwards the result protects the output. A
consumer that releases the result in its callback can reuse a one-slot pool.

GPU IO records prepare, inference and final-patch stages once. The bridge keeps
upload, nested inference and download state. The compatibility recorder retains
its original upload/reduced graph/download/guard/pixel operations; split host
extraction shares the old ncnn packing/cast/detach implementation and actual
recording-time FP/OpenMP environment.

When a complete batch is only waiting for retirement, its host steps and later
stages resume after physical confirmation and host finalization. A partial or
failed graph is finalized as failure after its submitted work retires; it does
not resume as success. Cancellation similarly discards output only after the
required physical cleanup.

An inference-owner token prevents another job, reconfiguration or close from
modifying the same Engine. Diagnostics belong to each Work until the completed
observation is published. Record callbacks acquire their own Engine mutex, so
they can run later after deferred AHB readiness without relying on a vanished
call stack. User leases and plan references are returned outside GPU cleanup.

These changes establish the retained accepted RIFE path. They do not qualify
all generic Compute consumers, sync-fd probes, real Android AHB execution,
device-loss recovery or every numerical/quality profile. Small fixed-model
pixel comparisons are scoped evidence, not full visual or performance approval.
