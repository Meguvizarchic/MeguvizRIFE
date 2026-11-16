# Model-upload ownership and explicit service

The official ncnn backend uses a pre-created transfer owner for every model
network. The immutable Engine holds the main network, reduced-resolution and
adaptive-scale networks, their transfer commands, copied model bytes, custom
layers, actual allocators, shaderpack references and Compute domain owner.
Network and command objects are installed before model upload may submit.
The exact private source SDK and its complete patch ledger must be rebuilt
together; individual candidate command objects must not replace an older SDK.

An unsuccessful `mgr_plan_create` can leave real upload work pending while
returning no plan. The failed Engine transfers its complete implementation to a
pre-reserved slot. Model/domain caller references may be dropped afterwards;
the upload owner keeps the objects it needs. This does not permit unloading the
library or destroying external objects that the caller separately owns.

Call `mgr_uploads_poll(domain, wait_ns)` from a worker to service abandoned
uploads of one exact origin domain. A null domain explicitly services this
module's abandoned uploads. Zero polls rather than waiting to acquire the GPU
execution slot. Each transfer wait is limited to 100 ms; multiple transfers and
driver calls are not a hard wall-clock API deadline.

SUCCESS means that no matching abandoned upload remains at that observation.
It does not mean a previously failed plan is now available. The caller must
explicitly retry plan creation, or continue shutdown. It also does not drain
live Engines, accepted inference jobs or consumer-owned outputs. BUSY retains
all required objects and allows another service attempt.

There are 16 pre-reserved Engine slots in the module and a separate Compute
owner limit per actual device. These are object-count admission limits, not a
bound on total network, GPU or driver memory. Permanent unknown retirement may
keep a slot occupied. No completion, fence satisfaction or successful close is
fabricated to reclaim capacity.

The host must keep its loader reference through service, successful owner
release and return from the API. Official single-DSO/static composition covers
the Engine callbacks with that code-image lifetime. Separately loaded RIFE
modules need their own code-image agreement.

This model-upload path does not repair the old self-owned ncnn model-loading
entry, VkCompute hidden-submit failure retirement, inference Work/output leases,
AHB/FD retirement or unified device-loss recovery. Those paths retain separate
implementation and validation gates.
