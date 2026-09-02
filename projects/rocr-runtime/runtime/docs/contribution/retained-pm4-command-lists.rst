.. meta::
   :description: Design and qualification plan for retained PM4 graph command lists
   :keywords: ROCR, HIP graphs, PM4, AQL, GFX11, GFX12

.. _retained-pm4-command-lists:

Retained PM4 graph command lists
================================

Status
------

This is an experimental, opt-in design. It is not a general replacement for
ordinary AQL dispatch. Runtime enablement is limited to ASICs that have passed
the qualification gates in this document; an encoder compiling for an ISA does
not qualify that ISA for runtime use.

The initial prototype is implemented at the HIP/ROCclr boundary and is enabled
with ``DEBUG_HIP_GRAPH_PM4=1``. The shippable design moves PM4 encoding,
validation, executable allocation, and vendor-packet materialization behind an
opaque ROCR prepared-command-list object. HIP retains ownership of graph
lifetime and submits the resulting vendor packet through its existing stream
queue so queue tracking, stream ordering, profiling, and completion remain
consistent with ordinary graph replay.

All declared GFX11 and GFX12 targets have compile-time encoders. Hardware that
has not completed qualification additionally requires
``DEBUG_HIP_GRAPH_PM4_UNQUALIFIED=1``. This second gate is for bring-up only and
does not add the ASIC to the default qualification allowlist.

Reference snapshot
------------------

The reference implementation and measurements are frozen at:

* ROCm branch ``pwilkin/ilintar-experiments``, base commit
  ``2f725f1807a17efd772156649829052012a949f7``.
* Redline branch ``pwilkin/hipgraph-exec-update``, commit
  ``20474b8c1a5bf31189e5a94b34a8e9689c028e54``. The GFX11 and GFX12 register
  maps and dependency boundaries are semantically matched against Redline's
  Apache-2.0 encoders; this provenance must be preserved.
* Hardware ``gfx1151`` (Radeon 8060S / Strix Halo).
* Workload Ornith 1.5 35B-A3B IQ4_XS, token generation with
  ``llama-bench -p 0 -n 128 -r 4 -o json``.

The locked, counterbalanced AQL/PM4/Vulkan bracket used two four-sample arms
per backend:

.. list-table:: Reference token-generation result
   :header-rows: 1

   * - Backend
     - Mean tokens/s
     - Median tokens/s
     - Range
   * - Ordinary HIP AQL graph
     - 67.2651
     - 67.1490
     - 66.6806--67.9936
   * - Vulkan/RADV
     - 71.3325
     - 71.3075
     - 70.8369--71.7543
   * - Retained PM4 through HIP's AQL queue
     - 78.7651
     - 78.7161
     - 78.2785--79.3113

Retained PM4 improved the matched AQL mean by 17.10% and the Vulkan mean by
10.42%. A final safety-gated run logged a 1,585-dispatch warm graph encoded in
46,716 dwords and a 1,525-dispatch decode graph encoded in 45,036 dwords.

Numerical reference
-------------------

The correctness oracle hashes every float in the 248,320-entry vocabulary for
16 autoregressive steps. Ordinary AQL and retained PM4 produced identical
top-token IDs, hexadecimal top logits, and the following full-logit hashes:

.. code-block:: text

   2e85ee028c4b3384 e85fbbeb4e77d279 abe6e73da1f64637 2809f3726cd528b3
   51e99dfd0caa62e5 1c2170cf944acbfa 63f05a3fe10b995a a5969729823db0d9
   6feb2bbf426a3b09 fc1692d086963485 21a7d969401b0e29 dd14dff8f8882e83
   6774f63ab90f96d0 9a69083ce4c68d61 c00d35ac8a8fd936 03b4d4f9e3bb1c52

Nanbeige queue-scratch qualification
------------------------------------

Nanbeige4.1 3B BF16 contains a wave32 flash-attention tile with a 32-byte
fixed private segment. Its gfx1151 code-object metadata, runtime kernel-object
offset, and entry offset identify the specialization as
``flash_attn_tile<128, 128, 2, 1, false>``. The command list retains 422
dispatches in 12,050 dwords and reports a maximum private segment of 32 bytes.

Sixteen autoregressive steps over the full 166,144-entry vocabulary matched
ordinary AQL exactly. Each step matched the raw-float hash, top-token ID, and
hexadecimal top logit. The hashes were:

.. code-block:: text

   96c565bf1f5b741a dbbd5bf1bce27f29 47db84f3e1b81e3e a03042a9a4eb0c82
   2ad0ef2c3f984466 56a7902498272377 0c872fc0d886bfb1 80ee8a0b94015405
   c7d620757f865e59 9deb3dbaa1331a20 7f8a8a4c5360e879 a2d7607ef8ab45dc
   b902fd8b72be78cf 4b985fdc09b4352e 45620b3d9b4d6412 6b8256e1219f19df

The locked throughput bracket used
``llama-bench -dev ROCm0 -p 0 -n 128 -r 3 -o json`` in the counterbalanced
order vanilla, custom AQL, custom PM4, custom PM4, custom AQL, vanilla. Each
row contains six samples:

.. list-table:: Nanbeige4.1 3B BF16 token-generation result
   :header-rows: 1

   * - Backend
     - Mean tokens/s
     - Median tokens/s
     - Range
   * - Vanilla ROCm
     - 30.0275
     - 30.0385
     - 29.9521--30.1212
   * - Custom ordinary AQL
     - 30.0889
     - 30.0910
     - 30.0621--30.1087
   * - Custom retained PM4
     - 30.6051
     - 30.6221
     - 30.4681--30.6786

Retained PM4 improved the mean by 1.92% over vanilla ROCm and 1.72% over the
matched custom AQL runtime. A separate lifecycle gate launched one 32-dispatch,
48-byte-private graph sixteen times across two streams, destroyed the graph and
executable before stream synchronization, and ran graph-memory trim
concurrently. Its final atomic count was exactly 512 of 512.

The synthetic floor uses a non-atomic dependent ``value = value + 1`` chain.
Atomic increments and no-op kernels are not acceptable visibility gates.

Non-negotiable ownership invariants
-----------------------------------

1. A prepared command list is owned by one instantiated graph packet batch;
   there is no process-global pointer-keyed cache.
2. A graph packet rebuild or graph-exec parameter update invalidates that
   batch's prepared list.
3. Every in-flight launch retains its prepared-list handle until the launch's
   completion command is destroyed. Updating or destroying the graph cannot
   free an IB still visible to the GPU.
4. Kernarg pointers, code objects, and pointees embedded in the IB remain live
   for the same interval as the launch. Existing graph update locking must
   prevent in-place kernarg reuse while an earlier launch is active.
5. HIP submits one ROCR-materialized vendor AQL packet through its existing
   stream queue. A prepared list must not create an untracked application KFD
   queue.
6. Unsupported packet types, ABI properties, dependency shapes, profiler
   states, or architectures fall back to ordinary AQL before any PM4 work is
   submitted.
7. Failure after publication is a launch failure, never an AQL fallback; a
   fallback after partial submission could execute a graph twice.

Dependency contract
-------------------

The conservative serial policy places a correctness boundary before every
dispatch after the first. For GFX11, the accepted boundary is:

1. ``CS_PARTIAL_FLUSH`` / compute-idle;
2. ``ACQUIRE_MEM`` with same-agent scalar/vector read-cache invalidation
   (GCR ``0x00380``);
3. the consumer dispatch.

The command list begins with a system acquire so host-side kernarg updates are
visible on every replay and ends with compute-idle before the outer vendor
packet can publish completion.

GFX12 uses its separate register map and GCR encoding. GFX11 PM4 words must
never be selected by a GFX12 prefix check, or vice versa. A later
resource-aware policy may use a narrower VMEM-only boundary only when verified
kernel metadata proves the consumer's cache class. Unknown metadata always
uses the fail-closed scalar/vector boundary.

Target ROCR abstraction
-----------------------

The first shippable interface should remain experimental/private while its ABI
is being exercised. The intended operations are:

``query_capability``
  Return the encoder family, supported ABI features, and runtime enablement
  state for an agent. Compile support and hardware-qualified enablement are
  reported separately.

``prepare``
  Validate a stable array of captured kernel-dispatch packets plus explicit
  dependency classes, lower it with the agent's encoder, allocate executable
  storage, and return an opaque handle. No raw PM4 address is returned.

``materialize_packet``
  Fill a 64-byte vendor AQL packet for a prepared handle using caller-supplied
  fence scopes and completion signal. HIP publishes this packet through its
  normal queue path.

``materialize_packet_for_queue``
  Bind a prepared handle to the destination AQL queue before filling the vendor
  packet. Command lists that contain private-segment dispatches require this
  operation so ROCR can validate and retain the queue's scratch allocation.

``destroy``
  Release the executable allocation after the caller has proved that every
  launch retaining the handle completed.

Re-preparing on graph update is the initial correctness-first behavior. An
in-place patch API may be added only after packet offsets, concurrent launches,
and kernarg lifetime have permanent tests.

Queue scratch contract
----------------------

Private segments are queue state, not agent or graph state. A retained list
records the largest wave32 and wave64 private-segment requirements separately.
At queue materialization ROCR:

1. unwraps tool intercept layers and verifies that the destination is an AQL
   queue owned by the same GPU agent;
2. verifies that the queue has a stable main-scratch allocation whose bytes per
   wave cover every retained dispatch;
3. acquires a shared lease that prevents asynchronous main-scratch reclaim
   while the queue-specific command-list binding exists; and
4. clones the immutable command-list template once per queue ID and patches the
   clone with that queue's ``COMPUTE_TMPRING_SIZE`` value.

The graph-level template is never patched in place, so simultaneous launches
on queues with different scratch configurations cannot race. Queue IDs are
unique for their lifetime and each queue-specific executable clone remains
owned by the prepared list until its final in-flight launch completes.
The binding releases its scratch lease when the prepared list is destroyed;
the lease state is reference counted independently of the queue object, so a
queue may be destroyed before a graph that was previously materialized on it.

ROCR does not allocate a second graph-private scratch pool. HIP compute queues
normally start with reusable main scratch, and the retained path reuses that
same allocation. A queue with no main scratch, an insufficient bytes-per-wave
allocation, or single-use large scratch fails materialization before the
vendor packet is published; HIP may then submit the original AQL batch. The
legacy queue-less materialization entry point remains valid only for lists that
do not require scratch.

Dynamic call stacks remain unsupported. Kernels that require the legacy
private-segment-buffer user-SGPR descriptor also remain fail-closed; GFX11/12
architected-flat-scratch kernels that describe private storage only through the
AQL packet are supported. Per-ASIC qualification must include private-memory
correctness, two-queue concurrent launch, graph destruction, and graph-memory
trim before scratch is considered enabled by default on that ASIC.

Architecture matrix
-------------------

The source tree currently declares these concrete targets:

.. list-table:: Planned encoder and enablement coverage
   :header-rows: 1

   * - Family
     - Targets
     - Encoder
     - Initial runtime state
   * - GFX11 RDNA3
     - ``gfx1100``, ``gfx1101``, ``gfx1102``, ``gfx1103``
     - Legacy GFX10/GFX11 register map
     - Disabled until per-ASIC qualification
   * - GFX11 RDNA3.5
     - ``gfx1150``, ``gfx1151``, ``gfx1152``, ``gfx1153``
     - Legacy GFX10/GFX11 register map
     - ``gfx1151`` qualified; others disabled
   * - GFX12 RDNA4
     - ``gfx1200``, ``gfx1201``
     - Separate GFX12 register map, implemented
     - Disabled until per-ASIC qualification
   * - GFX12.5
     - ``gfx1250``
     - GFX12-family encoder implemented; explicit 12.5 review still required
     - Disabled until hardware qualification

Generic targets are compile targets, not runtime qualification identities.
Unknown steppings remain disabled even if their major version is 11 or 12.
Unqualified concrete targets can be exercised only through the second debug
gate described above.

Qualification gates
-------------------

Every concrete ASIC requires all of the following before runtime enablement:

1. CPU golden tests for emitted PM4 words, compared with the matching Redline
   family encoder and reviewed against AMD packet/register definitions.
2. Exact 512-dispatch non-atomic RMW on ordinary AQL and retained PM4.
3. Diamond, fan-out/fan-in, disabled-node, and cross-stream fallback tests.
4. Graph-exec parameter update, packet-count change, graph-memory trim, and
   immediate asynchronous graph destroy.
5. Concurrent launches of one graph exec on multiple streams with independent
   storage, followed by update and destroy stress.
6. Scratch, dynamic LDS, implicit SGPR, kernarg preload, cooperative dispatch,
   and partial-workgroup cases either pass dedicated tests or explicitly fall
   back.
7. Profiler start/stop and activity tracing prove ordinary-AQL fallback without
   losing per-kernel records.
8. Full-model logits or intermediate tensors match the ordinary-AQL reference.
9. Locked, counterbalanced AQL/PM4/Vulkan performance reproduces a useful win.
10. Repeated create/launch/update/destroy soak and injected failure tests
    complete without VM faults, queue-removal failures, GPU resets, or leaked
    executable allocations.

Rollout policy
--------------

The feature remains opt-in while any target is in qualification. Runtime logs
must distinguish: selected, compile-supported but unqualified, unsupported
packet shape, unsupported kernel ABI, profiler fallback, and submission
failure. A future beta policy may enable only an explicit allowlist of
qualified ASIC and firmware combinations. Making retained PM4 the default is a
separate decision after multi-device soak testing and fault-recovery review.
