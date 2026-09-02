/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// End-to-end data-correctness test for the single-stream collapse heuristic.
//
// The collapse heuristic (DEBUG_HIP_GRAPH_MIN_OVERLAP) can silently fold a
// multi-stream graph onto one in-order stream. This test builds a diamond graph
// with real cross-branch data dependencies and verifies the output is correct
// regardless of whether collapse fires, so a dropped/mis-ordered dependency
// would corrupt the result and fail the test (not just check liveness).
//
// To cover both code paths, run the binary twice:
//   DEBUG_HIP_GRAPH_MIN_OVERLAP=0 ./GraphPerformance "Performance_Graph_CollapseCorrectness"
//   DEBUG_HIP_GRAPH_MIN_OVERLAP=2 ./GraphPerformance "Performance_Graph_CollapseCorrectness"
// (the flag is read once at hipGraphInstantiate, hence separate processes).

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr int kN = 4096;          // small kernels so the gate may collapse them
constexpr int kBlock = 256;
constexpr int kIters = 200;       // many launches to surface nondeterministic races
constexpr int kPoison = -1;       // a stale (pre-dependency) read yields this

__global__ void k_set(int* a, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = val;
}

__global__ void k_mul(const int* in, int* out, int factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * factor;
}

__global__ void k_add(const int* x, const int* y, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i];
}

__global__ void k_delay_set(int* output, int index, uint64_t count) {
#if HT_AMD
  const uint64_t start = wall_clock64();
  while (wall_clock64() - start < count) {
  }
#else
  const uint64_t start = clock64();
  while (clock64() - start < count) {
  }
#endif
  output[index] = 1;
}

__global__ void k_increment(int* value) { value[0] = value[0] + 1; }

__global__ void k_add_value(int* value, int addend) { value[0] = value[0] + addend; }

__global__ void k_atomic_increment(int* value) { atomicAdd(value, 1); }

__global__ void k_private_increment(int* value, int selector) {
  volatile int scratch[8];
  const int slot = (selector + static_cast<int>(threadIdx.x)) & 7;
  scratch[slot] = value[0] + 1;
  if (threadIdx.x == 0) {
    value[0] = scratch[slot];
  }
}

__global__ void k_private_atomic_increment(int* value, int selector) {
  volatile int scratch[8];
  const int slot = (selector + static_cast<int>(threadIdx.x)) & 7;
  scratch[slot] = 1;
  if (threadIdx.x == 0) {
    atomicAdd(value, scratch[slot]);
  }
}

hipGraphNode_t AddKernel(hipGraph_t graph, std::vector<hipGraphNode_t> deps,
                         void* func, std::vector<void*> args) {
  hipKernelNodeParams p{};
  p.func = func;
  p.gridDim = dim3((kN + kBlock - 1) / kBlock, 1, 1);
  p.blockDim = dim3(kBlock, 1, 1);
  p.sharedMemBytes = 0;
  p.kernelParams = args.data();
  p.extra = nullptr;
  hipGraphNode_t node{};
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps.empty() ? nullptr : deps.data(),
                                  deps.size(), &p));
  return node;
}

}  // namespace

/**
 * Diamond DAG:  a=set(1) -> b=a*2 (branch 1)
 *                        -> c=a*3 (branch 2)
 *               out = b + c  (join, depends on both branches)
 * Correct result is out[i] == 5 for all i. Buffers are poisoned before each
 * launch, so any node running before its producer reads kPoison and fails.
 */
TEST_CASE("Performance_Graph_CollapseCorrectness") {
  const char* env = std::getenv("DEBUG_HIP_GRAPH_MIN_OVERLAP");
  INFO("DEBUG_HIP_GRAPH_MIN_OVERLAP=" << (env ? env : "(default)"));

  const size_t bytes = static_cast<size_t>(kN) * sizeof(int);
  int *a = nullptr, *b = nullptr, *c = nullptr, *out = nullptr;
  HIP_CHECK(hipMalloc(&a, bytes));
  HIP_CHECK(hipMalloc(&b, bytes));
  HIP_CHECK(hipMalloc(&c, bytes));
  HIP_CHECK(hipMalloc(&out, bytes));

  int one = 1, two = 2, three = 3, n = kN;
  void* a_p = a;
  void* b_p = b;
  void* c_p = c;
  void* out_p = out;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t n_set =
      AddKernel(graph, {}, reinterpret_cast<void*>(k_set), {&a_p, &one, &n});
  hipGraphNode_t n_b = AddKernel(graph, {n_set}, reinterpret_cast<void*>(k_mul),
                                 {&a_p, &b_p, &two, &n});
  hipGraphNode_t n_c = AddKernel(graph, {n_set}, reinterpret_cast<void*>(k_mul),
                                 {&a_p, &c_p, &three, &n});
  AddKernel(graph, {n_b, n_c}, reinterpret_cast<void*>(k_add),
            {&b_p, &c_p, &out_p, &n});

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<int> host(kN);
  for (int iter = 0; iter < kIters; ++iter) {
    // Poison every buffer so a stale read (wrong order) is observable.
    HIP_CHECK(hipMemsetAsync(a, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(b, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(c, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(out, 0xFF, bytes, stream));

    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipMemcpyAsync(host.data(), out, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int bad = -1;
    for (int i = 0; i < kN; ++i) {
      if (host[i] != 5) { bad = i; break; }
    }
    INFO("iter=" << iter << " first_bad_index=" << bad
                 << " value=" << (bad >= 0 ? host[bad] : 5)
                 << " (poison=" << kPoison << ")");
    REQUIRE(bad == -1);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
  HIP_CHECK(hipFree(out));
}

TEST_CASE("Performance_Graph_DisabledBoundarySynchronization") {
  const size_t bytes = static_cast<size_t>(kN) * sizeof(int);
  int *a = nullptr, *b = nullptr, *c = nullptr, *out = nullptr;
  HIP_CHECK(hipMalloc(&a, bytes));
  HIP_CHECK(hipMalloc(&b, bytes));
  HIP_CHECK(hipMalloc(&c, bytes));
  HIP_CHECK(hipMalloc(&out, bytes));

  int one = 1, two = 2, three = 3, four = 4, n = kN;
  void* a_p = a;
  void* b_p = b;
  void* c_p = c;
  void* out_p = out;

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t set_node =
      AddKernel(graph, {}, reinterpret_cast<void*>(k_set), {&a_p, &one, &n});
  hipGraphNode_t b_node = AddKernel(graph, {set_node}, reinterpret_cast<void*>(k_mul),
                                    {&a_p, &b_p, &two, &n});
  hipGraphNode_t c_first = AddKernel(graph, {set_node}, reinterpret_cast<void*>(k_mul),
                                     {&a_p, &c_p, &three, &n});
  hipGraphNode_t c_last = AddKernel(graph, {c_first}, reinterpret_cast<void*>(k_mul),
                                    {&a_p, &c_p, &four, &n});
  AddKernel(graph, {b_node, c_last}, reinterpret_cast<void*>(k_add),
            {&b_p, &c_p, &out_p, &n});

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  int expected = 0;
  SECTION("disable first consumer packet") {
    HIP_CHECK(hipGraphNodeSetEnabled(exec, c_first, 0));
    expected = 6;
  }
  SECTION("disable last producer packet") {
    HIP_CHECK(hipGraphNodeSetEnabled(exec, c_last, 0));
    expected = 5;
  }
  SECTION("disable all packets in producer segment") {
    HIP_CHECK(hipGraphNodeSetEnabled(exec, c_first, 0));
    HIP_CHECK(hipGraphNodeSetEnabled(exec, c_last, 0));
    expected = 1;
  }

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  std::vector<int> host(kN);
  for (int iteration = 0; iteration < kIters; ++iteration) {
    HIP_CHECK(hipMemsetAsync(a, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(b, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(c, 0xFF, bytes, stream));
    HIP_CHECK(hipMemsetAsync(out, 0xFF, bytes, stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipMemcpyAsync(host.data(), out, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(std::all_of(host.begin(), host.end(),
                        [expected](int value) { return value == expected; }));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
  HIP_CHECK(hipFree(out));
}

TEST_CASE("Performance_Graph_CollapsedIndependentCompletion") {
  constexpr int kNodes = 32;
  constexpr int kIterations = 20;
  int clock_rate = 0;
  hipDevice_t device = 0;
  HIP_CHECK(hipGetDevice(&device));
#if HT_AMD
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeWallClockRate, device));
#else
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, device));
#endif
  const uint64_t delay = static_cast<uint64_t>(clock_rate) / 2;

  int* output = nullptr;
  HIP_CHECK(hipMalloc(&output, kNodes * sizeof(int)));
  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  for (int i = 0; i < kNodes; ++i) {
    uint64_t count = i + 1 == kNodes ? 0 : delay;
    void* output_arg = output;
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(k_delay_set);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    void* args[] = {&output_arg, &i, &count};
    params.kernelParams = args;
    hipGraphNode_t node{};
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &params));
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  std::vector<int> host(kNodes);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    HIP_CHECK(hipMemsetAsync(output, 0, kNodes * sizeof(int), stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(
        hipMemcpyAsync(host.data(), output, kNodes * sizeof(int), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(std::all_of(host.begin(), host.end(), [](int value) { return value == 1; }));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(output));
}

TEST_CASE("Performance_Graph_RmwAndAsyncDestroy") {
  constexpr int kNodes = 512;
  constexpr int kLaunches = 64;

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t dependency = nullptr;
  void* value_arg = device_value;
  void* kernel_args[] = {&value_arg};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(k_increment);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.kernelParams = kernel_args;

  for (int i = 0; i < kNodes; ++i) {
    hipGraphNode_t node{};
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr,
                                    dependency ? 1 : 0, &params));
    dependency = node;
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemsetAsync(device_value, 0, sizeof(*device_value), stream));

  for (int i = 0; i < kLaunches; ++i) {
    HIP_CHECK(hipGraphLaunch(exec, stream));
  }

  // Queued graph work must remain valid through the following ordered copy.
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));

  int host_value = 0;
  HIP_CHECK(
      hipMemcpyAsync(&host_value, device_value, sizeof(host_value), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(host_value == kNodes * kLaunches);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_value));
}

TEST_CASE("Performance_Graph_RmwUpdateAndDisable") {
  constexpr int kNodes = 128;
  constexpr int kIterations = 32;
  constexpr int kUpdatedNode = kNodes / 2;

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<hipGraphNode_t> nodes;
  nodes.reserve(kNodes);
  hipGraphNode_t dependency = nullptr;
  void* value_arg = device_value;
  int addend = 1;
  void* kernel_args[] = {&value_arg, &addend};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(k_add_value);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.kernelParams = kernel_args;

  for (int i = 0; i < kNodes; ++i) {
    hipGraphNode_t node{};
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr,
                                    dependency ? 1 : 0, &params));
    nodes.push_back(node);
    dependency = node;
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  auto run_and_check = [&](int expected) {
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      HIP_CHECK(hipMemsetAsync(device_value, 0, sizeof(*device_value), stream));
      HIP_CHECK(hipGraphLaunch(exec, stream));
      int actual = 0;
      HIP_CHECK(hipMemcpyAsync(&actual, device_value, sizeof(actual), hipMemcpyDeviceToHost,
                               stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      REQUIRE(actual == expected);
    }
  };

  run_and_check(kNodes);

  int updated_addend = 7;
  void* updated_args[] = {&value_arg, &updated_addend};
  hipKernelNodeParams updated_params = params;
  updated_params.kernelParams = updated_args;
  HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, nodes[kUpdatedNode], &updated_params));
  run_and_check(kNodes + updated_addend - addend);

  HIP_CHECK(hipGraphNodeSetEnabled(exec, nodes[kUpdatedNode], 0));
  run_and_check(kNodes - 1);
  HIP_CHECK(hipGraphNodeSetEnabled(exec, nodes[kUpdatedNode], 1));
  run_and_check(kNodes + updated_addend - addend);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device_value));
}

TEST_CASE("Performance_Graph_Pm4ConcurrentLaunchAndTrim") {
  constexpr int kNodes = 64;
  constexpr int kLaunches = 32;

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t dependency = nullptr;
  void* value_arg = device_value;
  void* kernel_args[] = {&value_arg};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(k_atomic_increment);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.kernelParams = kernel_args;
  for (int i = 0; i < kNodes; ++i) {
    hipGraphNode_t node{};
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr,
                                    dependency ? 1 : 0, &params));
    dependency = node;
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t streams[2]{};
  HIP_CHECK(hipStreamCreate(&streams[0]));
  HIP_CHECK(hipStreamCreate(&streams[1]));

  for (int i = 0; i < kLaunches; ++i) {
    HIP_CHECK(hipGraphLaunch(exec, streams[i & 1]));
  }
  hipError_t trim_status = hipSuccess;
  std::thread trimmer([&] {
    for (int i = 0; i < 16; ++i) {
      const hipError_t status = hipDeviceGraphMemTrim(0);
      if (status != hipSuccess) {
        trim_status = status;
        return;
      }
    }
  });

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  trimmer.join();
  HIP_CHECK(trim_status);
  HIP_CHECK(hipStreamSynchronize(streams[0]));
  HIP_CHECK(hipStreamSynchronize(streams[1]));

  int actual = 0;
  HIP_CHECK(hipMemcpy(&actual, device_value, sizeof(actual), hipMemcpyDeviceToHost));
  REQUIRE(actual == kNodes * kLaunches);

  HIP_CHECK(hipStreamDestroy(streams[0]));
  HIP_CHECK(hipStreamDestroy(streams[1]));
  HIP_CHECK(hipFree(device_value));
}

TEST_CASE("Performance_Graph_Pm4QueueScratch") {
  constexpr int kNodes = 8;

  hipFuncAttributes attributes{};
  HIP_CHECK(hipFuncGetAttributes(&attributes,
                                 reinterpret_cast<const void*>(k_private_increment)));
  INFO("private bytes=" << attributes.localSizeBytes);
  REQUIRE(attributes.localSizeBytes > 0);

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t dependency = nullptr;
  void* value_arg = device_value;
  int selector = 3;
  void* kernel_args[] = {&value_arg, &selector};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(k_private_increment);
  params.gridDim = dim3(1);
  params.blockDim = dim3(32);
  params.kernelParams = kernel_args;
  for (int i = 0; i < kNodes; ++i) {
    hipGraphNode_t node{};
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr,
                                    dependency ? 1 : 0, &params));
    dependency = node;
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));

  int actual = 0;
  HIP_CHECK(hipMemcpyAsync(&actual, device_value, sizeof(actual), hipMemcpyDeviceToHost,
                           stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(actual == kNodes);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device_value));
}

TEST_CASE("Performance_Graph_Pm4QueueScratchConcurrentLaunchAndTrim") {
  constexpr int kNodes = 32;
  constexpr int kLaunches = 16;

  hipFuncAttributes attributes{};
  HIP_CHECK(hipFuncGetAttributes(
      &attributes, reinterpret_cast<const void*>(k_private_atomic_increment)));
  REQUIRE(attributes.localSizeBytes > 0);

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t dependency = nullptr;
  void* value_arg = device_value;
  int selector = 5;
  void* kernel_args[] = {&value_arg, &selector};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(k_private_atomic_increment);
  params.gridDim = dim3(1);
  params.blockDim = dim3(32);
  params.kernelParams = kernel_args;
  for (int i = 0; i < kNodes; ++i) {
    hipGraphNode_t node{};
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr,
                                    dependency ? 1 : 0, &params));
    dependency = node;
  }

  hipGraphExec_t exec{};
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t streams[2]{};
  HIP_CHECK(hipStreamCreate(&streams[0]));
  HIP_CHECK(hipStreamCreate(&streams[1]));
  for (int i = 0; i < kLaunches; ++i) {
    HIP_CHECK(hipGraphLaunch(exec, streams[i & 1]));
  }

  hipError_t trim_status = hipSuccess;
  std::thread trimmer([&] { trim_status = hipDeviceGraphMemTrim(0); });
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  trimmer.join();
  HIP_CHECK(trim_status);
  HIP_CHECK(hipStreamSynchronize(streams[0]));
  HIP_CHECK(hipStreamSynchronize(streams[1]));

  int actual = 0;
  HIP_CHECK(hipMemcpy(&actual, device_value, sizeof(actual), hipMemcpyDeviceToHost));
  REQUIRE(actual == kNodes * kLaunches);

  HIP_CHECK(hipStreamDestroy(streams[0]));
  HIP_CHECK(hipStreamDestroy(streams[1]));
  HIP_CHECK(hipFree(device_value));
}
