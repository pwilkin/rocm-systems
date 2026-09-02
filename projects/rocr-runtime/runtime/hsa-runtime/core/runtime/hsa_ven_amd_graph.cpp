/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
 * SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
 *
 * The GFX11 register map and dependency sequence are adapted from
 * redline-rocr at pwilkin/redline 20474b8c1a5b.
 */

#include "inc/hsa_ven_amd_graph.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/inc/agent.h"
#include "core/inc/amd_aql_queue.h"
#include "core/inc/amd_graph_command_encoder.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/hsa_ven_amd_loader_impl.h"
#include "core/inc/intercept_queue.h"
#include "core/inc/memory_region.h"
#include "core/inc/runtime.h"
#include "inc/hsa_ext_amd.h"

namespace rocr {
namespace AMD {
hsa_status_t handleException();
}

namespace graph {
namespace {

constexpr uint16_t kPm4IndirectBufferFormat = 1;
constexpr uint32_t kPacket3IndirectBuffer = 0x3f;
constexpr uint32_t kIbValid = 1u << 23;
constexpr uint32_t kIbTemporalLu = 3u << 28;

class CommandList {
 public:
  struct QueueBinding {
    void* ib;
    std::shared_ptr<std::atomic<uint32_t>> scratch_users;
  };

  CommandList(AMD::GpuAgent* agent, void* ib, uint32_t dwords, uint32_t dispatches,
              hsa_ven_amd_graph_encoder_family_t family, size_t tmpring_patch_dword,
              uint32_t private_wave32, uint32_t private_wave64)
      : agent_(agent),
        ib_(ib),
        dwords_(dwords),
        dispatches_(dispatches),
        family_(family),
        tmpring_patch_dword_(tmpring_patch_dword),
        private_wave32_(private_wave32),
        private_wave64_(private_wave64) {}

  ~CommandList() {
    for (const auto& [queue_id, binding] : queue_ibs_) {
      static_cast<void>(queue_id);
      agent_->system_deallocator()(binding.ib);
      binding.scratch_users->fetch_sub(1, std::memory_order_release);
    }
    agent_->system_deallocator()(ib_);
  }

  AMD::GpuAgent* agent() const { return agent_; }
  void* ib() const { return ib_; }
  uint32_t dwords() const { return dwords_; }
  uint32_t dispatches() const { return dispatches_; }
  hsa_ven_amd_graph_encoder_family_t family() const { return family_; }
  uint32_t max_private_segment_size() const {
    return std::max(private_wave32_, private_wave64_);
  }
  uint32_t private_wave32() const { return private_wave32_; }
  uint32_t private_wave64() const { return private_wave64_; }
  bool requires_scratch() const { return private_wave32_ != 0 || private_wave64_ != 0; }

  void* QueueIb(uint64_t queue_id, uint32_t compute_tmpring_size,
                const std::shared_ptr<std::atomic<uint32_t>>& scratch_users) {
    if (!requires_scratch()) {
      return ib_;
    }
    if (scratch_users == nullptr) {
      return nullptr;
    }
    const auto existing = queue_ibs_.find(queue_id);
    if (existing != queue_ibs_.end()) {
      scratch_users->fetch_sub(1, std::memory_order_release);
      return existing->second.ib;
    }
    if (compute_tmpring_size == 0 || tmpring_patch_dword_ >= dwords_) {
      scratch_users->fetch_sub(1, std::memory_order_release);
      return nullptr;
    }

    const size_t bytes = static_cast<size_t>(dwords_) * sizeof(uint32_t);
    void* binding =
        agent_->system_allocator()(bytes, 4096, core::MemoryRegion::AllocateExecutable);
    if (binding == nullptr) {
      scratch_users->fetch_sub(1, std::memory_order_release);
      return nullptr;
    }
    std::memcpy(binding, ib_, bytes);
    static_cast<uint32_t*>(binding)[tmpring_patch_dword_] = compute_tmpring_size;
    std::atomic_thread_fence(std::memory_order_release);
    try {
      queue_ibs_.emplace(queue_id, QueueBinding{binding, scratch_users});
    } catch (...) {
      agent_->system_deallocator()(binding);
      scratch_users->fetch_sub(1, std::memory_order_release);
      throw;
    }
    return binding;
  }

 private:
  AMD::GpuAgent* agent_;
  void* ib_;
  uint32_t dwords_;
  uint32_t dispatches_;
  hsa_ven_amd_graph_encoder_family_t family_;
  size_t tmpring_patch_dword_;
  uint32_t private_wave32_;
  uint32_t private_wave64_;
  std::unordered_map<uint64_t, QueueBinding> queue_ibs_;
};

std::mutex command_lists_mutex;
std::unordered_set<CommandList*> command_lists;

bool DiagnosticsEnabled() {
  const char* value = std::getenv("HSA_GRAPH_COMMAND_LIST_DIAGNOSTICS");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

template <typename Image>
void PrintEncoderRejection(size_t index, hsa_ven_amd_graph_encoder_family_t family,
                           const hsa_kernel_dispatch_packet_t& packet,
                           const Image& image, uint32_t kernel_flags,
                           hsa_ven_amd_graph_dependency_t dependency) {
  if (!DiagnosticsEnabled()) {
    return;
  }
  std::fprintf(
      stderr,
      "[ROCR graph PM4] reject packet=%zu family=%u kernel=0x%016llx entry=0x%016llx "
      "private=%u dynamic=%u properties=0x%04x preload=%u grid=%ux%ux%u "
      "workgroup=%ux%ux%u group=%u kernarg=%p dependency=%u flags=0x%x\n",
      index, static_cast<unsigned>(family),
      static_cast<unsigned long long>(packet.kernel_object),
      static_cast<unsigned long long>(image.code_entry), packet.private_segment_size,
      (kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK) != 0,
      image.kernel_code_properties, image.kernarg_preload_length, packet.grid_size_x,
      packet.grid_size_y, packet.grid_size_z, packet.workgroup_size_x,
      packet.workgroup_size_y, packet.workgroup_size_z, packet.group_segment_size,
      packet.kernarg_address, static_cast<unsigned>(dependency), kernel_flags);
}

bool GetCapability(AMD::GpuAgent* agent, hsa_ven_amd_graph_capabilities_t* capabilities) {
  if (agent->supported_isas().empty()) {
    return false;
  }
  const auto* isa = agent->supported_isas()[0];
  const auto capability = GetGraphCommandCapability(
      isa->GetMajorVersion(), isa->GetMinorVersion(), isa->GetStepping());
  capabilities->encoder_family = capability.family;
  capabilities->flags = 0;
  if (capability.compile_supported) {
    capabilities->flags |= HSA_VEN_AMD_GRAPH_CAPABILITY_COMPILE_SUPPORTED;
  }
  if (capability.runtime_qualified) {
    capabilities->flags |= HSA_VEN_AMD_GRAPH_CAPABILITY_RUNTIME_QUALIFIED;
  }
  return true;
}

CommandList* FindCommandList(hsa_ven_amd_graph_command_list_t handle) {
  auto* command_list = reinterpret_cast<CommandList*>(handle.handle);
  return command_lists.count(command_list) != 0 ? command_list : nullptr;
}

AMD::AqlQueue* UnwrapAqlQueue(hsa_queue_t* queue) {
  core::Queue* core_queue = core::Queue::Convert(queue);
  if (core_queue == nullptr || !core_queue->IsValid()) {
    return nullptr;
  }
  while (core::InterceptQueue::IsType(core_queue)) {
    core_queue = static_cast<core::InterceptQueue*>(core_queue)->wrapped.get();
    if (core_queue == nullptr || !core_queue->IsValid()) {
      return nullptr;
    }
  }
  return AMD::AqlQueue::IsType(core_queue) ? static_cast<AMD::AqlQueue*>(core_queue)
                                           : nullptr;
}

bool LoadGfx11KernelImage(const hsa_kernel_dispatch_packet_t& packet,
                          Gfx11KernelImage* image) {
  const void* host_address = nullptr;
  if (image == nullptr ||
      rocr::hsa_ven_amd_loader_query_host_address(
          reinterpret_cast<const void*>(packet.kernel_object), &host_address) !=
          HSA_STATUS_SUCCESS ||
      host_address == nullptr) {
    return false;
  }
  constexpr size_t kDescriptorOffset = 16;
  const auto* descriptor = reinterpret_cast<const hsa_amd_metadata_kernel_descriptor_t*>(
      static_cast<const uint8_t*>(host_address) + kDescriptorOffset);
  const int64_t entry_offset = descriptor->kernel_code_entry_byte_offset;
  uint64_t code_entry = packet.kernel_object;
  if (entry_offset >= 0) {
    if (code_entry > UINT64_MAX - static_cast<uint64_t>(entry_offset)) {
      return false;
    }
    code_entry += static_cast<uint64_t>(entry_offset);
  } else {
    const uint64_t magnitude = static_cast<uint64_t>(-(entry_offset + 1)) + 1;
    if (code_entry < magnitude) {
      return false;
    }
    code_entry -= magnitude;
  }
  *image = {
      code_entry,
      descriptor->compute_pgm_rsrc1,
      descriptor->compute_pgm_rsrc2,
      descriptor->compute_pgm_rsrc3,
      descriptor->kernel_code_properties,
      descriptor->kernarg_preload.length,
  };
  return true;
}

bool LoadGfx12KernelImage(const hsa_kernel_dispatch_packet_t& packet,
                          Gfx12KernelImage* image) {
  Gfx11KernelImage legacy{};
  if (image == nullptr || !LoadGfx11KernelImage(packet, &legacy)) {
    return false;
  }
  *image = {
      legacy.code_entry,
      legacy.compute_pgm_rsrc1,
      legacy.compute_pgm_rsrc2,
      legacy.compute_pgm_rsrc3,
      legacy.kernel_code_properties,
      legacy.kernarg_preload_length,
  };
  return true;
}

}  // namespace

hsa_status_t GetCapabilities(hsa_agent_t hsa_agent,
                             hsa_ven_amd_graph_capabilities_t* capabilities) {
  if (capabilities == nullptr ||
      capabilities->struct_size < sizeof(hsa_ven_amd_graph_capabilities_t)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  core::Agent* agent = core::Agent::Convert(hsa_agent);
  if (agent == nullptr || !agent->IsValid() ||
      agent->device_type() != core::Agent::kAmdGpuDevice) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  auto* gpu_agent = static_cast<AMD::GpuAgent*>(agent);
  std::memset(reinterpret_cast<uint8_t*>(capabilities) + offsetof(
                  hsa_ven_amd_graph_capabilities_t, version_major),
              0, sizeof(*capabilities) -
                     offsetof(hsa_ven_amd_graph_capabilities_t, version_major));
  capabilities->struct_size = sizeof(*capabilities);
  capabilities->version_major = HSA_VEN_AMD_GRAPH_VERSION_MAJOR;
  capabilities->version_minor = HSA_VEN_AMD_GRAPH_VERSION_MINOR;
  return GetCapability(gpu_agent, capabilities) ? HSA_STATUS_SUCCESS
                                                 : HSA_STATUS_ERROR_INVALID_ISA;
}

hsa_status_t Create(hsa_agent_t hsa_agent,
                    const hsa_ven_amd_graph_command_list_desc_t* desc,
                    hsa_ven_amd_graph_command_list_t* handle) {
  if (desc == nullptr || handle == nullptr ||
      desc->struct_size < sizeof(hsa_ven_amd_graph_command_list_desc_t) ||
      desc->packets == nullptr || desc->packet_count < 2 ||
      (desc->flags & ~HSA_VEN_AMD_GRAPH_COMMAND_LIST_ALLOW_UNQUALIFIED) != 0 ||
      desc->kernel_flags == nullptr || desc->kernel_flag_count != desc->packet_count ||
      desc->dependency_count != desc->packet_count - 1 ||
      (desc->dependency_count != 0 && desc->dependencies == nullptr)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  handle->handle = 0;

  core::Agent* agent = core::Agent::Convert(hsa_agent);
  if (agent == nullptr || !agent->IsValid() ||
      agent->device_type() != core::Agent::kAmdGpuDevice) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  auto* gpu_agent = static_cast<AMD::GpuAgent*>(agent);
  hsa_ven_amd_graph_capabilities_t capabilities{};
  capabilities.struct_size = sizeof(capabilities);
  const bool allow_unqualified =
      (desc->flags & HSA_VEN_AMD_GRAPH_COMMAND_LIST_ALLOW_UNQUALIFIED) != 0;
  if (!GetCapability(gpu_agent, &capabilities) ||
      (capabilities.flags & HSA_VEN_AMD_GRAPH_CAPABILITY_COMPILE_SUPPORTED) == 0 ||
      ((capabilities.flags & HSA_VEN_AMD_GRAPH_CAPABILITY_RUNTIME_QUALIFIED) == 0 &&
       !allow_unqualified) ||
      capabilities.encoder_family == HSA_VEN_AMD_GRAPH_ENCODER_NONE) {
    return HSA_STATUS_ERROR_INVALID_ISA;
  }

  auto validate_packet = [](const hsa_kernel_dispatch_packet_t& packet) {
    const uint8_t type =
        (packet.header >> HSA_PACKET_HEADER_TYPE) &
        ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
    return type == HSA_PACKET_TYPE_KERNEL_DISPATCH && packet.completion_signal.handle == 0;
  };
  std::vector<uint32_t> words;
  uint32_t dispatch_count = 0;
  size_t tmpring_patch_dword = static_cast<size_t>(-1);
  uint32_t private_wave32 = 0;
  uint32_t private_wave64 = 0;
  constexpr uint16_t kEnableWavefrontSize32 = 1u << 10;
  if (capabilities.encoder_family == HSA_VEN_AMD_GRAPH_ENCODER_GFX11) {
    Gfx11CommandEncoder builder;
    for (size_t i = 0; i < desc->packet_count; ++i) {
      const auto& packet = desc->packets[i];
      if (!validate_packet(packet)) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
      const auto dependency = i == 0 ? HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW
                                     : desc->dependencies[i - 1];
      Gfx11KernelImage image{};
      if (!LoadGfx11KernelImage(packet, &image)) {
        if (DiagnosticsEnabled()) {
          std::fprintf(stderr,
                       "[ROCR graph PM4] reject packet=%zu family=%u kernel=0x%016llx "
                       "reason=loader-descriptor\n",
                       i, static_cast<unsigned>(capabilities.encoder_family),
                       static_cast<unsigned long long>(packet.kernel_object));
        }
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if (!builder.Append(packet, image, desc->kernel_flags[i], dependency)) {
        PrintEncoderRejection(i, capabilities.encoder_family, packet, image,
                              desc->kernel_flags[i], dependency);
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if ((image.kernel_code_properties & kEnableWavefrontSize32) != 0) {
        private_wave32 = std::max(private_wave32, packet.private_segment_size);
      } else {
        private_wave64 = std::max(private_wave64, packet.private_segment_size);
      }
    }
    builder.Finish();
    words = builder.words();
    dispatch_count = static_cast<uint32_t>(builder.dispatch_count());
    tmpring_patch_dword = builder.tmpring_patch_dword();
  } else if (capabilities.encoder_family == HSA_VEN_AMD_GRAPH_ENCODER_GFX12) {
    Gfx12CommandEncoder builder;
    for (size_t i = 0; i < desc->packet_count; ++i) {
      const auto& packet = desc->packets[i];
      if (!validate_packet(packet)) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
      const auto dependency = i == 0 ? HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW
                                     : desc->dependencies[i - 1];
      Gfx12KernelImage image{};
      if (!LoadGfx12KernelImage(packet, &image)) {
        if (DiagnosticsEnabled()) {
          std::fprintf(stderr,
                       "[ROCR graph PM4] reject packet=%zu family=%u kernel=0x%016llx "
                       "reason=loader-descriptor\n",
                       i, static_cast<unsigned>(capabilities.encoder_family),
                       static_cast<unsigned long long>(packet.kernel_object));
        }
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if (!builder.Append(packet, image, desc->kernel_flags[i], dependency)) {
        PrintEncoderRejection(i, capabilities.encoder_family, packet, image,
                              desc->kernel_flags[i], dependency);
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if ((image.kernel_code_properties & kEnableWavefrontSize32) != 0) {
        private_wave32 = std::max(private_wave32, packet.private_segment_size);
      } else {
        private_wave64 = std::max(private_wave64, packet.private_segment_size);
      }
    }
    builder.Finish();
    words = builder.words();
    dispatch_count = static_cast<uint32_t>(builder.dispatch_count());
    tmpring_patch_dword = builder.tmpring_patch_dword();
  } else {
    return HSA_STATUS_ERROR_INVALID_ISA;
  }
  if (words.empty() || words.size() > 0x000fffff || tmpring_patch_dword >= words.size()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  const size_t bytes = words.size() * sizeof(uint32_t);
  void* ib = gpu_agent->system_allocator()(bytes, 4096, core::MemoryRegion::AllocateExecutable);
  if (ib == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  std::memcpy(ib, words.data(), bytes);
  std::atomic_thread_fence(std::memory_order_release);

  auto* command_list = new (std::nothrow)
      CommandList(gpu_agent, ib, static_cast<uint32_t>(words.size()),
                  dispatch_count, capabilities.encoder_family, tmpring_patch_dword,
                  private_wave32, private_wave64);
  if (command_list == nullptr) {
    gpu_agent->system_deallocator()(ib);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  {
    std::lock_guard<std::mutex> lock(command_lists_mutex);
    command_lists.insert(command_list);
  }
  handle->handle = reinterpret_cast<uint64_t>(command_list);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t Destroy(hsa_ven_amd_graph_command_list_t handle) {
  CommandList* command_list = nullptr;
  {
    std::lock_guard<std::mutex> lock(command_lists_mutex);
    command_list = FindCommandList(handle);
    if (command_list == nullptr) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    command_lists.erase(command_list);
  }
  delete command_list;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GetInfo(hsa_ven_amd_graph_command_list_t handle,
                     hsa_ven_amd_graph_command_list_info_t attribute, void* value) {
  if (value == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(command_lists_mutex);
  CommandList* command_list = FindCommandList(handle);
  if (command_list == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  switch (attribute) {
    case HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_DISPATCH_COUNT:
      *static_cast<uint32_t*>(value) = command_list->dispatches();
      return HSA_STATUS_SUCCESS;
    case HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_DWORD_COUNT:
      *static_cast<uint32_t*>(value) = command_list->dwords();
      return HSA_STATUS_SUCCESS;
    case HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_ENCODER_FAMILY:
      *static_cast<hsa_ven_amd_graph_encoder_family_t*>(value) =
          command_list->family();
      return HSA_STATUS_SUCCESS;
    case HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_MAX_PRIVATE_SEGMENT_SIZE:
      *static_cast<uint32_t*>(value) = command_list->max_private_segment_size();
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t Materialize(hsa_ven_amd_graph_command_list_t handle, hsa_queue_t* queue,
                         hsa_fence_scope_t acquire_scope,
                         hsa_fence_scope_t release_scope, uint32_t barrier,
                         hsa_signal_t completion_signal,
                         hsa_ven_amd_graph_materialized_packet_t* output) {
  if (output == nullptr || acquire_scope > HSA_FENCE_SCOPE_SYSTEM ||
      release_scope > HSA_FENCE_SCOPE_SYSTEM || barrier > 1) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(command_lists_mutex);
  CommandList* command_list = FindCommandList(handle);
  if (command_list == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  void* ib = command_list->ib();
  if (command_list->requires_scratch()) {
    AMD::AqlQueue* aql_queue = UnwrapAqlQueue(queue);
    if (aql_queue == nullptr || aql_queue->GetAgent() != command_list->agent()) {
      return HSA_STATUS_ERROR_INVALID_QUEUE;
    }
    uint32_t compute_tmpring_size = 0;
    std::shared_ptr<std::atomic<uint32_t>> scratch_users;
    const hsa_status_t scratch_status = aql_queue->GetGraphScratchState(
        command_list->private_wave32(), command_list->private_wave64(),
        &compute_tmpring_size, &scratch_users);
    if (scratch_status != HSA_STATUS_SUCCESS) {
      return scratch_status;
    }
    ib = command_list->QueueIb(aql_queue->public_handle()->id, compute_tmpring_size,
                               scratch_users);
    if (ib == nullptr) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
  }

  struct AqlPm4Ib {
    uint16_t header;
    uint16_t vendor_header;
    uint32_t indirect_buffer[4];
    uint32_t dword_count_remaining;
    uint32_t reserved[8];
    hsa_signal_t completion_signal;
  } packet{};
  static_assert(sizeof(packet) == 64, "vendor PM4 packet must be one AQL slot");

  packet.header =
      (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
      (barrier << HSA_PACKET_HEADER_BARRIER) |
      (acquire_scope << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (release_scope << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  packet.vendor_header = kPm4IndirectBufferFormat;
  const uint64_t address = reinterpret_cast<uint64_t>(ib);
  packet.indirect_buffer[0] =
      (3u << 30) | (2u << 16) | (kPacket3IndirectBuffer << 8);
  packet.indirect_buffer[1] = static_cast<uint32_t>(address) & 0xfffffffc;
  packet.indirect_buffer[2] = static_cast<uint32_t>(address >> 32);
  packet.indirect_buffer[3] = command_list->dwords() | kIbValid | kIbTemporalLu;
  packet.dword_count_remaining = 10;
  packet.completion_signal = completion_signal;

  std::memset(output, 0, sizeof(*output));
  std::memcpy(output->packet.bytes, &packet, sizeof(packet));
  output->full_header =
      static_cast<uint32_t>(packet.header) |
      (static_cast<uint32_t>(packet.vendor_header) << 16);
  return HSA_STATUS_SUCCESS;
}

}  // namespace graph
}  // namespace rocr

hsa_status_t HSA_API hsa_ven_amd_graph_get_capabilities(
    hsa_agent_t agent, hsa_ven_amd_graph_capabilities_t* capabilities) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    return rocr::graph::GetCapabilities(agent, capabilities);
  } catch (...) {
    return rocr::AMD::handleException();
  }
}

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_create(
    hsa_agent_t agent, const hsa_ven_amd_graph_command_list_desc_t* desc,
    hsa_ven_amd_graph_command_list_t* command_list) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    return rocr::graph::Create(agent, desc, command_list);
  } catch (...) {
    return rocr::AMD::handleException();
  }
}

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_destroy(
    hsa_ven_amd_graph_command_list_t command_list) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    return rocr::graph::Destroy(command_list);
  } catch (...) {
    return rocr::AMD::handleException();
  }
}

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_get_info(
    hsa_ven_amd_graph_command_list_t command_list,
    hsa_ven_amd_graph_command_list_info_t attribute, void* value) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    return rocr::graph::GetInfo(command_list, attribute, value);
  } catch (...) {
    return rocr::AMD::handleException();
  }
}

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_materialize_packet(
    hsa_ven_amd_graph_command_list_t command_list, hsa_fence_scope_t acquire_scope,
    hsa_fence_scope_t release_scope, uint32_t barrier, hsa_signal_t completion_signal,
    hsa_ven_amd_graph_materialized_packet_t* packet) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    return rocr::graph::Materialize(command_list, nullptr, acquire_scope, release_scope,
                                    barrier, completion_signal, packet);
  } catch (...) {
    return rocr::AMD::handleException();
  }
}

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_materialize_packet_for_queue(
    hsa_ven_amd_graph_command_list_t command_list, hsa_queue_t* queue,
    hsa_fence_scope_t acquire_scope, hsa_fence_scope_t release_scope, uint32_t barrier,
    hsa_signal_t completion_signal, hsa_ven_amd_graph_materialized_packet_t* packet) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen()) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (queue == nullptr) {
      return HSA_STATUS_ERROR_INVALID_QUEUE;
    }
    return rocr::graph::Materialize(command_list, queue, acquire_scope, release_scope,
                                    barrier, completion_signal, packet);
  } catch (...) {
    return rocr::AMD::handleException();
  }
}
