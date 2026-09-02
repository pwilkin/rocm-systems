/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
 * SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_GRAPH_COMMAND_ENCODER_H_
#define HSA_RUNTIME_CORE_INC_AMD_GRAPH_COMMAND_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "inc/hsa.h"
#include "inc/hsa_ven_amd_graph.h"

namespace rocr::graph {

struct GraphCommandCapability {
  hsa_ven_amd_graph_encoder_family_t family;
  bool compile_supported;
  bool runtime_qualified;
};

inline GraphCommandCapability GetGraphCommandCapability(uint32_t major, uint32_t minor,
                                                        uint32_t stepping) {
  if (major == 11) {
    return {HSA_VEN_AMD_GRAPH_ENCODER_GFX11, true, minor == 5 && stepping == 1};
  }
  if (major == 12) {
    return {HSA_VEN_AMD_GRAPH_ENCODER_GFX12, minor == 0, false};
  }
  return {HSA_VEN_AMD_GRAPH_ENCODER_NONE, false, false};
}

struct Gfx11KernelImage {
  uint64_t code_entry;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint32_t compute_pgm_rsrc3;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload_length;
};

class Gfx11CommandEncoder {
 public:
  Gfx11CommandEncoder() { AcquireSystem(); }

  bool Append(const hsa_kernel_dispatch_packet_t& packet, const Gfx11KernelImage& image,
              uint32_t kernel_flags, hsa_ven_amd_graph_dependency_t dependency) {
    if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK) != 0 ||
        (image.kernel_code_properties & ~kSupportedKernelProperties) != 0 ||
        (packet.private_segment_size != 0 &&
         (image.kernel_code_properties & kEnableSgprPrivateSegmentBuffer) != 0) ||
        image.kernarg_preload_length != 0 || image.code_entry == 0 ||
        (image.code_entry & 0xff) != 0) {
      return false;
    }
    if (packet.workgroup_size_x == 0 || packet.workgroup_size_y == 0 ||
        packet.workgroup_size_z == 0 ||
        packet.grid_size_x % packet.workgroup_size_x != 0 ||
        packet.grid_size_y % packet.workgroup_size_y != 0 ||
        packet.grid_size_z % packet.workgroup_size_z != 0) {
      return false;
    }

    const uint32_t lds_blocks = packet.group_segment_size / kLdsGranule +
                                (packet.group_segment_size % kLdsGranule != 0);
    if (lds_blocks > (kLdsSizeMask >> kLdsSizeShift)) {
      return false;
    }
    const uint32_t rsrc2 =
        (image.compute_pgm_rsrc2 & ~kLdsSizeMask) | (lds_blocks << kLdsSizeShift);

    if (dispatch_count_ != 0) {
      if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW) {
        DependencyRmwSameAgent();
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY &&
                 (kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY) != 0) {
        DependencyRmwVmem();
      } else {
        return false;
      }
    }

    const uint32_t pgm[] = {
        static_cast<uint32_t>(image.code_entry >> 8),
        static_cast<uint32_t>(image.code_entry >> 40),
    };
    SetShRegs(kComputePgmLo, pgm, 2);
    const uint32_t rsrc[] = {image.compute_pgm_rsrc1, rsrc2};
    SetShRegs(kComputePgmRsrc1, rsrc, 2);
    SetShRegs(kComputePgmRsrc3, &image.compute_pgm_rsrc3, 1);
    const uint32_t zero = 0;
    const size_t tmpring_words_before = words_.size();
    SetShRegs(kComputeTmpRingSize, &zero, 1);
    if (tmpring_patch_dword_ == kNoPatchDword && words_.size() != tmpring_words_before) {
      tmpring_patch_dword_ = words_.size() - 1;
    }
    const uint32_t workgroup[] = {packet.workgroup_size_x, packet.workgroup_size_y,
                                  packet.workgroup_size_z};
    SetShRegs(kComputeNumThreadX, workgroup, 3);
    SetShRegs(kComputeResourceLimits, &zero, 1);

    uint32_t user_sgprs[6] = {};
    size_t user_sgpr_count = 0;
    if ((image.kernel_code_properties & kEnableSgprPrivateSegmentBuffer) != 0) {
      user_sgpr_count += 4;
    }
    if ((image.kernel_code_properties & kEnableSgprKernargSegmentPtr) != 0) {
      if (packet.kernarg_address == nullptr) {
        return false;
      }
      const uint64_t kernarg = reinterpret_cast<uint64_t>(packet.kernarg_address);
      user_sgprs[user_sgpr_count++] = static_cast<uint32_t>(kernarg);
      user_sgprs[user_sgpr_count++] = static_cast<uint32_t>(kernarg >> 32);
    }
    if (user_sgpr_count != 0) {
      SetShRegs(kComputeUserData0, user_sgprs, user_sgpr_count);
    }

    const uint32_t workgroups[] = {
        packet.grid_size_x / packet.workgroup_size_x,
        packet.grid_size_y / packet.workgroup_size_y,
        packet.grid_size_z / packet.workgroup_size_z,
    };
    const uint32_t wave_mode =
        (image.kernel_code_properties & kEnableWavefrontSize32) != 0 ? 1u << 15 : 0;
    const uint32_t initiator = (1u << 0) | (1u << 2) | (1u << 3) | wave_mode;
    words_.push_back(Packet3(kPacket3DispatchDirect, 4, true));
    words_.insert(words_.end(), workgroups, workgroups + 3);
    words_.push_back(initiator);
    ++dispatch_count_;
    return true;
  }

  void Finish() { WaitComputeIdle(); }
  const std::vector<uint32_t>& words() const { return words_; }
  size_t dispatch_count() const { return dispatch_count_; }
  size_t tmpring_patch_dword() const { return tmpring_patch_dword_; }

 private:
  static constexpr uint32_t kPacket3SetShReg = 0x76;
  static constexpr uint32_t kPacket3DispatchDirect = 0x15;
  static constexpr uint32_t kPacket3EventWrite = 0x46;
  static constexpr uint32_t kPacket3AcquireMem = 0x58;
  static constexpr uint32_t kComputeNumThreadX = 0x207;
  static constexpr uint32_t kComputePgmLo = 0x20c;
  static constexpr uint32_t kComputePgmRsrc1 = 0x212;
  static constexpr uint32_t kComputeResourceLimits = 0x215;
  static constexpr uint32_t kComputeTmpRingSize = 0x218;
  static constexpr uint32_t kComputePgmRsrc3 = 0x228;
  static constexpr uint32_t kComputeUserData0 = 0x240;
  static constexpr uint32_t kLdsSizeMask = 0x00ff8000;
  static constexpr uint32_t kLdsSizeShift = 15;
  static constexpr uint32_t kLdsGranule = 512;
  static constexpr uint16_t kEnableSgprPrivateSegmentBuffer = 1 << 0;
  static constexpr uint16_t kEnableSgprKernargSegmentPtr = 1 << 3;
  static constexpr uint16_t kEnableWavefrontSize32 = 1 << 10;
  static constexpr uint16_t kSupportedKernelProperties =
      kEnableSgprPrivateSegmentBuffer | kEnableSgprKernargSegmentPtr |
      kEnableWavefrontSize32;

  static uint32_t Packet3(uint32_t opcode, uint32_t body_dwords, bool compute) {
    return (3u << 30) | ((body_dwords - 1) << 16) | (opcode << 8) |
           (compute ? 1u << 1 : 0);
  }

  void AcquireSystem() {
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 7, false), 0, UINT32_MAX, 0xff, 0, 0, 4,
        (1u << 16) | (1u << 15) | (1u << 14) | (1u << 9) | (1u << 8) |
            (1u << 7) | (1u << 6) | (1u << 5) | (1u << 4) | 1u,
    });
  }

  void WaitComputeIdle() {
    words_.push_back(Packet3(kPacket3EventWrite, 1, false));
    words_.push_back(0x407);
  }

  void DependencyRmwSameAgent() {
    WaitComputeIdle();
    EmitAcquire(0x00380);
  }

  void DependencyRmwVmem() {
    WaitComputeIdle();
    EmitAcquire(0x00300);
  }

  void EmitAcquire(uint32_t gcr_cntl) {
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 7, false), 0, UINT32_MAX, 0x00ffffff, 0, 0, 10,
        gcr_cntl,
    });
  }

  void SetShRegs(uint32_t first, const uint32_t* values, size_t count) {
    size_t i = 0;
    while (i < count) {
      while (i < count) {
        const uint32_t reg = first + static_cast<uint32_t>(i);
        const auto it = register_state_.find(reg);
        if (it == register_state_.end() || it->second != values[i]) {
          break;
        }
        ++i;
      }
      if (i == count) {
        break;
      }
      const size_t run_start = i;
      while (i < count) {
        const uint32_t reg = first + static_cast<uint32_t>(i);
        const auto it = register_state_.find(reg);
        if (it != register_state_.end() && it->second == values[i]) {
          break;
        }
        register_state_[reg] = values[i];
        ++i;
      }
      const size_t run_count = i - run_start;
      words_.push_back(Packet3(kPacket3SetShReg,
                               1 + static_cast<uint32_t>(run_count), true));
      words_.push_back(first + static_cast<uint32_t>(run_start));
      words_.insert(words_.end(), values + run_start, values + run_start + run_count);
    }
  }

  std::vector<uint32_t> words_;
  std::map<uint32_t, uint32_t> register_state_;
  size_t dispatch_count_ = 0;
  static constexpr size_t kNoPatchDword = static_cast<size_t>(-1);
  size_t tmpring_patch_dword_ = kNoPatchDword;
};

struct Gfx12KernelImage {
  uint64_t code_entry;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint32_t compute_pgm_rsrc3;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload_length;
};

class Gfx12CommandEncoder {
 public:
  Gfx12CommandEncoder() { AcquireInterNode(); }

  bool Append(const hsa_kernel_dispatch_packet_t& packet, const Gfx12KernelImage& image,
              uint32_t kernel_flags, hsa_ven_amd_graph_dependency_t dependency) {
    if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK) != 0 ||
        (image.kernel_code_properties & ~kSupportedKernelProperties) != 0 ||
        image.kernarg_preload_length != 0 || image.code_entry == 0 ||
        (image.code_entry & 0xff) != 0) {
      return false;
    }
    if (packet.workgroup_size_x == 0 || packet.workgroup_size_y == 0 ||
        packet.workgroup_size_z == 0) {
      return false;
    }

    const uint32_t lds_blocks = packet.group_segment_size / kLdsGranule +
                                (packet.group_segment_size % kLdsGranule != 0);
    if (lds_blocks > (kLdsSizeMask >> kLdsSizeShift)) {
      return false;
    }
    const uint32_t rsrc2 =
        (image.compute_pgm_rsrc2 & ~kLdsSizeMask) | (lds_blocks << kLdsSizeShift);

    if (dispatch_count_ != 0) {
      if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW) {
        DependencyRmwSameAgent();
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY &&
                 (kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY) != 0) {
        DependencyRmwVmem();
      } else {
        return false;
      }
    }

    const uint32_t pgm[] = {
        static_cast<uint32_t>(image.code_entry >> 8),
        static_cast<uint32_t>(image.code_entry >> 40),
    };
    SetShRegs(kComputePgmLo, pgm, 2);
    const uint32_t rsrc[] = {image.compute_pgm_rsrc1, rsrc2};
    SetShRegs(kComputePgmRsrc1, rsrc, 2);
    SetShRegs(kComputePgmRsrc3, &image.compute_pgm_rsrc3, 1);
    const uint32_t zero = 0;
    SetShRegs(kComputeTmpRingSize, &zero, 1);
    const uint32_t workgroup[] = {packet.workgroup_size_x, packet.workgroup_size_y,
                                  packet.workgroup_size_z};
    SetShRegs(kComputeNumThreadX, workgroup, 3);
    const uint32_t resource_limits = 0x3ff;
    SetShRegs(kComputeResourceLimits, &resource_limits, 1);

    if ((image.kernel_code_properties & kEnableSgprKernargSegmentPtr) != 0) {
      if (packet.kernarg_address == nullptr) {
        return false;
      }
      const uint64_t kernarg = reinterpret_cast<uint64_t>(packet.kernarg_address);
      const uint32_t user_sgprs[] = {static_cast<uint32_t>(kernarg),
                                     static_cast<uint32_t>(kernarg >> 32)};
      SetShRegs(kComputeUserData0, user_sgprs, 2);
    }

    FlushShRegs();

    const uint32_t dimensions[] = {packet.grid_size_x, packet.grid_size_y,
                                   packet.grid_size_z};
    const uint32_t wave_mode =
        (image.kernel_code_properties & kEnableWavefrontSize32) != 0 ? 1u << 15 : 0;
    const uint32_t initiator = (1u << 0) | (1u << 2) | (1u << 5) | wave_mode;
    words_.push_back(Packet3(kPacket3DispatchDirect, 4, true));
    words_.insert(words_.end(), dimensions, dimensions + 3);
    words_.push_back(initiator);
    ++dispatch_count_;
    return true;
  }

  void Finish() { WaitComputeIdle(); }
  const std::vector<uint32_t>& words() const { return words_; }
  size_t dispatch_count() const { return dispatch_count_; }
  size_t tmpring_patch_dword() const { return tmpring_patch_dword_; }

 private:
  static constexpr uint32_t kPacket3SetShRegPairs = 0xba;
  static constexpr uint32_t kPacket3DispatchDirect = 0x15;
  static constexpr uint32_t kPacket3EventWrite = 0x46;
  static constexpr uint32_t kPacket3AcquireMem = 0x58;
  static constexpr uint32_t kComputeNumThreadX = 0x207;
  static constexpr uint32_t kComputePgmLo = 0x20c;
  static constexpr uint32_t kComputePgmRsrc1 = 0x212;
  static constexpr uint32_t kComputeResourceLimits = 0x215;
  static constexpr uint32_t kComputeTmpRingSize = 0x218;
  static constexpr uint32_t kComputePgmRsrc3 = 0x228;
  static constexpr uint32_t kComputeUserData0 = 0x240;
  static constexpr uint32_t kLdsSizeMask = 0x00ff8000;
  static constexpr uint32_t kLdsSizeShift = 15;
  static constexpr uint32_t kLdsGranule = 512;
  static constexpr uint16_t kEnableSgprKernargSegmentPtr = 1 << 3;
  static constexpr uint16_t kEnableWavefrontSize32 = 1 << 10;
  static constexpr uint16_t kSupportedKernelProperties =
      kEnableSgprKernargSegmentPtr | kEnableWavefrontSize32;

  static uint32_t Packet3(uint32_t opcode, uint32_t body_dwords, bool compute) {
    return (3u << 30) | ((body_dwords - 1) << 16) | (opcode << 8) |
           (compute ? 1u << 1 : 0);
  }

  void AcquireInterNode() { EmitAcquire(0xff, 4, 0x10180); }

  void WaitComputeIdle() {
    words_.push_back(Packet3(kPacket3EventWrite, 1, false));
    words_.push_back(0x407);
  }

  void DependencyRmwSameAgent() {
    WaitComputeIdle();
    EmitAcquire(0x00ffffff, 10, 0x00380);
  }

  void DependencyRmwVmem() {
    WaitComputeIdle();
    EmitAcquire(0x00ffffff, 10, 0x00300);
  }

  void EmitAcquire(uint32_t size_hi, uint32_t poll_interval, uint32_t gcr_cntl) {
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 7, false), 0, UINT32_MAX, size_hi, 0, 0,
        poll_interval, gcr_cntl,
    });
  }

  void SetShRegs(uint32_t first, const uint32_t* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      const uint32_t reg = first + static_cast<uint32_t>(i);
      const auto it = register_state_.find(reg);
      if (it == register_state_.end() || it->second != values[i]) {
        register_state_[reg] = values[i];
        pending_registers_.push_back(reg);
        pending_registers_.push_back(values[i]);
      }
    }
  }

  void FlushShRegs() {
    if (pending_registers_.empty()) {
      return;
    }
    const size_t header = words_.size();
    words_.push_back(Packet3(kPacket3SetShRegPairs,
                             static_cast<uint32_t>(pending_registers_.size()), true));
    words_.insert(words_.end(), pending_registers_.begin(), pending_registers_.end());
    if (tmpring_patch_dword_ == kNoPatchDword) {
      for (size_t i = 0; i < pending_registers_.size(); i += 2) {
        if (pending_registers_[i] == kComputeTmpRingSize) {
          tmpring_patch_dword_ = header + i + 2;
          break;
        }
      }
    }
    pending_registers_.clear();
  }

  std::vector<uint32_t> words_;
  std::vector<uint32_t> pending_registers_;
  std::map<uint32_t, uint32_t> register_state_;
  size_t dispatch_count_ = 0;
  static constexpr size_t kNoPatchDword = static_cast<size_t>(-1);
  size_t tmpring_patch_dword_ = kNoPatchDword;
};

}  // namespace rocr::graph

#endif  // HSA_RUNTIME_CORE_INC_AMD_GRAPH_COMMAND_ENCODER_H_
