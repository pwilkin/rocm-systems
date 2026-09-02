/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
 * SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "core/inc/amd_graph_command_encoder.h"

namespace {

rocr::graph::Gfx11KernelImage Image(uint16_t properties = 0x409) {
  return {
      0x1'0000,
      0x11,
      0x22,
      0x33,
      properties,
      0,
  };
}

rocr::graph::Gfx12KernelImage Image12(uint16_t properties = 0x408) {
  return {
      0x1'0000,
      0x11,
      0x22,
      0x33,
      properties,
      0,
  };
}

hsa_kernel_dispatch_packet_t Packet() {
  hsa_kernel_dispatch_packet_t packet{};
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.kernarg_address = reinterpret_cast<void*>(0x1234'5678'9abc'def0ULL);
  return packet;
}

TEST(GraphCommandEncoder, Gfx11WordsMatchRedline20474) {
  constexpr std::array<uint32_t, 60> expected = {
      0xc0065800, 0x00000000, 0xffffffff, 0x000000ff, 0x00000000, 0x00000000,
      0x00000004, 0x0001c3f1, 0xc0027602, 0x0000020c, 0x00000100, 0x00000000,
      0xc0027602, 0x00000212, 0x00000011, 0x00000022, 0xc0017602, 0x00000228,
      0x00000033, 0xc0017602, 0x00000218, 0x00000000, 0xc0037602, 0x00000207,
      0x00000100, 0x00000001, 0x00000001, 0xc0017602, 0x00000215, 0x00000000,
      0xc0067602, 0x00000240, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x9abcdef0, 0x12345678, 0xc0031502, 0x00000001, 0x00000001, 0x00000001,
      0x0000800d, 0xc0004600, 0x00000407, 0xc0065800, 0x00000000, 0xffffffff,
      0x00ffffff, 0x00000000, 0x00000000, 0x0000000a, 0x00000380, 0xc0031502,
      0x00000001, 0x00000001, 0x00000001, 0x0000800d, 0xc0004600, 0x00000407,
  };

  rocr::graph::Gfx11CommandEncoder encoder;
  const auto packet = Packet();
  const auto image = Image();
  ASSERT_TRUE(encoder.Append(packet, image, 0,
                             HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  ASSERT_TRUE(encoder.Append(packet, image, 0,
                             HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  encoder.Finish();
  ASSERT_EQ(encoder.dispatch_count(), 2u);
  ASSERT_EQ(encoder.words().size(), expected.size());
  EXPECT_TRUE(std::equal(encoder.words().begin(), encoder.words().end(), expected.begin()));
}

TEST(GraphCommandEncoder, Gfx11VmemBoundaryRequiresVerifiedMetadata) {
  auto packet = Packet();
  const auto image = Image();
  rocr::graph::Gfx11CommandEncoder rejected;
  ASSERT_TRUE(rejected.Append(packet, image, 0,
                              HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  EXPECT_FALSE(rejected.Append(packet, image, 0,
                               HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY));

  rocr::graph::Gfx11CommandEncoder accepted;
  ASSERT_TRUE(accepted.Append(packet, image, HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY,
                              HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  ASSERT_TRUE(accepted.Append(packet, image, HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY,
                              HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY));
  accepted.Finish();
  EXPECT_NE(std::find(accepted.words().begin(), accepted.words().end(), 0x00000300),
            accepted.words().end());
}

TEST(GraphCommandEncoder, Gfx11RejectsUnsupportedAbiShapes) {
  auto packet = Packet();
  auto image = Image();

  packet.private_segment_size = 4;
  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  packet.private_segment_size = 0;

  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, image, HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK,
      HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));

  image.kernel_code_properties |= 1u << 2;
  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  image = Image();

  packet.grid_size_x = 255;
  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  packet = Packet();

  packet.kernarg_address = nullptr;
  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));

  packet = Packet();
  image.code_entry += 4;
  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));

  packet = Packet();
  image = Image();
  image.kernarg_preload_length = 1;
  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
}

TEST(GraphCommandEncoder, QueueScratchUsesPatchableTmpRingState) {
  auto packet = Packet();
  packet.private_segment_size = 32;

  rocr::graph::Gfx11CommandEncoder gfx11;
  ASSERT_TRUE(gfx11.Append(packet, Image(0x408), 0,
                           HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  ASSERT_LT(gfx11.tmpring_patch_dword(), gfx11.words().size());
  EXPECT_EQ(gfx11.words()[gfx11.tmpring_patch_dword()], 0u);

  rocr::graph::Gfx12CommandEncoder gfx12;
  ASSERT_TRUE(gfx12.Append(packet, Image12(), 0,
                           HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  ASSERT_LT(gfx12.tmpring_patch_dword(), gfx12.words().size());
  EXPECT_EQ(gfx12.words()[gfx12.tmpring_patch_dword()], 0u);
}

TEST(GraphCommandEncoder, Gfx12WordsMatchRedline20474) {
  constexpr std::array<uint32_t, 62> expected = {
      0xc0065800, 0x00000000, 0xffffffff, 0x000000ff, 0x00000000, 0x00000000,
      0x00000004, 0x00010180, 0xc0027602, 0x0000020c, 0x00000100, 0x00000000,
      0xc0027602, 0x00000212, 0x00000011, 0x00000022, 0xc0017602, 0x00000223,
      0x00000033, 0xc0017602, 0x00000216, 0x00000000, 0xc0037602, 0x00000207,
      0x00000100, 0x00000001, 0x00000001, 0xc0017602, 0x00000215, 0x000003ff,
      0xc0047602, 0x00000230, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
      0xc0027602, 0x00000240, 0x9abcdef0, 0x12345678, 0xc0031502, 0x00000100,
      0x00000001, 0x00000001, 0x00008025, 0xc0004600, 0x00000407, 0xc0065800,
      0x00000000, 0xffffffff, 0x00ffffff, 0x00000000, 0x00000000, 0x0000000a,
      0x00000380, 0xc0031502, 0x00000100, 0x00000001, 0x00000001, 0x00008025,
      0xc0004600, 0x00000407,
  };

  rocr::graph::Gfx12CommandEncoder encoder;
  const auto packet = Packet();
  const auto image = Image12();
  ASSERT_TRUE(encoder.Append(packet, image, 0,
                             HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  ASSERT_TRUE(encoder.Append(packet, image, 0,
                             HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  encoder.Finish();
  ASSERT_EQ(encoder.dispatch_count(), 2u);
  ASSERT_EQ(encoder.words().size(), expected.size());
  EXPECT_TRUE(std::equal(encoder.words().begin(), encoder.words().end(), expected.begin()));
}

TEST(GraphCommandEncoder, Gfx12UsesSeparateAbiAndVmemBoundary) {
  const auto packet = Packet();
  auto image = Image12();
  rocr::graph::Gfx12CommandEncoder encoder;
  ASSERT_TRUE(encoder.Append(packet, image, HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY,
                             HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  ASSERT_TRUE(encoder.Append(packet, image, HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY,
                             HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY));
  encoder.Finish();
  EXPECT_NE(std::find(encoder.words().begin(), encoder.words().end(), 0x00000300),
            encoder.words().end());

  image.kernel_code_properties |= 1u << 0;
  EXPECT_FALSE(rocr::graph::Gfx12CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));

  image = Image12();
  image.kernarg_preload_length = 1;
  EXPECT_FALSE(rocr::graph::Gfx12CommandEncoder().Append(
      packet, image, 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
}

TEST(GraphCommandEncoder, CapabilityTableCoversDeclaredGfx11AndGfx12Families) {
  for (const auto [minor, stepping] :
       {std::pair{0u, 0u}, std::pair{0u, 1u}, std::pair{0u, 2u}, std::pair{0u, 3u},
        std::pair{5u, 0u}, std::pair{5u, 1u}, std::pair{5u, 2u}, std::pair{5u, 3u}}) {
    const auto capability = rocr::graph::GetGraphCommandCapability(11, minor, stepping);
    EXPECT_EQ(capability.family, HSA_VEN_AMD_GRAPH_ENCODER_GFX11);
    EXPECT_TRUE(capability.compile_supported);
    EXPECT_EQ(capability.runtime_qualified, minor == 5 && stepping == 1);
  }
  for (const auto [minor, stepping] :
       {std::pair{0u, 0u}, std::pair{0u, 1u}, std::pair{5u, 0u}}) {
    const auto capability = rocr::graph::GetGraphCommandCapability(12, minor, stepping);
    EXPECT_EQ(capability.family, HSA_VEN_AMD_GRAPH_ENCODER_GFX12);
    EXPECT_TRUE(capability.compile_supported);
    EXPECT_FALSE(capability.runtime_qualified);
  }
  const auto unsupported = rocr::graph::GetGraphCommandCapability(10, 3, 0);
  EXPECT_EQ(unsupported.family, HSA_VEN_AMD_GRAPH_ENCODER_NONE);
  EXPECT_FALSE(unsupported.compile_supported);
  EXPECT_FALSE(unsupported.runtime_qualified);
}

TEST(GraphCommandEncoder, DynamicLdsIsEncodedAndOverflowRejected) {
  auto packet = Packet();
  packet.group_segment_size = 1536;
  constexpr std::array<uint32_t, 2> expected_rsrc = {0x00000011, 0x00018022};

  rocr::graph::Gfx11CommandEncoder gfx11;
  ASSERT_TRUE(gfx11.Append(packet, Image(), 0,
                           HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  const auto gfx11_rsrc = std::search(gfx11.words().begin(), gfx11.words().end(),
                                      expected_rsrc.begin(), expected_rsrc.end());
  EXPECT_NE(gfx11_rsrc, gfx11.words().end());

  rocr::graph::Gfx12CommandEncoder gfx12;
  ASSERT_TRUE(gfx12.Append(packet, Image12(), 0,
                           HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  const auto gfx12_rsrc = std::search(gfx12.words().begin(), gfx12.words().end(),
                                      expected_rsrc.begin(), expected_rsrc.end());
  EXPECT_NE(gfx12_rsrc, gfx12.words().end());

  packet.group_segment_size = UINT32_MAX;
  EXPECT_FALSE(rocr::graph::Gfx11CommandEncoder().Append(
      packet, Image(), 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
  EXPECT_FALSE(rocr::graph::Gfx12CommandEncoder().Append(
      packet, Image12(), 0, HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW));
}

}  // namespace
