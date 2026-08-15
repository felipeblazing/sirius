/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Regression guards for freed-while-read corruption across streams. Batch handoff in the
// engine is event-ordered, not host-synced, so a GPU->HOST conversion must wait the source's
// writer event before reading, and rebind_stream must order the adopting stream after the
// stream whose readers may still be in flight. Each test parks the producer/reader stream with
// a host function so the race window is deterministic.

#include "catch.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

struct ordering_test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  ordering_test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

ordering_test_env& env()
{
  static ordering_test_env e;
  return e;
}

constexpr std::size_t kRows      = 1 << 16;
constexpr std::int32_t kStale    = 0x2AAAAAAA;
constexpr std::int32_t kExpected = 0x1BBBBBBB;

struct delay_state {
  std::atomic<bool> release{false};
};

/// Host function that parks a stream until the test releases it.
void CUDART_CB block_until_released(void* userData)
{
  auto* state = static_cast<delay_state*>(userData);
  while (!state->release.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

/// Round-trip a HOST representation back to a GPU table on `stream` and return column 0's bytes.
std::vector<std::int32_t> read_back(cucascade::representation_converter_registry& registry,
                                    cucascade::idata_representation& host_rep,
                                    rmm::cuda_stream_view stream)
{
  auto gpu_rep =
    registry.convert<cucascade::gpu_table_representation>(host_rep, env().gpu_space, stream);
  auto& gpu = gpu_rep->cast<cucascade::gpu_table_representation>();
  auto view = gpu.get_table_view();
  std::vector<std::int32_t> out(static_cast<std::size_t>(view.column(0).size()));
  REQUIRE(cudaMemcpyAsync(out.data(),
                          view.column(0).head<std::int32_t>(),
                          out.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  return out;
}

/// Warm first-call costs on the conversion path so they do not eat into the race window.
void warm_conversion_path(cucascade::representation_converter_registry& registry)
{
  rmm::cuda_stream stream;
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(kRows),
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view());
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto rep = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(cols)), *env().gpu_space, stream.view());
  auto host_rep =
    registry.convert<cucascade::host_data_representation>(*rep, env().host_space, stream.view());
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
}

/// Enqueue a column's final write behind a parked host function on the producer stream, convert
/// GPU->HOST on another stream mid-flight, and require the host image to carry the final bytes.
void run_ordering_scenario(cucascade::representation_converter_registry& registry)
{
  warm_conversion_path(registry);

  rmm::cuda_stream producer_stream;
  rmm::cuda_stream conversion_stream;

  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(kRows),
                                       cudf::mask_state::UNALLOCATED,
                                       producer_stream.view());
  {
    std::vector<std::int32_t> stale(kRows, kStale);
    REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                            stale.data(),
                            kRows * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice,
                            producer_stream.value()) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(producer_stream.value()) == cudaSuccess);
  }

  // The source must be pinned: a pageable H2D behind a parked stream blocks the host thread.
  delay_state gate;
  std::vector<std::int32_t> final_bytes(kRows, kExpected);
  REQUIRE(cudaHostRegister(final_bytes.data(), kRows * sizeof(std::int32_t), 0) == cudaSuccess);
  REQUIRE(cudaLaunchHostFunc(producer_stream.value(), block_until_released, &gate) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                          final_bytes.data(),
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice,
                          producer_stream.value()) == cudaSuccess);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table   = std::make_unique<cudf::table>(std::move(cols));
  auto gpu_rep = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), *env().gpu_space, producer_stream.view());

  // A correctly ordered converter blocks here, so the gate is released from another thread.
  std::thread releaser([&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    gate.release.store(true, std::memory_order_release);
  });
  auto host_rep = registry.convert<cucascade::host_data_representation>(
    *gpu_rep, env().host_space, conversion_stream.view());
  releaser.join();

  REQUIRE(cudaStreamSynchronize(producer_stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(conversion_stream.value()) == cudaSuccess);
  REQUIRE(cudaHostUnregister(final_bytes.data()) == cudaSuccess);

  auto const out = read_back(registry, *host_rep, conversion_stream.view());
  REQUIRE(out.size() == kRows);
  std::size_t stale_count = 0;
  for (auto v : out) {
    if (v != kExpected) { ++stale_count; }
  }
  INFO("host image carries " << stale_count << " stale (torn) values of " << kRows);
  REQUIRE(stale_count == 0);
}

}  // namespace

TEST_CASE("builtin fast GPU->HOST conversion orders after the producer's writer event",
          "[spill_converter_ordering]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  run_ordering_scenario(registry);
}

// A consumer has enqueued reads on the producer stream and dropped its read lock; the writer
// event is already signaled, so only rebind_stream's reader-ordering edge protects the buffers
// from the downgrade's free.
TEST_CASE("rebind_stream orders the adopting stream after a straggler reader",
          "[spill_converter_ordering]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  warm_conversion_path(registry);

  rmm::cuda_stream reader_stream;
  rmm::cuda_stream downgrade_stream;

  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(kRows),
                                       cudf::mask_state::UNALLOCATED,
                                       reader_stream.view());
  {
    std::vector<std::int32_t> expected(kRows, kExpected);
    REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                            expected.data(),
                            kRows * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice,
                            reader_stream.value()) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);
  }
  auto const* batch_bytes = col->view().head<std::int32_t>();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto gpu_rep = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(cols)), *env().gpu_space, reader_stream.view());

  // Park the reader stream, then enqueue the straggler read behind the park.
  delay_state gate;
  std::vector<std::int32_t> reader_out(kRows, 0);
  REQUIRE(cudaHostRegister(reader_out.data(), kRows * sizeof(std::int32_t), 0) == cudaSuccess);
  REQUIRE(cudaLaunchHostFunc(reader_stream.value(), block_until_released, &gate) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(reader_out.data(),
                          batch_bytes,
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost,
                          reader_stream.value()) == cudaSuccess);

  // The adopting stream must now be fenced behind the parked reader stream.
  gpu_rep->rebind_stream(downgrade_stream.view());
  cudaEvent_t probe{nullptr};
  REQUIRE(cudaEventCreateWithFlags(&probe, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(probe, downgrade_stream.value()) == cudaSuccess);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(cudaEventQuery(probe) == cudaErrorNotReady);

  // Full downgrade shape: convert on the adopting stream, free the source, and force VA reuse
  // with a poison fill while the reader is still parked.
  std::thread releaser([&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    gate.release.store(true, std::memory_order_release);
  });
  auto host_rep = registry.convert<cucascade::host_data_representation>(
    *gpu_rep, env().host_space, downgrade_stream.view());
  gpu_rep.reset();
  {
    rmm::device_buffer poison(kRows * sizeof(std::int32_t), downgrade_stream.view());
    REQUIRE(cudaMemsetAsync(
              poison.data(), 0xEE, kRows * sizeof(std::int32_t), downgrade_stream.value()) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(downgrade_stream.value()) == cudaSuccess);
  }
  releaser.join();
  REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(downgrade_stream.value()) == cudaSuccess);
  REQUIRE(cudaEventDestroy(probe) == cudaSuccess);
  REQUIRE(cudaHostUnregister(reader_out.data()) == cudaSuccess);

  std::size_t scribbled = 0;
  for (auto v : reader_out) {
    if (v != kExpected) { ++scribbled; }
  }
  INFO("straggler reader observed " << scribbled << " scribbled values of " << kRows);
  REQUIRE(scribbled == 0);

  auto const out = read_back(registry, *host_rep, downgrade_stream.view());
  REQUIRE(out.size() == kRows);
  std::size_t torn = 0;
  for (auto v : out) {
    if (v != kExpected) { ++torn; }
  }
  INFO("host image carries " << torn << " torn values of " << kRows);
  REQUIRE(torn == 0);
}
