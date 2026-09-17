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

/**
 * @file test_dynamic_filter_probe.cpp
 * @brief Single-GPU probe-kernel semantics of the membership dynamic filters (IN-list,
 *        small IN-list, Bloom): heterogeneous integer probe carriers (no materialized cast),
 *        the optional prior keep-mask (dead rows skip the lookup), sentinel conservation, the
 *        refusal of non-integer probe types, INT8/INT16 and unsigned keys, carrier-typed build
 *        sets, cross-carrier mask identity, and STRING keys probed through 64-bit XXHash_64
 *        fingerprints (pinned against a host reference of the same hash).
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

using sirius::op::classify_membership_key;
using sirius::op::membership_key_domain;
using sirius::op::membership_key_family;
using sirius::op::membership_key_rep;
using sirius::op::membership_key_supported;
using sirius::op::membership_probe_compatible;
using sirius::op::sirius_dynamic_bloom_filter;
using sirius::op::sirius_dynamic_in_list_filter;
using sirius::op::sirius_dynamic_small_in_list_filter;

namespace {

constexpr int kDevice = 0;  // build == probe device: the source replica answers directly

template <typename T>
std::unique_ptr<cudf::column> make_values(std::vector<T> const& values,
                                          cudf::data_type type,
                                          rmm::cuda_stream_view stream)
{
  auto col       = cudf::make_numeric_column(type,
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(col->mutable_view().data<T>(),
                                   values.data(),
                                   values.size() * sizeof(T),
                                   cudaMemcpyHostToDevice,
                                   stream.value());
  REQUIRE(err == cudaSuccess);
  stream.synchronize();  // callers pass temporaries; these tests do not benchmark ingestion
  return col;
}

std::unique_ptr<cudf::column> make_int32(std::vector<std::int32_t> const& v,
                                         rmm::cuda_stream_view stream)
{
  return make_values(v, cudf::data_type{cudf::type_id::INT32}, stream);
}

std::unique_ptr<cudf::column> make_int64(std::vector<std::int64_t> const& v,
                                         rmm::cuda_stream_view stream)
{
  return make_values(v, cudf::data_type{cudf::type_id::INT64}, stream);
}

std::vector<std::uint8_t> mask_to_host(cudf::column_view const& mask, rmm::cuda_stream_view stream)
{
  REQUIRE(mask.type().id() == cudf::type_id::BOOL8);
  std::vector<std::uint8_t> host(static_cast<std::size_t>(mask.size()));
  auto const err = cudaMemcpyAsync(host.data(),
                                   mask.data<bool>(),
                                   host.size() * sizeof(bool),
                                   cudaMemcpyDeviceToHost,
                                   stream.value());
  REQUIRE(err == cudaSuccess);
  stream.synchronize();
  return host;
}

/// Upload a packed 1-bit/row keep-mask (bit row%32 of word row/32, 1 = keep) built from @p keep.
rmm::device_buffer upload_prior_mask(std::vector<bool> const& keep, rmm::cuda_stream_view stream)
{
  std::vector<std::uint32_t> words((keep.size() + 31) / 32, 0U);
  for (std::size_t row = 0; row < keep.size(); ++row) {
    if (keep[row]) { words[row / 32] |= (1U << (row % 32)); }
  }
  rmm::device_buffer out{words.data(), words.size() * sizeof(std::uint32_t), stream};
  stream.synchronize();
  return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Heterogeneous probe carriers (the killed probe-key cast)
//===----------------------------------------------------------------------===//

TEST_CASE("IN-list over INT64 keys probes an INT32 carrier without a cast",
          "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto keys = make_int64({10, 20, 30, 40}, stream);
  sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
  REQUIRE(filter.has_persistent_set());

  auto probe = make_int32({10, 15, 20, -3, 40}, stream);
  auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
  REQUIRE(mask != nullptr);
  auto const host = mask_to_host(mask->view(), stream);
  CHECK(host == std::vector<std::uint8_t>{1, 0, 1, 0, 1});
}

TEST_CASE("IN-list over INT32 keys drops out-of-range INT64 probe values",
          "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto keys = make_int32({1, 2, 3}, stream);
  sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};

  auto probe = make_int64({1, 5'000'000'000LL, 3, -5'000'000'000LL}, stream);
  auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
  REQUIRE(mask != nullptr);
  auto const host = mask_to_host(mask->view(), stream);
  CHECK(host == std::vector<std::uint8_t>{1, 0, 1, 0});
}

TEST_CASE("IN-list sentinel semantics under heterogeneous probes", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  SECTION("an INT64 probe equal to the INT32 set's empty sentinel is kept conservatively")
  {
    auto keys = make_int32({7}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int64(
      {static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()), 7, 8}, stream);
    auto mask = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1, 0});  // sentinel keep stays conservative
  }

  SECTION("an INT32 probe against an INT64 set is exact: no widened value hits the sentinel")
  {
    auto keys = make_int64({7}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int32({std::numeric_limits<std::int32_t>::min(), 7}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{0, 1});  // INT32_MIN widened != INT64 sentinel
  }

  SECTION("the homogeneous sentinel keep is unchanged")
  {
    auto keys = make_int64({5}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int64({std::numeric_limits<std::int64_t>::min(), 5, 6}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1, 0});
  }
}

TEST_CASE("small IN-list probes heterogeneous integer carriers", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  SECTION("INT64 needles, INT32 probe")
  {
    auto keys = make_int64({7, 1'000'000'000'000LL}, stream);
    sirius_dynamic_small_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int32({7, -7, 0}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 0, 0});
  }

  SECTION("INT32 needles, INT64 probe with out-of-range values")
  {
    auto keys = make_int32({5, 6}, stream);
    sirius_dynamic_small_in_list_filter filter{keys->view(), stream, mr};
    auto probe = make_int64({5, 6'000'000'000LL, 6}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 0, 1});
  }
}

TEST_CASE("Bloom filter has no false negatives across probe carriers", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  SECTION("INT64 build keys, INT32 probe")
  {
    auto keys = make_int64({100, 200, 300, 9'000'000'000LL}, stream);
    sirius_dynamic_bloom_filter filter{keys->view(), stream, mr};
    auto probe = make_int32({100, 200, 300}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1, 1});  // every inserted key must test positive
  }

  SECTION("INT32 build keys, INT64 probe")
  {
    auto keys = make_int32({100, 200, 300}, stream);
    sirius_dynamic_bloom_filter filter{keys->view(), stream, mr};
    auto probe = make_int64({100, 300}, stream);
    auto mask  = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host == std::vector<std::uint8_t>{1, 1});
  }
}

TEST_CASE("membership filters refuse non-integer probe carriers", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const n      = cudf::size_type{4};

  auto keys = make_int64({1, 2, 3}, stream);
  sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
  sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
  sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};

  auto const decimal = cudf::make_fixed_point_column(
    cudf::data_type{cudf::type_id::DECIMAL64, -2}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const date = cudf::make_timestamp_column(
    cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const fp = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);

  // An unsigned carrier is a semantic mismatch for a signed set too: the planner's casts make a
  // signed<->unsigned key pair unreachable, so it declines rather than reinterpreting bits.
  auto const unsigned_probe = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT32}, n, cudf::mask_state::UNALLOCATED, stream, mr);

  for (auto const* probe : {&decimal, &date, &fp, &unsigned_probe}) {
    CHECK(in_list.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
    CHECK(small_list.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
    CHECK(bloom.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
  }
}

//===----------------------------------------------------------------------===//
// Prior keep-mask (mask-aware probing)
//===----------------------------------------------------------------------===//

TEST_CASE("prior keep-mask gates the membership probes", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // > 32 rows so the packed-word indexing crosses word boundaries. Membership = even values;
  // prior mask keeps rows divisible by 3.
  constexpr std::size_t n = 70;
  std::vector<std::int64_t> key_values;
  std::vector<std::int32_t> probe_values(n);
  std::vector<bool> keep(n);
  std::vector<std::uint8_t> expected_unmasked(n);
  std::vector<std::uint8_t> expected_masked(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (i % 2 == 0) { key_values.push_back(static_cast<std::int64_t>(i)); }
    probe_values[i]      = static_cast<std::int32_t>(i);
    keep[i]              = (i % 3 == 0);
    expected_unmasked[i] = (i % 2 == 0) ? 1 : 0;
    expected_masked[i]   = (i % 2 == 0 && i % 3 == 0) ? 1 : 0;
  }
  auto keys               = make_int64(key_values, stream);
  auto probe              = make_int32(probe_values, stream);
  auto prior              = upload_prior_mask(keep, stream);
  auto const* prior_words = static_cast<std::uint32_t const*>(prior.data());

  auto all_dead = upload_prior_mask(std::vector<bool>(n, false), stream);
  auto all_live = upload_prior_mask(std::vector<bool>(n, true), stream);

  SECTION("IN-list")
  {
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};

    auto masked = filter.compute_mask(probe->view(), prior_words, kDevice, stream, mr);
    REQUIRE(masked != nullptr);
    CHECK(mask_to_host(masked->view(), stream) == expected_masked);

    auto dead = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_dead.data()), kDevice, stream, mr);
    REQUIRE(dead != nullptr);
    CHECK(mask_to_host(dead->view(), stream) == std::vector<std::uint8_t>(n, 0));

    auto live = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_live.data()), kDevice, stream, mr);
    REQUIRE(live != nullptr);
    CHECK(mask_to_host(live->view(), stream) == expected_unmasked);

    auto unmasked = filter.compute_mask(probe->view(), nullptr, kDevice, stream, mr);
    REQUIRE(unmasked != nullptr);
    CHECK(mask_to_host(unmasked->view(), stream) == expected_unmasked);
  }

  SECTION("small IN-list")
  {
    // Needles capped at k_max_keys: membership = {0, 6, 12} over the same probe.
    auto small_keys = make_int64({0, 6, 12}, stream);
    sirius_dynamic_small_in_list_filter filter{small_keys->view(), stream, mr};
    std::vector<std::uint8_t> expected(n, 0);
    for (auto const v : {0, 6, 12}) {
      expected[static_cast<std::size_t>(v)] = keep[static_cast<std::size_t>(v)] ? 1 : 0;
    }

    auto masked = filter.compute_mask(probe->view(), prior_words, kDevice, stream, mr);
    REQUIRE(masked != nullptr);
    CHECK(mask_to_host(masked->view(), stream) == expected);

    auto dead = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_dead.data()), kDevice, stream, mr);
    REQUIRE(dead != nullptr);
    CHECK(mask_to_host(dead->view(), stream) == std::vector<std::uint8_t>(n, 0));
  }

  SECTION("Bloom")
  {
    sirius_dynamic_bloom_filter filter{keys->view(), stream, mr};

    auto masked = filter.compute_mask(probe->view(), prior_words, kDevice, stream, mr);
    REQUIRE(masked != nullptr);
    auto const host = mask_to_host(masked->view(), stream);
    for (std::size_t i = 0; i < n; ++i) {
      if (!keep[i]) {
        CHECK(host[i] == 0);  // dead rows never pass, whatever the filter says
      } else if (i % 2 == 0) {
        CHECK(host[i] == 1);  // live in-set rows must pass (no false negatives)
      }
    }

    auto dead = filter.compute_mask(
      probe->view(), static_cast<std::uint32_t const*>(all_dead.data()), kDevice, stream, mr);
    REQUIRE(dead != nullptr);
    CHECK(mask_to_host(dead->view(), stream) == std::vector<std::uint8_t>(n, 0));
  }
}

TEST_CASE("prior-masked probe still propagates the probe's null mask", "[dynamic_filter][probe]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto keys = make_int64({1, 2, 3, 4}, stream);
  sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};

  auto probe     = make_int32({1, 2, 9, 4}, stream);
  auto null_mask = cudf::create_null_mask(4, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 1, 2, false, stream);
  probe->set_null_mask(std::move(null_mask), 1);

  auto prior = upload_prior_mask({true, true, true, false}, stream);
  auto mask  = filter.compute_mask(
    probe->view(), static_cast<std::uint32_t const*>(prior.data()), kDevice, stream, mr);
  REQUIRE(mask != nullptr);
  CHECK(mask->null_count() == 1);
  auto const host = mask_to_host(mask->view(), stream);
  CHECK(host[0] == 1);  // live, in set
  CHECK(host[2] == 0);  // live, not in set
  CHECK(host[3] == 0);  // dead row
}

//===----------------------------------------------------------------------===//
// Key reps, carriers, and cross-carrier mask identity
//===----------------------------------------------------------------------===//

namespace {

template <typename T>
std::unique_ptr<cudf::column> make_typed(std::vector<std::int64_t> const& values,
                                         rmm::cuda_stream_view stream)
{
  std::vector<T> typed;
  typed.reserve(values.size());
  for (auto const v : values) {
    typed.push_back(static_cast<T>(v));
  }
  return make_values(typed, cudf::data_type{cudf::type_to_id<T>()}, stream);
}

template <typename T>
std::unique_ptr<cudf::column> make_unsigned(std::vector<std::uint64_t> const& values,
                                            rmm::cuda_stream_view stream)
{
  std::vector<T> typed;
  typed.reserve(values.size());
  for (auto const v : values) {
    typed.push_back(static_cast<T>(v));
  }
  return make_values(typed, cudf::data_type{cudf::type_to_id<T>()}, stream);
}

/// Calls fn(T{}) for each signed integer carrier.
template <class Fn>
void for_each_signed_carrier(Fn&& fn)
{
  fn(std::int8_t{});
  fn(std::int16_t{});
  fn(std::int32_t{});
  fn(std::int64_t{});
}

/// Calls fn(T{}) for each unsigned integer carrier.
template <class Fn>
void for_each_unsigned_carrier(Fn&& fn)
{
  fn(std::uint8_t{});
  fn(std::uint16_t{});
  fn(std::uint32_t{});
  fn(std::uint64_t{});
}

std::vector<std::uint8_t> and_with(std::vector<std::uint8_t> mask, std::vector<bool> const& keep)
{
  for (std::size_t i = 0; i < mask.size(); ++i) {
    mask[i] = (mask[i] != 0 && keep[i]) ? 1 : 0;
  }
  return mask;
}

template <class Filter>
std::vector<std::uint8_t> probe_mask(Filter const& filter,
                                     cudf::column_view const& probe,
                                     std::uint32_t const* prior_words,
                                     rmm::cuda_stream_view stream)
{
  auto mask = filter.compute_mask(
    probe, prior_words, kDevice, stream, cudf::get_current_device_resource_ref());
  REQUIRE(mask != nullptr);
  REQUIRE(mask->null_count() == 0);
  return mask_to_host(mask->view(), stream);
}

}  // namespace

TEST_CASE("membership key domain classifies integer types onto four reps",
          "[dynamic_filter][key_domain]")
{
  using id          = cudf::type_id;
  auto const expect = [](id t, membership_key_rep rep, membership_key_family family) {
    auto const domain = classify_membership_key(cudf::data_type{t});
    REQUIRE(domain.has_value());
    CHECK(domain->rep == rep);
    CHECK(domain->family == family);
    CHECK(domain->native == cudf::data_type{t});
    CHECK(membership_key_supported(cudf::data_type{t}));
  };
  expect(id::INT8, membership_key_rep::i32, membership_key_family::signed_int);
  expect(id::INT16, membership_key_rep::i32, membership_key_family::signed_int);
  expect(id::INT32, membership_key_rep::i32, membership_key_family::signed_int);
  expect(id::INT64, membership_key_rep::i64, membership_key_family::signed_int);
  expect(id::UINT8, membership_key_rep::u32, membership_key_family::unsigned_int);
  expect(id::UINT16, membership_key_rep::u32, membership_key_family::unsigned_int);
  expect(id::UINT32, membership_key_rep::u32, membership_key_family::unsigned_int);
  expect(id::UINT64, membership_key_rep::u64, membership_key_family::unsigned_int);
  // Strings ride the u64 rep as fingerprints.
  expect(id::STRING, membership_key_rep::u64, membership_key_family::string_hash);

  for (auto const t : {id::EMPTY,
                       id::BOOL8,
                       id::FLOAT32,
                       id::FLOAT64,
                       id::TIMESTAMP_DAYS,
                       id::TIMESTAMP_MICROSECONDS,
                       id::DURATION_SECONDS,
                       id::DECIMAL32,
                       id::DECIMAL64,
                       id::DECIMAL128,
                       id::DICTIONARY32,
                       id::LIST,
                       id::STRUCT}) {
    CHECK_FALSE(classify_membership_key(cudf::data_type{t}).has_value());
    CHECK_FALSE(membership_key_supported(cudf::data_type{t}));
  }

  // The host mirror of the probe dispatch: same-signedness integer carriers only, and a
  // materialized STRING column only for the string family.
  auto const signed_domain   = *classify_membership_key(cudf::data_type{id::INT64});
  auto const unsigned_domain = *classify_membership_key(cudf::data_type{id::UINT32});
  auto const string_domain   = *classify_membership_key(cudf::data_type{id::STRING});
  for (auto const t : {id::INT8, id::INT16, id::INT32, id::INT64}) {
    CHECK(membership_probe_compatible(signed_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(unsigned_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(string_domain, cudf::data_type{t}));
  }
  for (auto const t : {id::UINT8, id::UINT16, id::UINT32, id::UINT64}) {
    CHECK(membership_probe_compatible(unsigned_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(signed_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(string_domain, cudf::data_type{t}));
  }
  for (auto const t : {id::FLOAT64, id::TIMESTAMP_DAYS, id::DECIMAL64, id::STRING, id::BOOL8}) {
    CHECK_FALSE(membership_probe_compatible(signed_domain, cudf::data_type{t}));
    CHECK_FALSE(membership_probe_compatible(unsigned_domain, cudf::data_type{t}));
  }
  CHECK(membership_probe_compatible(string_domain, cudf::data_type{id::STRING}));
  // The u64 rep is shared with UINT64 keys, but the families never cross: a UINT64 probe is not
  // a fingerprint and a dictionary-encoded carrier is not a string the kernel can hash.
  for (auto const t : {id::UINT64, id::DICTIONARY32, id::INT64, id::FLOAT64}) {
    CHECK_FALSE(membership_probe_compatible(string_domain, cudf::data_type{t}));
  }

  // The three filters' type gates are the same predicate.
  for (auto const t : {id::INT8, id::INT64, id::UINT16, id::UINT64, id::FLOAT64, id::STRING}) {
    CHECK(sirius_dynamic_bloom_filter::supports(cudf::data_type{t}) ==
          membership_key_supported(cudf::data_type{t}));
  }
}

TEST_CASE("hash IN-list set bytes are sized at the key rep, not the build carrier",
          "[dynamic_filter][key_domain]")
{
  using id            = cudf::type_id;
  auto const bytes_of = [](id t) {
    return sirius_dynamic_in_list_filter::estimated_set_bytes(1000, cudf::data_type{t});
  };
  CHECK(bytes_of(id::INT8) == bytes_of(id::INT32));
  CHECK(bytes_of(id::INT16) == bytes_of(id::INT32));
  CHECK(bytes_of(id::UINT8) == bytes_of(id::INT32));
  CHECK(bytes_of(id::UINT32) == bytes_of(id::INT32));
  CHECK(bytes_of(id::UINT64) == bytes_of(id::INT64));
  CHECK(bytes_of(id::INT64) == 2 * bytes_of(id::INT32));
  CHECK(bytes_of(id::STRING) == bytes_of(id::INT64));  // 8-byte fingerprints, not string bytes
}

// The three filters must answer identically whatever carrier the probe arrives at, with or
// without a prior keep-mask, for both signed reps. IN-lists are exact against a host oracle; Bloom
// is compared against its own widest-carrier answer, which must also contain every inserted key.
TEST_CASE("membership masks are identical across probe carriers and prior masks",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Every value fits INT8 so each carrier sees the same logical probe.
  std::vector<std::int64_t> const key_values{1, 2, 3, 50, 100, 127, -128, -7};
  std::vector<std::int64_t> probe_values;
  std::vector<std::uint8_t> expected;
  std::vector<bool> keep;
  for (std::int64_t v = -128; v <= 127; ++v) {  // 256 rows: crosses prior-mask word boundaries
    probe_values.push_back(v);
    expected.push_back(std::find(key_values.begin(), key_values.end(), v) != key_values.end() ? 1
                                                                                              : 0);
    keep.push_back((v % 3) == 0);
  }
  auto prior                 = upload_prior_mask(keep, stream);
  auto const* prior_words    = static_cast<std::uint32_t const*>(prior.data());
  auto const expected_masked = and_with(expected, keep);

  auto const run = [&](auto key_tag) {
    using key_type  = decltype(key_tag);
    auto const keys = make_typed<key_type>(key_values, stream);
    sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
    sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
    sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
    CHECK(in_list.domain().native == keys->type());

    // Reference: the native (widest) carrier with no prior.
    auto const native          = make_typed<std::int64_t>(probe_values, stream);
    auto const bloom_reference = probe_mask(bloom, native->view(), nullptr, stream);
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != 0) { REQUIRE(bloom_reference[i] == 1); }  // no false negatives
    }
    auto const bloom_reference_masked = and_with(bloom_reference, keep);

    for_each_signed_carrier([&](auto carrier_tag) {
      using carrier_type = decltype(carrier_tag);
      auto const probe   = make_typed<carrier_type>(probe_values, stream);
      INFO("key=" << static_cast<int>(keys->type().id())
                  << " carrier=" << static_cast<int>(probe->type().id()));
      CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(in_list, probe->view(), prior_words, stream) == expected_masked);
      CHECK(probe_mask(small_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(small_list, probe->view(), prior_words, stream) == expected_masked);
      CHECK(probe_mask(bloom, probe->view(), nullptr, stream) == bloom_reference);
      CHECK(probe_mask(bloom, probe->view(), prior_words, stream) == bloom_reference_masked);
    });
  };
  run(std::int32_t{});
  run(std::int64_t{});
}

TEST_CASE("INT8 and INT16 build keys publish carrier-typed 32-bit sets",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Probe values: two hits, an INT8-range miss, and values only a wider carrier can hold (an
  // INT16 value, an INT32 value, and one beyond INT32 that must range-check to non-member).
  std::vector<std::int64_t> const probe_values{7, -5, 100, 0, 300, 70'000, 5'000'000'000LL};

  auto const run = [&](auto key_tag, std::vector<std::int64_t> key_values) {
    using key_type  = decltype(key_tag);
    auto const keys = make_typed<key_type>(key_values, stream);
    REQUIRE(sirius_dynamic_in_list_filter::supports(keys->view()));
    REQUIRE(sirius_dynamic_small_in_list_filter::supports(keys->view()));
    REQUIRE(sirius_dynamic_bloom_filter::supports(keys->type()));

    sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
    sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
    sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
    for (auto const* domain : {&in_list.domain(), &small_list.domain(), &bloom.domain()}) {
      CHECK(domain->rep == membership_key_rep::i32);
      CHECK(domain->family == membership_key_family::signed_int);
      CHECK(domain->native == keys->type());
    }
    CHECK(in_list.has_persistent_set());

    for_each_signed_carrier([&](auto carrier_tag) {
      using carrier_type = decltype(carrier_tag);
      std::vector<std::int64_t> values;
      std::vector<std::uint8_t> expected;
      for (auto const v : probe_values) {
        if (v < std::numeric_limits<carrier_type>::min() ||
            v > std::numeric_limits<carrier_type>::max()) {
          continue;
        }
        values.push_back(v);
        expected.push_back(
          std::find(key_values.begin(), key_values.end(), v) != key_values.end() ? 1 : 0);
      }
      auto const probe = make_typed<carrier_type>(values, stream);
      INFO("key=" << static_cast<int>(keys->type().id())
                  << " carrier=" << static_cast<int>(probe->type().id()));
      CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(small_list, probe->view(), nullptr, stream) == expected);
      auto const bloom_mask = probe_mask(bloom, probe->view(), nullptr, stream);
      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != 0) { CHECK(bloom_mask[i] == 1); }
      }
      // Anything beyond INT32 can never be a member of a 32-bit set.
      for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] > std::numeric_limits<std::int32_t>::max()) { CHECK(bloom_mask[i] == 0); }
      }
    });
  };
  run(std::int8_t{}, {7, -5, 100});
  run(std::int16_t{}, {7, -5, 300});
}

TEST_CASE("unsigned keys probe every unsigned carrier and decline signed ones",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  std::vector<std::uint64_t> const probe_values{
    0, 1, 200, 255, 60'000, 3'000'000'000ULL, 5'000'000'000ULL};

  auto const run = [&](
                     auto key_tag, std::vector<std::uint64_t> key_values, membership_key_rep rep) {
    using key_type  = decltype(key_tag);
    auto const keys = make_unsigned<key_type>(key_values, stream);
    REQUIRE(membership_key_supported(keys->type()));

    sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
    sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
    sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
    for (auto const* domain : {&in_list.domain(), &small_list.domain(), &bloom.domain()}) {
      CHECK(domain->rep == rep);
      CHECK(domain->family == membership_key_family::unsigned_int);
    }

    for_each_unsigned_carrier([&](auto carrier_tag) {
      using carrier_type = decltype(carrier_tag);
      std::vector<std::uint64_t> values;
      std::vector<std::uint8_t> expected;
      for (auto const v : probe_values) {
        if (v > std::numeric_limits<carrier_type>::max()) { continue; }
        values.push_back(v);
        expected.push_back(
          std::find(key_values.begin(), key_values.end(), v) != key_values.end() ? 1 : 0);
      }
      auto const probe = make_unsigned<carrier_type>(values, stream);
      INFO("key=" << static_cast<int>(keys->type().id())
                  << " carrier=" << static_cast<int>(probe->type().id()));
      CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
      CHECK(probe_mask(small_list, probe->view(), nullptr, stream) == expected);
      auto const bloom_mask = probe_mask(bloom, probe->view(), nullptr, stream);
      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != 0) { CHECK(bloom_mask[i] == 1); }
      }
    });

    // Signed carriers are a semantic mismatch, not a width one.
    auto const signed_probe = make_typed<std::int32_t>({0, 1, 200}, stream);
    CHECK(in_list.compute_mask(signed_probe->view(), kDevice, stream, mr) == nullptr);
    CHECK(small_list.compute_mask(signed_probe->view(), kDevice, stream, mr) == nullptr);
    CHECK(bloom.compute_mask(signed_probe->view(), kDevice, stream, mr) == nullptr);
  };
  run(std::uint8_t{}, {1, 200, 255}, membership_key_rep::u32);
  run(std::uint16_t{}, {1, 200, 60'000}, membership_key_rep::u32);
  run(std::uint32_t{}, {0, 200, 3'000'000'000ULL}, membership_key_rep::u32);
  run(std::uint64_t{}, {0, 200, 5'000'000'000ULL}, membership_key_rep::u64);
}

TEST_CASE("unsigned hash IN-list reserves the maximum as its sentinel, so 0 is exact",
          "[dynamic_filter][probe][key_domain]")
{
  auto const stream      = cudf::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();
  constexpr auto u32_max = std::numeric_limits<std::uint32_t>::max();
  constexpr auto u64_max = std::numeric_limits<std::uint64_t>::max();

  SECTION("u32 set: 0 is a real key that can be absent; UINT32_MAX is kept conservatively")
  {
    auto const keys = make_unsigned<std::uint32_t>({5}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto const probe = make_unsigned<std::uint32_t>({0, 5, u32_max}, stream);
    CHECK(probe_mask(filter, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1, 1});

    // A UINT64 probe equal to the u32 sentinel narrows onto it and is kept; one past the u32
    // range is a definite non-member.
    auto const wide =
      make_unsigned<std::uint64_t>({0, u32_max, std::uint64_t{u32_max} + 1}, stream);
    CHECK(probe_mask(filter, wide->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1, 0});
  }

  SECTION("u64 set: UINT64_MAX is kept conservatively, 0 is exact")
  {
    auto const keys = make_unsigned<std::uint64_t>({5}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto const probe = make_unsigned<std::uint64_t>({0, 5, u64_max}, stream);
    CHECK(probe_mask(filter, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1, 1});
    // A UINT32 probe widened into a u64 set never lands on the sentinel.
    auto const narrow = make_unsigned<std::uint32_t>({u32_max, 5}, stream);
    CHECK(probe_mask(filter, narrow->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 1});
  }

  SECTION("a build key equal to the unsigned sentinel is still kept on probe")
  {
    auto const keys = make_unsigned<std::uint32_t>({u32_max, 3}, stream);
    sirius_dynamic_in_list_filter filter{keys->view(), stream, mr};
    auto const probe = make_unsigned<std::uint32_t>({u32_max, 3, 4}, stream);
    CHECK(probe_mask(filter, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{1, 1, 0});
  }
}

//===----------------------------------------------------------------------===//
// STRING keys: 64-bit fingerprints
//===----------------------------------------------------------------------===//

namespace {

// Reference XXH64 (the canonical algorithm, seed as given) over a byte string. Independent of
// both cudf's column API and the in-kernel hasher, so a drift in either side's seed or byte view
// shows up here rather than as silently dropped join matches.
std::uint64_t xxh64_reference(std::string const& input, std::uint64_t seed)
{
  constexpr std::uint64_t p1 = 0x9E3779B185EBCA87ULL;
  constexpr std::uint64_t p2 = 0xC2B2AE3D27D4EB4FULL;
  constexpr std::uint64_t p3 = 0x165667B19E3779F9ULL;
  constexpr std::uint64_t p4 = 0x85EBCA77C2B2AE63ULL;
  constexpr std::uint64_t p5 = 0x27D4EB2F165667C5ULL;
  auto const rotl            = [](std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
  auto const round           = [&](std::uint64_t acc, std::uint64_t in) {
    acc += in * p2;
    acc = rotl(acc, 31);
    return acc * p1;
  };
  auto const merge = [&](std::uint64_t acc, std::uint64_t v) {
    acc ^= round(0, v);
    return acc * p1 + p4;
  };
  auto const* p         = reinterpret_cast<unsigned char const*>(input.data());
  auto const* const end = p + input.size();
  auto const read64     = [](unsigned char const* q) {
    std::uint64_t v = 0;
    std::memcpy(&v, q, sizeof v);
    return v;
  };
  auto const read32 = [](unsigned char const* q) {
    std::uint32_t v = 0;
    std::memcpy(&v, q, sizeof v);
    return v;
  };

  std::uint64_t h = 0;
  if (input.size() >= 32) {
    std::uint64_t v1 = seed + p1 + p2;
    std::uint64_t v2 = seed + p2;
    std::uint64_t v3 = seed;
    std::uint64_t v4 = seed - p1;
    do {
      v1 = round(v1, read64(p));
      v2 = round(v2, read64(p + 8));
      v3 = round(v3, read64(p + 16));
      v4 = round(v4, read64(p + 24));
      p += 32;
    } while (p <= end - 32);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1);
    h = merge(h, v2);
    h = merge(h, v3);
    h = merge(h, v4);
  } else {
    h = seed + p5;
  }
  h += static_cast<std::uint64_t>(input.size());
  while (p + 8 <= end) {
    h ^= round(0, read64(p));
    h = rotl(h, 27) * p1 + p4;
    p += 8;
  }
  if (p + 4 <= end) {
    h ^= static_cast<std::uint64_t>(read32(p)) * p1;
    h = rotl(h, 23) * p2 + p3;
    p += 4;
  }
  while (p < end) {
    h ^= static_cast<std::uint64_t>(*p) * p5;
    h = rotl(h, 11) * p1;
    ++p;
  }
  h ^= h >> 33;
  h *= p2;
  h ^= h >> 29;
  h *= p3;
  h ^= h >> 32;
  return h;
}

// Upload a STRING column; a nullopt entry is a null row (its offsets span zero bytes).
std::unique_ptr<cudf::column> make_strings(std::vector<std::optional<std::string>> const& values,
                                           rmm::cuda_stream_view stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  auto const n  = static_cast<cudf::size_type>(values.size());
  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(n) + 1, 0);
  std::string chars;
  cudf::size_type null_count = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i].has_value()) {
      chars += *values[i];
    } else {
      ++null_count;
    }
    offsets[i + 1] = static_cast<cudf::size_type>(chars.size());
  }
  auto offsets_col = make_values(offsets, cudf::data_type{cudf::type_id::INT32}, stream);
  rmm::device_buffer chars_buf{chars.data(), chars.size(), stream, mr};
  rmm::device_buffer null_mask{};
  if (null_count > 0) {
    null_mask = cudf::create_null_mask(n, cudf::mask_state::ALL_VALID, stream, mr);
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (!values[i].has_value()) {
        cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()),
                            static_cast<cudf::size_type>(i),
                            static_cast<cudf::size_type>(i) + 1,
                            false,
                            stream);
      }
    }
  }
  stream.synchronize();
  return cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), null_count, std::move(null_mask));
}

// Same, from plain strings (a distinct name keeps brace-list calls unambiguous).
std::unique_ptr<cudf::column> make_strings_from(std::vector<std::string> const& values,
                                                rmm::cuda_stream_view stream)
{
  std::vector<std::optional<std::string>> wrapped(values.begin(), values.end());
  return make_strings(wrapped, stream);
}

std::vector<std::uint64_t> device_fingerprints(cudf::column_view const& strings,
                                               rmm::cuda_stream_view stream)
{
  auto const hashed = cudf::hashing::xxhash_64(cudf::table_view{{strings}},
                                               cudf::DEFAULT_HASH_SEED,
                                               stream,
                                               cudf::get_current_device_resource_ref());
  REQUIRE(hashed->type().id() == cudf::type_id::UINT64);
  std::vector<std::uint64_t> host(static_cast<std::size_t>(hashed->size()));
  REQUIRE(cudaMemcpyAsync(host.data(),
                          hashed->view().data<std::uint64_t>(),
                          host.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  return host;
}

// Strings chosen to cover every XXH64 tail path: empty, < 4, 4..7, 8..31 bytes, exactly 32, a
// multiple of 32 plus each tail, multi-byte UTF-8, embedded NUL, and a 1 KiB string.
std::vector<std::string> const kStringCorpus = {
  "",
  "a",
  "abc",
  "abcd",
  "abcdefg",
  "abcdefgh",
  "AAAAAAAAAAAAAAAA",                                                              // 16
  "0123456789abcdef0123456789abcde",                                               // 31
  "0123456789abcdef0123456789abcdef",                                              // 32
  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",             // 65
  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234",         // 69
  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789ab",  // 76
  std::string("nul\0inside", 10),
  "\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80",  // e-acute, euro sign, emoji
  "AAAAAAAAAAAAAAAB",                      // 16, differs from the 16 A's in the last byte
  std::string(1024, 'z'),
  std::string(1023, 'z') + "y",
};

}  // namespace

TEST_CASE("string fingerprints: cudf::hashing::xxhash_64 matches the XXH64 reference",
          "[dynamic_filter][probe][key_domain][string]")
{
  // The build side hashes with the column API; the probe side hashes in-kernel with
  // XXHash_64<string_view>. Both must equal canonical XXH64(seed = cudf::DEFAULT_HASH_SEED) over
  // the raw UTF-8 bytes, which is the contract the membership tests below rely on.
  auto const stream = cudf::get_default_stream();
  auto const column = make_strings_from(kStringCorpus, stream);
  auto const device = device_fingerprints(column->view(), stream);
  REQUIRE(device.size() == kStringCorpus.size());
  for (std::size_t i = 0; i < kStringCorpus.size(); ++i) {
    INFO("corpus[" << i << "] length=" << kStringCorpus[i].size());
    CHECK(device[i] == xxh64_reference(kStringCorpus[i], cudf::DEFAULT_HASH_SEED));
  }
  // Distinct corpus strings have distinct fingerprints (a sanity floor for the oracle below).
  auto sorted = device;
  std::sort(sorted.begin(), sorted.end());
  CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
}

TEST_CASE("string keys probe STRING columns through in-kernel fingerprints",
          "[dynamic_filter][probe][key_domain][string]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Build keys: every other corpus entry (includes the empty string and the 1 KiB string). The
  // probe is the whole corpus plus near-misses of build keys, repeated to cross a prior-mask word
  // boundary.
  std::vector<std::string> key_values;
  for (std::size_t i = 0; i < kStringCorpus.size(); i += 2) {
    key_values.push_back(kStringCorpus[i]);
  }
  std::vector<std::string> probe_values;
  for (int rep = 0; rep < 3; ++rep) {
    for (auto const& s : kStringCorpus) {
      probe_values.push_back(s);
      probe_values.push_back(s + "x");  // near-miss: same prefix, one more byte
    }
  }
  REQUIRE(probe_values.size() > 64);

  // Host oracle on fingerprints (what the filter can see), which for this corpus equals exact
  // string membership because the fingerprints are pairwise distinct.
  std::vector<std::uint64_t> key_prints;
  for (auto const& k : key_values) {
    key_prints.push_back(xxh64_reference(k, cudf::DEFAULT_HASH_SEED));
  }
  std::vector<std::uint8_t> expected;
  std::vector<bool> keep;
  for (std::size_t i = 0; i < probe_values.size(); ++i) {
    auto const print = xxh64_reference(probe_values[i], cudf::DEFAULT_HASH_SEED);
    bool const hit   = std::find(key_prints.begin(), key_prints.end(), print) != key_prints.end();
    bool const exact =
      std::find(key_values.begin(), key_values.end(), probe_values[i]) != key_values.end();
    REQUIRE(hit == exact);
    expected.push_back(hit ? 1 : 0);
    keep.push_back(i % 3 == 0);
  }
  auto prior                 = upload_prior_mask(keep, stream);
  auto const* prior_words    = static_cast<std::uint32_t const*>(prior.data());
  auto const expected_masked = and_with(expected, keep);

  auto const keys  = make_strings_from(key_values, stream);
  auto const probe = make_strings_from(probe_values, stream);
  REQUIRE(sirius_dynamic_in_list_filter::supports(keys->view()));
  REQUIRE(sirius_dynamic_small_in_list_filter::supports(keys->view()));
  REQUIRE(sirius_dynamic_bloom_filter::supports(keys->type()));

  sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
  sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
  sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
  for (auto const* domain : {&in_list.domain(), &small_list.domain(), &bloom.domain()}) {
    CHECK(domain->rep == membership_key_rep::u64);
    CHECK(domain->family == membership_key_family::string_hash);
    CHECK(domain->native.id() == cudf::type_id::STRING);
  }
  REQUIRE(in_list.has_persistent_set());

  CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
  CHECK(probe_mask(in_list, probe->view(), prior_words, stream) == expected_masked);
  CHECK(probe_mask(small_list, probe->view(), nullptr, stream) == expected);
  CHECK(probe_mask(small_list, probe->view(), prior_words, stream) == expected_masked);

  auto const bloom_mask = probe_mask(bloom, probe->view(), nullptr, stream);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != 0) { CHECK(bloom_mask[i] == 1); }  // no false negatives
  }
  CHECK(probe_mask(bloom, probe->view(), prior_words, stream) == and_with(bloom_mask, keep));
}

TEST_CASE("string keys: a hash IN-list past the small-list cap builds a fingerprint set",
          "[dynamic_filter][probe][key_domain][string]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // 1000 keys "k<i>" for even i; probe every i plus one empty string and one long string.
  std::vector<std::string> key_values;
  for (int i = 0; i < 2000; i += 2) {
    key_values.push_back("k" + std::to_string(i));
  }
  std::vector<std::string> probe_values;
  std::vector<std::uint8_t> expected;
  for (int i = 0; i < 2000; ++i) {
    probe_values.push_back("k" + std::to_string(i));
    expected.push_back(i % 2 == 0 ? 1 : 0);
  }
  probe_values.emplace_back("");
  expected.push_back(0);
  probe_values.emplace_back(std::string(300, 'k'));
  expected.push_back(0);

  auto const keys = make_strings_from(key_values, stream);
  REQUIRE_FALSE(sirius_dynamic_small_in_list_filter::supports(keys->view()));  // above the cap
  sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
  sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
  REQUIRE(in_list.size() == key_values.size());

  auto const probe = make_strings_from(probe_values, stream);
  CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == expected);
  auto const bloom_mask  = probe_mask(bloom, probe->view(), nullptr, stream);
  std::size_t bloom_hits = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != 0) { REQUIRE(bloom_mask[i] == 1); }
    bloom_hits += bloom_mask[i];
  }
  // 16 bits/key: the false-positive rate is far below 50%, so misses must mostly fail.
  CHECK(bloom_hits < 1000 + 500);
}

TEST_CASE("string keys: null probe strings are non-members and keep the probe's null mask",
          "[dynamic_filter][probe][key_domain][string]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const keys = make_strings({"apple", "", "cherry"}, stream);
  sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
  sirius_dynamic_small_in_list_filter small_list{keys->view(), stream, mr};
  sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};

  std::vector<std::optional<std::string>> const probe_values{
    "apple", std::nullopt, "", "banana", std::nullopt, "cherry"};
  auto const probe = make_strings(probe_values, stream);
  REQUIRE(probe->null_count() == 2);

  auto const check = [&](auto const& filter, bool exact) {
    auto mask = filter.compute_mask(probe->view(), kDevice, stream, mr);
    REQUIRE(mask != nullptr);
    CHECK(mask->null_count() == 2);  // the probe's null mask rides along, as for integers
    auto const host = mask_to_host(mask->view(), stream);
    CHECK(host[0] == 1);
    CHECK(host[1] == 0);  // null row: definite non-member even before the null mask applies
    CHECK(host[2] == 1);  // the empty string is a real key
    if (exact) { CHECK(host[3] == 0); }
    CHECK(host[4] == 0);
    CHECK(host[5] == 1);
  };
  check(in_list, true);
  check(small_list, true);
  check(bloom, false);
}

TEST_CASE("string keys: null build strings are refused by the IN-lists and compacted by Bloom",
          "[dynamic_filter][probe][key_domain][string]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const keys = make_strings({"apple", std::nullopt, "cherry"}, stream);
  CHECK_FALSE(sirius_dynamic_in_list_filter::supports(keys->view()));
  CHECK_FALSE(sirius_dynamic_small_in_list_filter::supports(keys->view()));
  CHECK(sirius_dynamic_bloom_filter::supports(keys->type()));

  // cudf's xxhash_64 maps a null row to UINT64_MAX; Bloom drops nulls before hashing so that
  // fingerprint is never inserted on a null's behalf.
  sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
  auto const probe = make_strings({"apple", "cherry", "durian"}, stream);
  auto const mask  = probe_mask(bloom, probe->view(), nullptr, stream);
  CHECK(mask[0] == 1);
  CHECK(mask[1] == 1);
}

TEST_CASE("string keys decline non-string probes and integer keys decline string probes",
          "[dynamic_filter][probe][key_domain][string]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const string_keys = make_strings({"a", "b", "c"}, stream);
  sirius_dynamic_in_list_filter string_in_list{string_keys->view(), stream, mr};
  sirius_dynamic_small_in_list_filter string_small{string_keys->view(), stream, mr};
  sirius_dynamic_bloom_filter string_bloom{string_keys->view(), stream, mr};

  // A UINT64 probe shares the rep but is not a fingerprint: it must decline, not be looked up.
  auto const u64_probe   = make_unsigned<std::uint64_t>({1, 2, 3}, stream);
  auto const int64_probe = make_int64({1, 2, 3}, stream);
  for (auto const* probe : {&u64_probe, &int64_probe}) {
    CHECK(string_in_list.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
    CHECK(string_small.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
    CHECK(string_bloom.compute_mask((*probe)->view(), kDevice, stream, mr) == nullptr);
  }

  auto const int_keys = make_unsigned<std::uint64_t>({1, 2, 3}, stream);
  sirius_dynamic_in_list_filter int_in_list{int_keys->view(), stream, mr};
  sirius_dynamic_small_in_list_filter int_small{int_keys->view(), stream, mr};
  sirius_dynamic_bloom_filter int_bloom{int_keys->view(), stream, mr};
  auto const string_probe = make_strings({"1", "2", "3"}, stream);
  CHECK(int_in_list.compute_mask(string_probe->view(), kDevice, stream, mr) == nullptr);
  CHECK(int_small.compute_mask(string_probe->view(), kDevice, stream, mr) == nullptr);
  CHECK(int_bloom.compute_mask(string_probe->view(), kDevice, stream, mr) == nullptr);
}

TEST_CASE("string keys: an empty build column builds an empty fingerprint set",
          "[dynamic_filter][probe][key_domain][string]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const keys   = make_strings_from({}, stream);
  sirius_dynamic_in_list_filter in_list{keys->view(), stream, mr};
  sirius_dynamic_bloom_filter bloom{keys->view(), stream, mr};
  auto const probe = make_strings({"", "x"}, stream);
  CHECK(probe_mask(in_list, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 0});
  CHECK(probe_mask(bloom, probe->view(), nullptr, stream) == std::vector<std::uint8_t>{0, 0});
}
