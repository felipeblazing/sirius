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

#pragma once

// Host-only (no CUDA) classification of membership dynamic-filter keys. The three membership
// filters (small IN-list, hash IN-list, Bloom), the publisher, the publish-plan validator, and the
// planner's direct-route gate consult this header instead of spelling out type lists.
//
// Two closed axes describe every supported key:
//   * the key *rep*: the device element type a set / Bloom / needle buffer is instantiated over.
//     Every supported build type maps onto one of four reps, which bounds storage-variant
//     alternatives and kernel instantiations;
//   * the key *family*: which probe adapter (see cuda/dynamic_filter_probe.cuh) converts a probe
//     column at its own carrier into the rep. A family is the correctness surface: it names the
//     probe carriers whose values are comparable to the stored keys.

// cudf
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <optional>

namespace sirius::op {

/// Device element type of a membership set. Every supported key family maps onto one of these.
enum class membership_key_rep : std::uint8_t { i32, i64, u32, u64 };

/// Probe-adapter selector. Each value names one device adapter (cuda/dynamic_filter_probe.cuh)
/// and one arm each of `classify_membership_key` and `membership_probe_compatible`; a new key
/// family adds a value here and those three arms.
///
/// `decimal` keys are fixed-point columns compared by their unscaled integer representation: a
/// probe is comparable only at the same cudf scale, and DECIMAL128 sits on the `i64` rep, which
/// its build values must fit (see `membership_build_fits_rep`).
enum class membership_key_family : std::uint8_t { signed_int, unsigned_int, decimal };

struct membership_key_domain {
  membership_key_rep rep{membership_key_rep::i32};
  membership_key_family family{membership_key_family::signed_int};
  /// Build column type the filter was constructed from (the carrier the set was published for).
  cudf::data_type native{cudf::type_id::EMPTY};
  /// cudf scale of a `decimal` key (negative for SQL scale > 0); 0 for every other family.
  std::int32_t scale{0};

  [[nodiscard]] bool operator==(membership_key_domain const&) const = default;
};

/**
 * @brief Build-side classification of a key column type
 *
 * The rep is the narrowest listed rep that holds every value of @p build_type, so a build column
 * arriving at a narrowed carrier (compressed materialization) yields a carrier-sized set and wider
 * probes range-check down into it. Returns nullopt for a type no membership filter supports.
 *
 * DECIMAL128 is the one type whose rep (`i64`) is narrower than its carrier: the classification is
 * provisional on the build column's unscaled values fitting int64, which
 * `membership_build_fits_rep` checks on the runtime column and the filters' constructors enforce.
 */
[[nodiscard]] std::optional<membership_key_domain> classify_membership_key(
  cudf::data_type build_type) noexcept;

/**
 * @brief Single source of truth for the membership filters' `supports()` type gate
 *
 * True iff `classify_membership_key(t)` has a value. Filters add their own non-type gates (the
 * small IN-list size cap, null-free keys, the DECIMAL128 range check) on top of this.
 */
[[nodiscard]] bool membership_key_supported(cudf::data_type t) noexcept;

/**
 * @brief True when every non-null value of @p keys is representable in its domain's rep
 *
 * Only a DECIMAL128 build column can fail: its unscaled values may exceed int64, and a set built by
 * truncating them would produce false negatives. That case runs a min/max reduction on @p stream
 * and reads the bounds back (synchronizing); every other supported type answers true without GPU
 * work. An unsupported type answers false. Callers that gate publication call this once before
 * consulting the filters' `supports()`; the constructors re-check and throw on a violation.
 */
[[nodiscard]] bool membership_build_fits_rep(cudf::column_view const& keys,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @brief True when a probe column of type @p probe can be adapted to @p domain
 *
 * Host mirror of the device-side adapter dispatch: a probe type this rejects is one every filter's
 * `compute_mask` declines with a null result. Signed and unsigned carriers never mix; decimal
 * probes must carry the domain's scale (a rescale is a planner cast and never reaches a filter).
 */
[[nodiscard]] bool membership_probe_compatible(membership_key_domain const& domain,
                                               cudf::data_type probe) noexcept;

/// cudf type of the device element a rep is instantiated over (INT32/INT64/UINT32/UINT64).
[[nodiscard]] cudf::data_type membership_rep_type(membership_key_rep rep) noexcept;

/// Byte width of a rep's device element.
[[nodiscard]] std::size_t membership_rep_bytes(membership_key_rep rep) noexcept;

}  // namespace sirius::op
