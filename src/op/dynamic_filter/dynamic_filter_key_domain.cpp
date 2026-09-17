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

#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"

#include "helper/numeric_narrowing.hpp"

#include <limits>

namespace sirius::op {

// Each key family owns one arm here and a matching adapter arm in
// cuda/dynamic_filter_probe.cuh's dispatch_probe_adapter. Keep the arms explicit; a
// cudf::type_dispatcher here would silently widen the correctness surface.
std::optional<membership_key_domain> classify_membership_key(cudf::data_type build_type) noexcept
{
  using rep    = membership_key_rep;
  using family = membership_key_family;
  switch (build_type.id()) {
    // Signed integers, including the INT8/INT16 carriers compressed materialization narrows a
    // wider key to: the set is built at the narrowest rep that holds the carrier.
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
      return membership_key_domain{rep::i32, family::signed_int, build_type};
    case cudf::type_id::INT64:
      return membership_key_domain{rep::i64, family::signed_int, build_type};
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
      return membership_key_domain{rep::u32, family::unsigned_int, build_type};
    case cudf::type_id::UINT64:
      return membership_key_domain{rep::u64, family::unsigned_int, build_type};
    // Fixed-point keys are their unscaled integer storage at one scale. A DECIMAL64 key pinned
    // narrow arrives as DECIMAL32 at the same scale and builds a 32-bit set, exactly like an
    // INT16 carrier of an INTEGER key. DECIMAL128 has no 16-byte rep (cuco's static_set caps keys
    // at 8 bytes); it classifies onto i64 provisionally and membership_build_fits_rep decides.
    case cudf::type_id::DECIMAL32:
      return membership_key_domain{rep::i32, family::decimal, build_type, build_type.scale()};
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
      return membership_key_domain{rep::i64, family::decimal, build_type, build_type.scale()};
    // Every other type (temporal, floating-point, string, nested) declines.
    default: return std::nullopt;
  }
}

bool membership_key_supported(cudf::data_type t) noexcept
{
  return classify_membership_key(t).has_value();
}

bool membership_build_fits_rep(cudf::column_view const& keys,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const domain = classify_membership_key(keys.type());
  if (!domain.has_value()) { return false; }
  if (keys.type().id() != cudf::type_id::DECIMAL128) { return true; }
  // No non-null value: nothing can be truncated (Bloom compacts nulls; the IN-lists reject them).
  if (keys.size() == 0 || keys.null_count() == keys.size()) { return true; }
  // Exact unscaled bounds; nullopt only for a scale outside the SQL range, which DuckDB never
  // produces, and is treated as not fitting so nothing is built on an unverified column.
  auto const range = sirius::compute_exact_numeric_range(keys, stream, mr);
  if (!range.has_value() || range->domain != sirius::numeric_range_domain::DECIMAL) {
    return false;
  }
  return range->minimum >= static_cast<__int128_t>(std::numeric_limits<std::int64_t>::min()) &&
         range->maximum <= static_cast<__int128_t>(std::numeric_limits<std::int64_t>::max());
}

// Mirrors dispatch_probe_adapter: the probe carriers each family's adapter accepts.
bool membership_probe_compatible(membership_key_domain const& domain,
                                 cudf::data_type probe) noexcept
{
  switch (domain.family) {
    case membership_key_family::signed_int:
      switch (probe.id()) {
        case cudf::type_id::INT8:
        case cudf::type_id::INT16:
        case cudf::type_id::INT32:
        case cudf::type_id::INT64: return true;
        default: return false;
      }
    case membership_key_family::unsigned_int:
      switch (probe.id()) {
        case cudf::type_id::UINT8:
        case cudf::type_id::UINT16:
        case cudf::type_id::UINT32:
        case cudf::type_id::UINT64: return true;
        default: return false;
      }
    case membership_key_family::decimal:
      switch (probe.id()) {
        case cudf::type_id::DECIMAL32:
        case cudf::type_id::DECIMAL64:
        case cudf::type_id::DECIMAL128: return probe.scale() == domain.scale;
        default: return false;
      }
  }
  return false;
}

cudf::data_type membership_rep_type(membership_key_rep rep) noexcept
{
  switch (rep) {
    case membership_key_rep::i32: return cudf::data_type{cudf::type_id::INT32};
    case membership_key_rep::i64: return cudf::data_type{cudf::type_id::INT64};
    case membership_key_rep::u32: return cudf::data_type{cudf::type_id::UINT32};
    case membership_key_rep::u64: return cudf::data_type{cudf::type_id::UINT64};
  }
  return cudf::data_type{cudf::type_id::EMPTY};
}

std::size_t membership_rep_bytes(membership_key_rep rep) noexcept
{
  switch (rep) {
    case membership_key_rep::i32:
    case membership_key_rep::u32: return sizeof(std::int32_t);
    case membership_key_rep::i64:
    case membership_key_rep::u64: return sizeof(std::int64_t);
  }
  return sizeof(std::int64_t);
}

}  // namespace sirius::op
