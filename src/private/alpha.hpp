#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

namespace mgr_detail {
// IEEE-754 bit classification is independent of the caller/worker FP mode.
enum class Alpha { Invalid, Zero, One, Interior };
inline Alpha classify_alpha(const double& value) noexcept {
    static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53,
                  "RIFE requires IEEE-754 binary64 double");
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr std::uint64_t sign = UINT64_C(0x8000000000000000);
    constexpr std::uint64_t one = UINT64_C(0x3ff0000000000000);
    const std::uint64_t magnitude = bits & ~sign;
    if (magnitude == 0) return Alpha::Zero;
    if ((bits & sign) || magnitude > one) return Alpha::Invalid;
    return magnitude == one ? Alpha::One : Alpha::Interior;
}
inline bool alpha_endpoint(Alpha value) noexcept {
    return value == Alpha::Zero || value == Alpha::One;
}
}
