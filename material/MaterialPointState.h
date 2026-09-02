#pragma once

#include <cstddef>
#include <tuple>

namespace plugin
{
// The solution buffer includes the current state; point histories store it separately.
inline constexpr std::size_t SolutionHistoryCapacity = 20;
inline constexpr std::size_t MaterialStateHistoryCapacity = SolutionHistoryCapacity - 1;
static_assert( MaterialStateHistoryCapacity > 0 );

/// Marker used when an integration-point domain has no persistent state.
struct NoIntegrationPointState
{
};

template <typename Material>
struct MaterialPointTraits
{
    /// Persistent state required by one integration point of Material.
    using State = NoIntegrationPointState;
};

template <typename Material>
using MaterialPointState = typename MaterialPointTraits<Material>::State;

template <typename Material>
struct MaterialPointStateSlot
{
    MaterialPointState<Material> Value{};
};

template <typename... Materials>
class MaterialPointStateBundle
{
public:
    template <typename Material>
    MaterialPointState<Material>& Get()
    {
        return std::get<MaterialPointStateSlot<Material>>( mStates ).Value;
    }

    template <typename Material>
    const MaterialPointState<Material>& Get() const
    {
        return std::get<MaterialPointStateSlot<Material>>( mStates ).Value;
    }

private:
    std::tuple<MaterialPointStateSlot<Materials>...> mStates;
};
} // namespace plugin
