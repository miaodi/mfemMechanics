#pragma once

#include <tuple>
#include <type_traits>

namespace plugin
{
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

template <typename... Types>
struct UniqueMaterialPointTypes : std::true_type
{
};

template <typename Type, typename... Rest>
struct UniqueMaterialPointTypes<Type, Rest...>
    : std::bool_constant<( !std::is_same_v<Type, Rest> && ... ) && UniqueMaterialPointTypes<Rest...>::value>
{
};

template <typename... Materials>
class MaterialPointStateBundle
{
    static_assert( UniqueMaterialPointTypes<Materials...>::value,
                   "Each material type may appear at most once in a MaterialPointStateBundle." );

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
