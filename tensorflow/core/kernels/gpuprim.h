#pragma once

#if GOOGLE_CUDA 
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/cub/device/device_reduce.cuh"
#include "third_party/cub/device/device_segmented_reduce.cuh"
#include "third_party/cub/device/device_histogram.cuh"
#include "third_party/cub/iterator/counting_input_iterator.cuh"
#include "third_party/cub/iterator/transform_input_iterator.cuh"
#include "third_party/cub/warp/warp_reduce.cuh"
#include "third_party/cub/thread/thread_operators.cuh"
#include "third_party/gpus/cuda/include/cusparse.h"
#include "third_party/cub/block/block_load.cuh"
#include "third_party/cub/block/block_scan.cuh"
#include "third_party/cub/block/block_store.cuh"

namespace gpuprim = ::cub;
#else
#include "rocm/include/hipcub/hipcub.hpp"
namespace gpuprim = ::hipcub;
#endif

#if GOOGLE_CUDA 
// Required for sorting Eigen::half
namespace cub {
template <>
struct NumericTraits<Eigen::half>
    : BaseTraits<FLOATING_POINT, true, false, unsigned short int, Eigen::half> {
};
}  // namespace cub
#else
namespace rocprim {
  namespace detail {
    template<>
    struct radix_key_codec_base<Eigen::half> : radix_key_codec_floating<Eigen::half, unsigned short> { };
  };
};
#endif


