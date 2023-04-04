#define CV_CPU_SIMD_FILENAME "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/modules/dnn/src/layers/layers_common.simd.hpp"
#define CV_CPU_DISPATCH_MODE AVX
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODE AVX2
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODE AVX512_SKX
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODES_ALL AVX512_SKX, AVX2, AVX, BASELINE

#undef CV_CPU_SIMD_FILENAME
