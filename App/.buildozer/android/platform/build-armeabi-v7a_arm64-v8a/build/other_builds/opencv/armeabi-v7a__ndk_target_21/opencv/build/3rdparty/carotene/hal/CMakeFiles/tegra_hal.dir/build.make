# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /dcs/20/u2030590/.local/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /dcs/20/u2030590/.local/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build

# Include any dependencies generated for this target.
include 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/compiler_depend.make

# Include the progress variables for this target.
include 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/progress.make

# Include the compile flags for this target's objects.
include 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/flags.make

3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.o: 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/flags.make
3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.o: /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/3rdparty/carotene/hal/dummy.cpp
3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.o: 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.o"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.o -MF CMakeFiles/tegra_hal.dir/dummy.cpp.o.d -o CMakeFiles/tegra_hal.dir/dummy.cpp.o -c /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/3rdparty/carotene/hal/dummy.cpp

3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tegra_hal.dir/dummy.cpp.i"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/3rdparty/carotene/hal/dummy.cpp > CMakeFiles/tegra_hal.dir/dummy.cpp.i

3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tegra_hal.dir/dummy.cpp.s"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/3rdparty/carotene/hal/dummy.cpp -o CMakeFiles/tegra_hal.dir/dummy.cpp.s

# Object files for target tegra_hal
tegra_hal_OBJECTS = \
"CMakeFiles/tegra_hal.dir/dummy.cpp.o"

# External object files for target tegra_hal
tegra_hal_EXTERNAL_OBJECTS = \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/absdiff.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/accumulate.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/add.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/add_weighted.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/bitwise.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/blur.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/canny.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/channel_extract.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/channels_combine.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/cmp.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/colorconvert.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/common.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convert.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convert_depth.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convert_scale.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convolution.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/count_nonzero.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/div.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/dot_product.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/dummy.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/fast.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/fill_minmaxloc.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/flip.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/gaussian_blur.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/in_range.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/integral.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/laplacian.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/magnitude.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/meanstddev.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/median_filter.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/min_max.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/minmaxloc.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/morph.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/mul.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/norm.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/opticalflow.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/phase.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/pyramid.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/reduce.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/remap.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/resize.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/scharr.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/separable_filter.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/sobel.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/sub.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/sum.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/template_matching.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/threshold.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/warp_affine.cpp.o" \
"/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/warp_perspective.cpp.o"

3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/dummy.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/absdiff.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/accumulate.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/add.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/add_weighted.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/bitwise.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/blur.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/canny.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/channel_extract.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/channels_combine.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/cmp.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/colorconvert.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/common.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convert.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convert_depth.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convert_scale.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/convolution.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/count_nonzero.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/div.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/dot_product.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/dummy.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/fast.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/fill_minmaxloc.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/flip.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/gaussian_blur.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/in_range.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/integral.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/laplacian.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/magnitude.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/meanstddev.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/median_filter.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/min_max.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/minmaxloc.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/morph.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/mul.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/norm.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/opticalflow.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/phase.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/pyramid.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/reduce.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/remap.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/resize.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/scharr.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/separable_filter.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/sobel.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/sub.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/sum.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/template_matching.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/threshold.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/warp_affine.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/carotene/CMakeFiles/carotene_objs.dir/src/warp_perspective.cpp.o
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/build.make
3rdparty/lib/armeabi-v7a/libtegra_hal.a: 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../lib/armeabi-v7a/libtegra_hal.a"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal && $(CMAKE_COMMAND) -P CMakeFiles/tegra_hal.dir/cmake_clean_target.cmake
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tegra_hal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/build: 3rdparty/lib/armeabi-v7a/libtegra_hal.a
.PHONY : 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/build

3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/clean:
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal && $(CMAKE_COMMAND) -P CMakeFiles/tegra_hal.dir/cmake_clean.cmake
.PHONY : 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/clean

3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/depend:
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/3rdparty/carotene/hal /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/armeabi-v7a__ndk_target_21/opencv/build/3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rdparty/carotene/hal/CMakeFiles/tegra_hal.dir/depend

