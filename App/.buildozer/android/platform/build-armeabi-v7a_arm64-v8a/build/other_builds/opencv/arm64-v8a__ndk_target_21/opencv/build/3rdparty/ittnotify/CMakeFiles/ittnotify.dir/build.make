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
CMAKE_SOURCE_DIR = /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build

# Include any dependencies generated for this target.
include 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/compiler_depend.make

# Include the progress variables for this target.
include 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/progress.make

# Include the compile flags for this target's objects.
include 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/flags.make

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/flags.make
3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o: /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/ittnotify_static.c
3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o -MF CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o.d -o CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o -c /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/ittnotify_static.c

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.i"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/ittnotify_static.c > CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.i

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.s"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/ittnotify_static.c -o CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.s

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/flags.make
3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o: /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/jitprofiling.c
3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o -MF CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o.d -o CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o -c /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/jitprofiling.c

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.i"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/jitprofiling.c > CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.i

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.s"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && /dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify/src/ittnotify/jitprofiling.c -o CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.s

# Object files for target ittnotify
ittnotify_OBJECTS = \
"CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o" \
"CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o"

# External object files for target ittnotify
ittnotify_EXTERNAL_OBJECTS =

3rdparty/lib/arm64-v8a/libittnotify.a: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/ittnotify_static.c.o
3rdparty/lib/arm64-v8a/libittnotify.a: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/src/ittnotify/jitprofiling.c.o
3rdparty/lib/arm64-v8a/libittnotify.a: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/build.make
3rdparty/lib/arm64-v8a/libittnotify.a: 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C static library ../lib/arm64-v8a/libittnotify.a"
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && $(CMAKE_COMMAND) -P CMakeFiles/ittnotify.dir/cmake_clean_target.cmake
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ittnotify.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rdparty/ittnotify/CMakeFiles/ittnotify.dir/build: 3rdparty/lib/arm64-v8a/libittnotify.a
.PHONY : 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/build

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/clean:
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify && $(CMAKE_COMMAND) -P CMakeFiles/ittnotify.dir/cmake_clean.cmake
.PHONY : 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/clean

3rdparty/ittnotify/CMakeFiles/ittnotify.dir/depend:
	cd /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/3rdparty/ittnotify /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/3rdparty/ittnotify/CMakeFiles/ittnotify.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rdparty/ittnotify/CMakeFiles/ittnotify.dir/depend

