# Install script for directory: /dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/dcs/20/u2030590/.buildozer/android/platform/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sdk/etc/haarcascades" TYPE FILE FILES
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_eye.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_frontalcatface.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_frontalcatface_extended.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_fullbody.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_lefteye_2splits.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_licence_plate_rus_16stages.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_lowerbody.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_profileface.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_righteye_2splits.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_russian_plate_number.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_smile.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/haarcascades/haarcascade_upperbody.xml"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/sdk/etc/lbpcascades" TYPE FILE FILES
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/lbpcascades/lbpcascade_frontalcatface.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/lbpcascades/lbpcascade_frontalface.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/lbpcascades/lbpcascade_profileface.xml"
    "/dcs/20/u2030590/CS310-App/App/.buildozer/android/platform/build-armeabi-v7a_arm64-v8a/build/other_builds/opencv/arm64-v8a__ndk_target_21/opencv/data/lbpcascades/lbpcascade_silverware.xml"
    )
endif()

