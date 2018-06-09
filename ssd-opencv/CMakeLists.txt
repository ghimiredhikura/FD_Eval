cmake_minimum_required(VERSION 3.1)
project(face_detection)

set (CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)

add_executable(face_detection resnet_ssd_face.cpp)
target_link_libraries(face_detection ${OpenCV_LIBS} )
