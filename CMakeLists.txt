cmake_minimum_required(VERSION 3.0)

project(deeplabv3plus)
#enable_language(CUDA)
find_package(CUDA REQUIRED)


SET(CMAKE_BUILD_TYPE "Debug") 
set(CMAKE_CXX_FLAGS   "-std=c++11")             # c++11
set(CMAKE_CXX_FLAGS   "-g")                     # 调试信息
set(CMAKE_CXX_FLAGS   "-Wall")                  # 开启所有警告

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# 设置可执行二进制文件的目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 设置存放编译出来的库文件的目录
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# 并把该目录设置为链接目录
link_directories(${PROJECT_SOURCE_DIR}/lib)


# 设定头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
#find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
include_directories($CUDA_INCLUDE_DIRS)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/ylc/GMP/trt7/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/include)
link_directories(/home/ylc/GMP/trt7/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_VERSION})


#eigen
find_package(Eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})
message(STATUS ${Eigen3_VERSION})

#libcurl
find_package(CURL)
if(NOT CURL_FOUND)
     message(FATAL_ERROE "curl not found!")
else()
     message(STATUS ${CURL_VERSION_STRING})
endif (NOT CURL_FOUND)
include_directories(${CURL_INCLUDE_DIRS})
link_directories(${CURL_LIBRARIES})

#openmp
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
endif()

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/argmax.cu)
target_link_libraries(myplugins nvinfer cudart)

cuda_add_executable(inference  ${PROJECT_SOURCE_DIR}/inference.cpp 
${PROJECT_SOURCE_DIR}/calibrator.cpp 
${PROJECT_SOURCE_DIR}/common.cpp
${PROJECT_SOURCE_DIR}/deeplabv3plus.cpp 
${PROJECT_SOURCE_DIR}/preprocess.cu
${PROJECT_SOURCE_DIR}/utils.cpp 
${PROJECT_SOURCE_DIR}/argmax.cu)
target_link_libraries(inference nvinfer cudart ${OpenCV_LIBS} )
target_link_libraries(inference curl)


cuda_add_executable(serialize  ${PROJECT_SOURCE_DIR}/serialize.cpp 
${PROJECT_SOURCE_DIR}/calibrator.cpp 
${PROJECT_SOURCE_DIR}/common.cpp 
${PROJECT_SOURCE_DIR}/deeplabv3plus.cpp 
${PROJECT_SOURCE_DIR}/preprocess.cu 
${PROJECT_SOURCE_DIR}/utils.cpp 
${PROJECT_SOURCE_DIR}/argmax.cu)
target_link_libraries(serialize  nvinfer cudart ${OpenCV_LIBS} curl  OpenMP::OpenMP_CXX)

add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/common.cpp ${PROJECT_SOURCE_DIR}/deeplabv3plus.cpp ${PROJECT_SOURCE_DIR}/utils.cpp ${PROJECT_SOURCE_DIR}/utils.cpp ${PROJECT_SOURCE_DIR}/preprocess.cu ${PROJECT_SOURCE_DIR}/argmax.cu)
target_link_libraries(${PROJECT_NAME}  curl myplugins ${OpenCV_LIBS})