cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(VdP)

#before using cmake make sure to install libtorch
#then follow (from this directory in cmd):
#md build
#cd build
#cmake -DCMAKE_PREFIX_PATH=path\to\libtorch ..

#build from cmd: cmake --build . --config Release
#BUT recommended to use Visual Studio for building
#Open the project in VS and set the DEM project as Start Project


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#include_directories(C:/Users/plamb/Dev/odeint/boost_1_75_0)
#include_directories(C:/Users/plamb/Dev/DeepLearningBubble/eigen-3.3.9/Eigen)

add_executable(VdP VdP.cpp)
target_link_libraries(VdP "${TORCH_LIBRARIES}")
#target_link_libraries (VdP Eigen3::Eigen)
set_property(TARGET VdP PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET VdP
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:VdP>)
endif (MSVC)