cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DEM)
#tested on Windows 10
#before generation use command
#cmake -DCMAKE_PREFIX_PATH=\path\to\libtorch ..

#build from cmd: 
#cmake --build . --config Release
#Recommended to use Visual Studio for building
#Open the project in VS and set the DEM project as Start Project

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(DEM DEM.cpp)
target_link_libraries(DEM "${TORCH_LIBRARIES}")
set_property(TARGET DEM PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET DEM
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:DEM>)
endif (MSVC)