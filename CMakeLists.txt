cmake_minimum_required(VERSION 3.20)

project(FluidSimulator VERSION 1.0 LANGUAGES CXX CUDA)

# set this to off to get smaller binary, optimization etc (release)
# keep this on to make debugging as easy as possible, alongside more compiler checks such as address sanitizer
set(DEBUG ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu)
file(GLOB_RECURSE HEADERS includes/*.h)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE 
    includes
    external/imgui
    external/imgui/backends
    external/glfw/include
    external/glfw
)

find_package(OpenGL REQUIRED)
if (NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found!")
endif()

find_package(glfw3 REQUIRED)
if (NOT glfw3_FOUND)
    message(FATAL_ERROR "GLFW not found!")
endif()

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
)

target_include_directories(imgui PRIVATE external/imgui external/imgui/backends)
target_link_libraries(imgui PRIVATE glfw OpenGL::GL)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    imgui
    glfw
    OpenGL::GL
#    CUDA::cudart
)

if (WIN32)
    if(DEBUG == ON)
        target_compile_options(${PROJECT_NAME} PRIVATE /O2 /Ot)
    else 
        target_compile_options(${PROJECT_NAME} PRIVATE /Zi /JMC /Od)

else if(UNIX AND NOT APPLE) 
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(DEBUG == ON)
            target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -O0 -g -fsanitize=address,undefined -fno-inline)
        else 
            target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -O3)
    else    
        message("Compiler not supported.")
else 
    message("Unknown platform, not supported. Not Linux or Windows.")
endif()
