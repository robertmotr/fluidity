cmake_minimum_required(VERSION 3.20)

project(FluidSimulator VERSION 1.0 LANGUAGES C CXX CUDA)

# set this to off to get smaller binary, optimization etc (release)
# keep this on to make debugging as easy as possible, alongside more compiler checks such as address sanitizer
set(DEBUG ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

find_package(OpenGL REQUIRED)
if (NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found")
endif()

find_package(glfw3 REQUIRED)
if (NOT glfw3_FOUND)
    message(FATAL_ERROR "GLFW not found")
endif()

set_source_files_properties(external/glad/src/glad.c PROPERTIES LANGUAGE C)
add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu)
file(GLOB_RECURSE HEADERS includes/*.h)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE 
    includes
    external/imgui
    external/imgui/backends
    external/glfw/include
    external/glfw
    external/glad/include
    external/glad/src
    external/glad
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
)
target_include_directories(imgui PUBLIC external/imgui external/imgui/backends)

target_precompile_headers(${PROJECT_NAME} PRIVATE includes/pch.h)

target_link_libraries(${PROJECT_NAME} PUBLIC
    imgui
    glfw
    OpenGL::GL
    glad
    cuda
)

if(WIN32)
    if(DEBUG)
        target_compile_options(${PROJECT_NAME} PRIVATE /Zi /JMC /Od) # Debug options
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE /O2 /Ot) # Release options
    endif()
elseif(UNIX AND NOT APPLE) 
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(DEBUG)
            target_compile_options(${PROJECT_NAME} PRIVATE 
                -Wall -Wextra -Wpedantic -O0 -g -fno-inline) # Debug options
            target_link_libraries(${PROJECT_NAME} PRIVATE)
        else()
            target_compile_options(${PROJECT_NAME} PRIVATE 
                -Wall -Wextra -Wpedantic -O3) # Release options
        endif()
    else()
        message(FATAL_ERROR "Compiler not supported.")
    endif()
else()
    message(FATAL_ERROR "Unknown platform, not supported. Not Linux or Windows.")
endif()
