#pragma once

typedef unsigned int uint32;

// cuda related
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// std libs
#include <iostream>
#include <cstddef>
#include <stdio.h>

// imgui/graphics/rendering
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "glad/glad.h"
#include <GLFW/glfw3.h>