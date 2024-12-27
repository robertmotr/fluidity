#pragma once

#include "pch.h"

void errorCallback(int error, const char* description);

void framebufferSizeCallback(GLFWwindow* window, int width, int height);

GLFWwindow* glfwSetup(uint32_t width, uint32_t height, const char* title, 
                      GLFWmonitor *monitor, GLFWwindow *share);

ImGuiIO& setupImGui(GLFWwindow *window);

void showContent(GLFWwindow *window);