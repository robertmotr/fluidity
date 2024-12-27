#include "render.h"
#include "physics.h"
#include "globals.h"

void errorCallback(int error, const char* description) {
    std::cout << "Error: " << description << std::endl;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

GLFWwindow* glfwSetup(uint32_t width, uint32_t height, const char* title, 
                      GLFWmonitor *monitor, GLFWwindow *share) {
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return nullptr;
    }

    glViewport(0, 0, 800, 800);

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    return window;
}

ImGuiIO& setupImGui(GLFWwindow *window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130"); 
    return io;
}

void showContent(GLFWwindow *window) {
    ImGui::Begin("Fluid Simulation");
    ImGui::Text("test");
    ImGui::End();
}