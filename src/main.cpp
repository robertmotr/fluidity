#include "render.h"
#include "physics.h"

int main(int argc, const char **argv) {

    if(argv != nullptr) {
        // todo: add more command line options
        for(int i = 0; i < argc; i++) {
            if(strcmp(argv[i], "--use-gpu") == 0) {
                PhysicsGlobals::useGpu = true;
                RenderConfig::useGpu = true;
            }
        }
    }

    GLFWwindow* window = glfwSetup(800, 800, "Fluid Simulation", nullptr, nullptr);
    ImGuiIO& io = setupImGui(window);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        showContent(window);
        computeSimulationTick();

        ImGui::Render();
        glViewport(0, 0, 1920, 1080);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    
    glfwTerminate();
    return 0;
}
