#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

const uint32_t HEIGHT = 600;
const uint32_t WIDTH = 800;

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void initVulkan() {
        
    }

    void initWindow() {
        glfwInit();

        // Turn off OpenGL
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Temporarily disable window resizing (handled later)
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // Make the window
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
