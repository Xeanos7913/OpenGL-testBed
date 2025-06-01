#pragma once

#ifndef CALCIUM_ENGINE

#define CALCIUM_ENGINE
#define GLM_ENABLE_EXPERIMENTAL

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtx/quaternion.hpp>
#include <gtc/type_ptr.hpp>
#include <gtx/string_cast.hpp>
#include <gtx/intersect.hpp>
#include <functional>
#include <unordered_map>
#include <string>
#include <typeindex>
#include <tuple>
#include <any>
#include <array>
#include <unordered_set>
#include <algorithm>

#include "stb_image.h"
#include "tiny_obj_loader.h"

struct Vertex {
    glm::vec3 position;
    int positionIndex = -1;
    glm::vec2 texCoords;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;
    int boneIndex = -1;

    bool operator==(const Vertex& other) const {
        return positionIndex == other.positionIndex && texCoords == other.texCoords && normal == other.normal;
    }
};

// Custom hash function for Vertex
namespace std {
    template <>
    struct hash<Vertex> {
        size_t operator()(const Vertex& vertex) const {
            size_t seed = 0;
            hash<float> hasher;

            // Hash position
            seed ^= hasher(vertex.positionIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            seed ^= hasher(vertex.position.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.position.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.position.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash texCoords
            seed ^= hasher(vertex.texCoords.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.texCoords.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash normal
            seed ^= hasher(vertex.normal.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.normal.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.normal.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash tanget
            seed ^= hasher(vertex.tangent.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.tangent.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.tangent.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash bitangent
            seed ^= hasher(vertex.bitangent.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.bitangent.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.bitangent.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

			// Hash boneIndex
			seed ^= hasher(vertex.boneIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}

struct Edge {
    Vertex* v1;
    Vertex* v2;

    Edge(Vertex* a, Vertex* b) {
        if (a < b) { v1 = a; v2 = b; }
        else { v1 = b; v2 = a; }
    }

    bool operator==(const Edge& other) const {
        return v1 == other.v1 && v2 == other.v2;
    }
};

struct Triangle {
    Vertex* v1;
    Vertex* v2;
    Vertex* v3;

    Triangle(Vertex* a, Vertex* b, Vertex* c) : v1(a), v2(b), v3(c) {}
};

// Custom hash function for Edge
namespace std {
    template <>
    struct hash<Edge> {
        size_t operator()(const Edge& edge) const {
            size_t seed = 0;
            hash<Vertex*> hasher;

            // Combine the hashes of both vertices (order-independent)
            seed ^= hasher(edge.v1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(edge.v2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}

// Custom hash function for Triangle
namespace std {
    template <>
    struct hash<Triangle> {
        size_t operator()(const Triangle& triangle) const {
            size_t seed = 0;
            hash<Vertex*> hasher;

            // Sort the pointers first (ensures order-independent hashing)
            std::array<Vertex*, 3> sortedVertices = { triangle.v1, triangle.v2, triangle.v3 };
            std::sort(sortedVertices.begin(), sortedVertices.end());

            // Combine hashes of all three vertices
            for (auto* v : sortedVertices) {
                seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            return seed;
        }
    };
}

namespace Calcium {
// Define Edge comparison function
struct EdgeEqual {
    bool operator()(const Edge& e1, const Edge& e2) const {
        return (e1.v1 == e2.v1 && e1.v2 == e2.v2);
    }
};

// Define Triangle comparison function
struct TriangleEqual {
    bool operator()(const Triangle& t1, const Triangle& t2) const {
        std::array<Vertex*, 3> sortedT1 = { t1.v1, t1.v2, t1.v3 };
        std::array<Vertex*, 3> sortedT2 = { t2.v1, t2.v2, t2.v3 };

        std::sort(sortedT1.begin(), sortedT1.end());
        std::sort(sortedT2.begin(), sortedT2.end());

        return sortedT1 == sortedT2;
    }
};
}
namespace Calcium {

// Devilspawn. Yeah, idk what I was thinking while making this...
struct CustomFunction {
    std::function<void()> function;
    std::vector<std::any> args;

    template<typename Function, typename... Args>
    void setFunction(Function& func, Args&&... initialArgs) {
        function = [this, f = std::forward<Function>(func), argsTuple = std::make_tuple(std::forward<Args>(initialArgs)...)]() mutable {
            callFunction(f, argsTuple, std::index_sequence_for<Args...>{});
        };
        args = { std::forward<Args>(initialArgs)... };
    }

    template<std::size_t Index, typename Arg>
    void setArg(Arg&& newArg) {
        if (Index < args.size()) {
            args[Index] = std::forward<Arg>(newArg);
        }
        else {
            throw std::out_of_range("Index out of bounds");
        }
    }

    void operator()() {
        if (function) {
            function();
        }
    }

private:
    template<typename Function, typename Tuple, std::size_t... Indices>
    void callFunction(Function& f, Tuple& tuple, std::index_sequence<Indices...>) {
        f(std::any_cast<std::tuple_element_t<Indices, Tuple>>(args[Indices])...);
    }
};

class Entity;

class Component {
public:
	virtual ~Component() {}
    Entity* owner;
};

struct Transform {

    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;

    glm::mat4 model;

    Transform(glm::vec3 position, glm::quat rotation, glm::vec3 scale) : position(position), rotation(rotation), scale(scale){}

    void updateGlobalTransform() {
        glm::mat4 localTransform = glm::mat4(1.0f);
        localTransform = glm::translate(localTransform, position);
        localTransform *= glm::toMat4(rotation);
        localTransform = glm::scale(localTransform, scale);
        model = localTransform;
    }

    glm::vec3 getPosition() const {
        return position;
    }
	glm::quat getRotation() const {
		return rotation;
	}
    glm::vec3 getScale() const {
        return scale;
    }

	void setPosition(glm::vec3 position) {
		this->position = position;
	}
	void setRotation(glm::quat rotation) {
		this->rotation = rotation;
	}
	void setScale(glm::vec3 scale) {
		this->scale = scale;
	}

	void translate(glm::vec3 translation) {
		position += translation;
	}
	void rotate(glm::quat rotation) {
		this->rotation = rotation * this->rotation;
	}
	void scaleBy(glm::vec3 scale) {
		this->scale *= scale;
	}
};

typedef struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

class Entity {
public:
        
    bool useBuiltInTransform = true;
    Transform transform;

    Entity(glm::vec3 position, glm::quat rotation, glm::vec3 scale, bool useBuiltInTransform) : transform(position, rotation, scale), useBuiltInTransform(useBuiltInTransform){

    }

    void update() {
        if (useBuiltInTransform) {
            transform.updateGlobalTransform();
        }
    }

    template <typename T> 
    void addComponent(std::shared_ptr<T> component) {
        if (!std::is_base_of<Component, T>::value) {
            throw std::runtime_error("The component must inherit from the Component class.");
        }
        else {
            auto comp = std::static_pointer_cast<Component>(component);
            comp->owner = this;
            components[std::type_index(typeid(T))] = component;
        }
    }

    template <typename T>
    bool hasComponent() {
        return components.find(std::type_index(typeid(T))) != components.end();
    }

	template <typename T>
    std::shared_ptr<T> getComponent() {
        auto it = components.find(std::type_index(typeid(T)));
        if (it != components.end()) {
            return std::static_pointer_cast<T>(it->second);
        }
        return nullptr;
    }

	std::unordered_map<std::type_index, std::shared_ptr<void>> components;
};

class Shader {
public:
    unsigned int ID;

    Shader() {};

    // Constructor
    Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr) {

        std::string vertexCode;
        std::string fragmentCode;
        std::string geometryCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;
        std::ifstream gShaderFile;

        // ensure ifstream objects can throw exceptions
        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        if (geometryPath != nullptr)
            gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        try {
            // open the vertex and fragment shader files
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderstream;

            // read the file's buffer contents into the streams
            vShaderStream << vShaderFile.rdbuf();
            fShaderstream << fShaderFile.rdbuf();

            // close the file handlers
            vShaderFile.close();
            fShaderFile.close();

            // convert the streams into strings
            vertexCode = vShaderStream.str();
            fragmentCode = fShaderstream.str();

            // if geometry shader path is present, read it
            if (geometryPath != nullptr) {
                gShaderFile.open(geometryPath);
                std::stringstream gShaderStream;
                gShaderStream << gShaderFile.rdbuf();
                gShaderFile.close();
                geometryCode = gShaderStream.str();
            }
        }
        catch (std::ifstream::failure e) {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ" << std::endl;
        }

        const char* vShaderCode = vertexCode.c_str();
        const char* fShaderCode = fragmentCode.c_str();
        const char* gShaderCode = geometryPath != nullptr ? geometryCode.c_str() : nullptr;

        unsigned int vertex, fragment, geometry;
        int success;
        char infoLog[512];

        // Vertex Shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        // Print compile errors if any
        glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertex, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // Fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        // Print compile errors if any
        glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragment, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // Geometry Shader (if provided)
        if (geometryPath != nullptr) {
            geometry = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometry, 1, &gShaderCode, NULL);
            glCompileShader(geometry);
            // Print compile errors if any
            glGetShaderiv(geometry, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(geometry, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n" << infoLog << std::endl;
            }
        }

        // Shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        if (geometryPath != nullptr) {
            glAttachShader(ID, geometry);
        }
        glLinkProgram(ID);
        // Print linking errors if any
        glGetProgramiv(ID, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(ID, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }

        // Delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        if (geometryPath != nullptr) {
            glDeleteShader(geometry);
        }
    }

    Shader(const char* vertexCode, const char* fragmentCode, bool hardCode) {
        unsigned int vertex, fragment;
        int success;
        char infoLog[512];

        // Vertex Shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vertexCode, NULL);
        glCompileShader(vertex);
        // Print compile errors if any
        glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertex, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // Fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fragmentCode, NULL);
        glCompileShader(fragment);
        // Print compile errors if any
        glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragment, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // Shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        // Print linking errors if any
        glGetProgramiv(ID, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(ID, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }

        // Delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

    // Use/activate the shader program
    void use() {
        glUseProgram(ID);
    }

    // Utility uniform functions
    void setBool(const std::string& name, bool value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }

    void setInt(const std::string& name, int value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }

    void setFloat(const std::string& name, float value) const {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }

    void setMat4(const std::string& name, glm::mat4 matrix) const {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(matrix));
    }

    void setMat3(const std::string& name, glm::mat3 matrix) const {
        glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(matrix));
    }

    void setVec3(const std::string& name, glm::vec3 vec3) const {
        glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(vec3));
    }

    void setVec2(const std::string& name, glm::vec2 vec2) const {
        glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(vec2));
    }
};

class Texture2D {
public:
    std::string shaderTextureSamplerName;
    GLuint textureID;

    void Activate(int shaderProgram, int textureUnit) {
        glActiveTexture(textureUnit);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, shaderTextureSamplerName.c_str()), textureUnit);
    }

    Texture2D() {};

    Texture2D(
        const std::string _shaderTextureName,
        const std::string path,
        const bool generateMipmaps,
        const int force_channels)
        : shaderTextureSamplerName(_shaderTextureName) {
        // Load image from disk.
        auto* textureBuffer = stbi_load(path.c_str(), &width, &height, &channels, force_channels);
        if (textureBuffer == nullptr) {
            std::cerr << "ERROR: Failed to load image with path " << path << " into a texture." << std::endl;
            return;
        }
        else {
            std::cout << "- SOIL: Successfully loaded image with path '" << path << "'." << std::endl;
        }

        // Generate texture on GPU.
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        // Parameter options.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // Set texture filtering options.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, generateMipmaps ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Upload texture buffer.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureBuffer);

        // Mip maps.
        if (generateMipmaps) {
            glGenerateMipmap(GL_TEXTURE_2D);
        }

        // Clean up.
        stbi_image_free(textureBuffer);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void loadTexture(const std::string& path) {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        // Texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        int width, height, nrChannels;
        stbi_set_flip_vertically_on_load(false);
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            GLenum format = GL_RGB;
            if (nrChannels == 1)
                format = GL_RED;
            else if (nrChannels == 3)
                format = GL_RGB;
            else if (nrChannels == 4)
                format = GL_RGBA;

            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else {
            std::cerr << "Failed to load texture at path: " << path << "\n";
            textureID = 0;
        }
        stbi_image_free(data);
    }

    void fillTexture(glm::vec4 color) {
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_FLOAT, glm::value_ptr(color));
		glBindTexture(GL_TEXTURE_2D, 0);
    }

    ~Texture2D() {
        glDeleteTextures(1, &textureID);
    };

private:
    int width, height, channels;
};

class Texture3D {
public:
    unsigned char* textureBuffer = nullptr;
    GLuint textureID;

    void Activate(const int shaderProgram, const std::string glSamplerName, const int textureUnit) {
        glActiveTexture(GL_TEXTURE0 + textureUnit);
        glBindTexture(GL_TEXTURE_3D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, glSamplerName.c_str()), textureUnit);
    };

    void Clear(GLfloat clearColor[4]) {
        GLint previousBoundTextureID;
        glGetIntegerv(GL_TEXTURE_BINDING_3D, &previousBoundTextureID);
        glBindTexture(GL_TEXTURE_3D, textureID);
        glClearTexImage(textureID, 0, GL_RGBA, GL_FLOAT, &clearColor);
        glBindTexture(GL_TEXTURE_3D, previousBoundTextureID);
    }

    Texture3D(const std::vector<GLfloat>& textureBuffer, const int _width, const int _height, const int _depth, const bool generateMipmaps) :
        width(_width), height(_height), depth(_depth), clearData(4 * _width * _height * _depth, 0.0f) {
        // Generate texture on GPU.
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_3D, textureID);

        // Parameter options.
        const auto wrap = GL_CLAMP_TO_BORDER;
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap);

        const auto filter = GL_LINEAR_MIPMAP_LINEAR;
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Upload texture buffer.
        const int levels = 7;
        glTexStorage3D(GL_TEXTURE_3D, levels, GL_RGBA8, width, height, depth);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, width, height, depth, 0, GL_RGBA, GL_FLOAT, &textureBuffer[0]);
        if (generateMipmaps) glGenerateMipmap(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_3D, 0);
    };

private:
    int width, height, depth;
    std::vector<GLfloat> clearData;
};

class Framebuffer2D {
public:
    Framebuffer2D(int width, int height, int numColorAttachments, bool useDepthStencil)
        : m_FBO(0), m_DepthStencilAttachment(0), m_Width(width), m_Height(height),
        m_NumColorAttachments(numColorAttachments), m_UseDepthStencil(useDepthStencil) {
        CreateFramebuffer();
    };

    ~Framebuffer2D() {
        Cleanup();
    };

    void Bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    }

    static void Unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Resize(int newWidth, int newHeight) {
        if (newWidth == m_Width && newHeight == m_Height) return;

        m_Width = newWidth;
        m_Height = newHeight;

        Cleanup();
        CreateFramebuffer();
    }

    GLuint GetColorTexture(unsigned int index) const {
        if (index >= m_ColorAttachments.size()) {
            std::cerr << "ERROR: Invalid color attachment index!" << std::endl;
            return 0;
        }
        return m_ColorAttachments[index];
    }

    GLuint GetDepthStencilTexture() const {
        return m_UseDepthStencil ? m_DepthStencilAttachment : 0;
    }

    bool IsComplete() const {
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
        bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return complete;
    }

private:
    void CreateFramebuffer() {
        glCreateFramebuffers(1, &m_FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

        m_ColorAttachments.resize(m_NumColorAttachments);

        // Create color attachments
        glCreateTextures(GL_TEXTURE_2D, m_NumColorAttachments, m_ColorAttachments.data());
        for (int i = 0; i < m_NumColorAttachments; ++i) {
            glBindTexture(GL_TEXTURE_2D, m_ColorAttachments[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, m_ColorAttachments[i], 0);
        }

        // Create depth-stencil attachment if needed
        if (m_UseDepthStencil) {
            glCreateTextures(GL_TEXTURE_2D, 1, &m_DepthStencilAttachment);
            glBindTexture(GL_TEXTURE_2D, m_DepthStencilAttachment);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, m_Width, m_Height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_DepthStencilAttachment, 0);
        }

        // Define draw buffers
        std::vector<GLenum> attachments(m_NumColorAttachments);
        for (int i = 0; i < m_NumColorAttachments; ++i) {
            attachments[i] = GL_COLOR_ATTACHMENT0 + i;
        }
        glDrawBuffers(m_NumColorAttachments, attachments.data());

        if (!IsComplete()) {
            std::cerr << "ERROR: Framebuffer is not complete!" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Cleanup() {
        glDeleteFramebuffers(1, &m_FBO);
        glDeleteTextures(m_NumColorAttachments, m_ColorAttachments.data());

        if (m_UseDepthStencil) {
            glDeleteTextures(1, &m_DepthStencilAttachment);
        }
    }

    GLuint m_FBO;
    std::vector<GLuint> m_ColorAttachments;
    GLuint m_DepthStencilAttachment;

    int m_Width, m_Height;
    int m_NumColorAttachments;
    bool m_UseDepthStencil;
};

class Framebuffer3D {
public:
    Framebuffer3D(int width, int height, int depth, int numColorAttachments, bool useDepthStencil)
        : m_FBO(0), m_DepthStencilAttachment(0), m_Width(width), m_Height(height), m_Depth(depth),
        m_NumColorAttachments(numColorAttachments), m_UseDepthStencil(useDepthStencil) {
        CreateFramebuffer();
    };

    ~Framebuffer3D() {
        Cleanup();
    }

    void Bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    }

    static void Unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Resize(int newWidth, int newHeight, int newDepth) {
        if (newWidth == m_Width && newHeight == m_Height && newDepth == m_Depth) return;

        m_Width = newWidth;
        m_Height = newHeight;
        m_Depth = newDepth;

        Cleanup();
        CreateFramebuffer();
    }

    GLuint GetColorTexture(unsigned int index) const {
        if (index >= m_ColorAttachments.size()) {
            std::cerr << "ERROR: Invalid color attachment index!" << std::endl;
            return 0;
        }
        return m_ColorAttachments[index];
    }

    GLuint GetDepthStencilTexture() const {
        return m_UseDepthStencil ? m_DepthStencilAttachment : 0;
    }

    bool IsComplete() const {
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
        bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return complete;
    }

private:
    void CreateFramebuffer() {
        glCreateFramebuffers(1, &m_FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

        m_ColorAttachments.resize(m_NumColorAttachments);

        // Create 3D color attachments
        glCreateTextures(GL_TEXTURE_3D, m_NumColorAttachments, m_ColorAttachments.data());
        for (int i = 0; i < m_NumColorAttachments; ++i) {
            glBindTexture(GL_TEXTURE_3D, m_ColorAttachments[i]);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, m_Width, m_Height, m_Depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_3D, m_ColorAttachments[i], 0, 0);
        }

        // Create depth-stencil attachment (as a 2D texture, since OpenGL does not support 3D depth textures for FBOs)
        if (m_UseDepthStencil) {
            glCreateTextures(GL_TEXTURE_2D, 1, &m_DepthStencilAttachment);
            glBindTexture(GL_TEXTURE_2D, m_DepthStencilAttachment);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, m_Width, m_Height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_DepthStencilAttachment, 0);
        }

        // Define draw buffers
        std::vector<GLenum> attachments(m_NumColorAttachments);
        for (int i = 0; i < m_NumColorAttachments; ++i) {
            attachments[i] = GL_COLOR_ATTACHMENT0 + i;
        }
        glDrawBuffers(m_NumColorAttachments, attachments.data());

        if (!IsComplete()) {
            std::cerr << "ERROR: Framebuffer3D is not complete!" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Cleanup() {
        glDeleteFramebuffers(1, &m_FBO);
        glDeleteTextures(m_NumColorAttachments, m_ColorAttachments.data());

        if (m_UseDepthStencil) {
            glDeleteTextures(1, &m_DepthStencilAttachment);
        }
    }

    GLuint m_FBO;
    std::vector<GLuint> m_ColorAttachments;
    GLuint m_DepthStencilAttachment;

    int m_Width, m_Height, m_Depth;
    int m_NumColorAttachments;
    bool m_UseDepthStencil;
};

class Material {
public:
    Texture2D texture;
	Texture2D normalMap;
	Texture2D specularMap;
	Texture2D displacementMap;
	Texture2D roughnessMap;
	Texture2D aoMap;

    float tilingFactor = 1.0f;
    Material(bool useTexture = true) : useTexture(useTexture) {};
    bool useTexture = false;
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
    float diffuseReflectivity;
    float specularReflectivity;
    float emissivity;
    float transparency;
    float refractiveIndex;
    float specularDiffusion;
    
    Material(glm::vec3 diffuseColor,
        glm::vec3 specularColor,
        float diffuseReflectivity,
        float specularReflectivity,
        float emissivity,
        float transparency, float refractiveIndex, float specularDiffusion)
        : diffuseColor(diffuseColor),
          specularColor(specularColor),
          diffuseReflectivity(diffuseReflectivity),
          specularReflectivity(specularReflectivity),
          emissivity(emissivity),
        transparency(transparency), refractiveIndex(refractiveIndex), specularDiffusion(specularDiffusion){
        useTexture = false;
    }
    void loadTexture(const std::string& texturePath) {
		texture.loadTexture(texturePath);
    }

	void loadNormalMap(const std::string& normalMapPath) {
		normalMap.loadTexture(normalMapPath);
	}

	void loadSpecularMap(const std::string& specularMapPath) {
		specularMap.loadTexture(specularMapPath);
	}

	void loadDisplacementMap(const std::string& displacementMapPath) {
		displacementMap.loadTexture(displacementMapPath);
	}

	void loadRoughnessMap(const std::string& roughnessMapPath) {
		roughnessMap.loadTexture(roughnessMapPath);
	}

	void loadAOMap(const std::string& aoMapPath) {
		aoMap.loadTexture(aoMapPath);
	}

    void bindAllTextures(unsigned int shaderProgram) {
		texture.Activate(shaderProgram, GL_TEXTURE0);
		normalMap.Activate(shaderProgram, GL_TEXTURE1);
		specularMap.Activate(shaderProgram, GL_TEXTURE2);
		displacementMap.Activate(shaderProgram, GL_TEXTURE3);
		roughnessMap.Activate(shaderProgram, GL_TEXTURE4);
		aoMap.Activate(shaderProgram, GL_TEXTURE5);
    }

    void setUniforms(Shader& shader, std::string materialName) {
        shader.setVec3(materialName + ".diffuseColor", diffuseColor);
        shader.setVec3(materialName + ".specularColor", specularColor);
        shader.setFloat(materialName + ".diffuseReflectivity", diffuseReflectivity);
        shader.setFloat(materialName + ".specularReflectivity", specularReflectivity);
        shader.setFloat(materialName + ".emissivity", emissivity);
        shader.setFloat(materialName + ".transparency", transparency);
        shader.setFloat(materialName + ".refractiveIndex", refractiveIndex);
        shader.setFloat(materialName + ".specularDiffusion", specularDiffusion);
    }
};

class Mesh : public Component{

public:
    std::string objFilePath;
    std::string texturePath;
	std::string normalMapPath;
	std::string specularMapPath;
	std::string displacementMapPath;
	std::string roughnessMapPath;
	std::string aoMapPath;

    std::vector<Vertex> vertices;  // Stores vertex data
    std::vector<unsigned int> indices; // Stores indices for element drawing
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec3> biTangents;
    std::vector<Triangle> triangles;
    std::vector<glm::vec3> positions;
    unsigned int indexCount; // Number of indices

    unsigned int VAO, VBO, EBO;
    Material material{};

    CustomFunction loadMeshFunction;
	CustomFunction setupMeshFunction;
    CustomFunction loadTextureFunction;
    CustomFunction bindMeshFunction;
    CustomFunction drawFunction;

    bool useBuiltInMesh = true;
    bool debugMode = false;

    Mesh(const std::string& objFilePath, const std::string& texturePath, const std::string& normalMapPath, const std::string& specularMapPath, const std::string& displacementMapPath, const std::string& roughnessMapPath, const std::string& aoMapPath, bool useBuiltInMesh = true)
        : objFilePath(objFilePath), texturePath(texturePath), normalMapPath(normalMapPath), specularMapPath(specularMapPath), displacementMapPath(displacementMapPath), roughnessMapPath(roughnessMapPath), aoMapPath(aoMapPath), useBuiltInMesh(useBuiltInMesh) {
		
        if (useBuiltInMesh) {
            loadModel();
		    setupMesh();
		    loadTexture();
        }
        else {
            loadModel();
            setupMeshFunction();
            loadTextureFunction();
        }
	}

    Mesh(const std::string& objFilePath, glm::vec3 diffuseColor,
        glm::vec3 specularColor,
        float diffuseReflectivity,
        float specularReflectivity,
        float emissivity,
        float transparency, float refractiveIndex, float specularDiffusion) : objFilePath(objFilePath), material(diffuseColor, specularColor, diffuseReflectivity, specularReflectivity, emissivity, transparency, refractiveIndex, specularDiffusion){
        loadModel();
        setupMesh();
    }

    Mesh() {}

	std::vector<Vertex>& GetVertices() {
		return vertices;
	}

	std::vector<unsigned int>& GetIndices() {
		return indices;
	}

    // Load model using TinyObjLoader
    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFilePath.c_str())) {
            std::cerr << "Failed to load OBJ file: " << objFilePath << "\n";
            if (!warn.empty()) std::cerr << "WARN: " << warn << "\n";
            if (!err.empty()) std::cerr << "ERR: " << err << "\n";
            std::cout << objFilePath << "\n";
            return;
        }

        std::unordered_map<Vertex, unsigned int> uniqueVertices{};
        std::vector<glm::vec3> uniquePositions;

        for (const auto& shape : shapes) {
            for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
                // Indices for the triangle
                tinyobj::index_t idx0 = shape.mesh.indices[i + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[i + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[i + 2];

                Vertex vertex0{}, vertex1{}, vertex2{};
                glm::vec3 position1{}, position2{}, position3{};
                // Positions
                position1 = {
                    attrib.vertices[3 * idx0.vertex_index + 0],
                    attrib.vertices[3 * idx0.vertex_index + 1],
                    attrib.vertices[3 * idx0.vertex_index + 2]
                };
                position2 = {
                    attrib.vertices[3 * idx1.vertex_index + 0],
                    attrib.vertices[3 * idx1.vertex_index + 1],
                    attrib.vertices[3 * idx1.vertex_index + 2]
                };
                position3 = {
                    attrib.vertices[3 * idx2.vertex_index + 0],
                    attrib.vertices[3 * idx2.vertex_index + 1],
                    attrib.vertices[3 * idx2.vertex_index + 2]
                };

                auto addPosition1 = [&uniquePositions, &position1] {
                    auto it1 = std::find(uniquePositions.begin(), uniquePositions.end(), position1);
                    if (it1 == uniquePositions.end()) {
                        uniquePositions.push_back(position1);
                    }
                };

                auto addPosition2 = [&uniquePositions,&position2] {
                    auto it2 = std::find(uniquePositions.begin(), uniquePositions.end(), position2);
                    if (it2 == uniquePositions.end()) {
                        uniquePositions.push_back(position2);
                    }
                };

                auto addPosition3 = [&uniquePositions, &position3] {
                    auto it3 = std::find(uniquePositions.begin(), uniquePositions.end(), position3);
                    if (it3 == uniquePositions.end()) {
                        uniquePositions.push_back(position3);
                    }
                };

                addPosition1();
                addPosition2();
                addPosition3();

                vertex0.position = {
                    attrib.vertices[3 * idx0.vertex_index + 0],
                    attrib.vertices[3 * idx0.vertex_index + 1],
                    attrib.vertices[3 * idx0.vertex_index + 2]
                };
                vertex1.position = {
                    attrib.vertices[3 * idx1.vertex_index + 0],
                    attrib.vertices[3 * idx1.vertex_index + 1],
                    attrib.vertices[3 * idx1.vertex_index + 2]
                };
                vertex2.position = {
                    attrib.vertices[3 * idx2.vertex_index + 0],
                    attrib.vertices[3 * idx2.vertex_index + 1],
                    attrib.vertices[3 * idx2.vertex_index + 2]
                };

                // Texture Coordinates
                vertex0.texCoords = (idx0.texcoord_index >= 0) ?
                    glm::vec2(attrib.texcoords[2 * idx0.texcoord_index + 0],
                        attrib.texcoords[2 * idx0.texcoord_index + 1]) :
                    glm::vec2(0.0f, 0.0f);

                vertex1.texCoords = (idx1.texcoord_index >= 0) ?
                    glm::vec2(attrib.texcoords[2 * idx1.texcoord_index + 0],
                        attrib.texcoords[2 * idx1.texcoord_index + 1]) :
                    glm::vec2(0.0f, 0.0f);

                vertex2.texCoords = (idx2.texcoord_index >= 0) ?
                    glm::vec2(attrib.texcoords[2 * idx2.texcoord_index + 0],
                        attrib.texcoords[2 * idx2.texcoord_index + 1]) :
                    glm::vec2(0.0f, 0.0f);

                // Normals
                vertex0.normal = (idx0.normal_index >= 0) ?
                    glm::vec3(attrib.normals[3 * idx0.normal_index + 0],
                        attrib.normals[3 * idx0.normal_index + 1],
                        attrib.normals[3 * idx0.normal_index + 2]) :
                    glm::vec3(0.0f, 0.0f, 0.0f);

                vertex1.normal = (idx1.normal_index >= 0) ?
                    glm::vec3(attrib.normals[3 * idx1.normal_index + 0],
                        attrib.normals[3 * idx1.normal_index + 1],
                        attrib.normals[3 * idx1.normal_index + 2]) :
                    glm::vec3(0.0f, 0.0f, 0.0f);

                vertex2.normal = (idx2.normal_index >= 0) ?
                    glm::vec3(attrib.normals[3 * idx2.normal_index + 0],
                        attrib.normals[3 * idx2.normal_index + 1],
                        attrib.normals[3 * idx2.normal_index + 2]) :
                    glm::vec3(0.0f, 0.0f, 0.0f);

                // Calculate tangents and bitangents
                glm::vec3 edge1 = vertex1.position - vertex0.position;
                glm::vec3 edge2 = vertex2.position - vertex0.position;

                glm::vec2 deltaUV1 = vertex1.texCoords - vertex0.texCoords;
                glm::vec2 deltaUV2 = vertex2.texCoords - vertex0.texCoords;

                float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

                glm::vec3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
                glm::vec3 bitangent = f * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);

                // Accumulate tangents and bitangents
                vertex0.tangent += tangent;
                vertex0.bitangent += bitangent;
                vertex1.tangent += tangent;
                vertex1.bitangent += bitangent;
                vertex2.tangent += tangent;
                vertex2.bitangent += bitangent;

                // Avoid duplicating vertices
                auto addVertex = [&](Vertex& vertex) {
                    // Check if the position already exists in uniquePositions
                    auto posIt = std::find(uniquePositions.begin(), uniquePositions.end(), vertex.position);
                    if (posIt != uniquePositions.end()) {
                        vertex.positionIndex = static_cast<int>(std::distance(uniquePositions.begin(), posIt));
                    } else {
                        vertex.positionIndex = static_cast<int>(uniquePositions.size());
                        uniquePositions.push_back(vertex.position);
                    }

                    // Check if the vertex already exists in vertices
                    auto it = std::find_if(vertices.begin(), vertices.end(), [&](const Vertex& v) {
                        return v == vertex;
                    });
                    if (it == vertices.end()) {
                        uniqueVertices[vertex] = static_cast<unsigned int>(vertices.size());
                        vertices.push_back(vertex);
                    } else {
                        uniqueVertices[vertex] = static_cast<unsigned int>(std::distance(vertices.begin(), it));
                    }
                    indices.push_back(uniqueVertices[vertex]);
                };

                addVertex(vertex0);
                addVertex(vertex1);
                addVertex(vertex2);

                // Update the triangle pointers to point to the vertices stored in the vertices vector
                triangles.push_back(Triangle(&vertices[uniqueVertices[vertex0]], &vertices[uniqueVertices[vertex1]], &vertices[uniqueVertices[vertex2]]));
            }
        }

        // Normalize tangents and bitangents
        for (auto& vertex : vertices) {
            vertex.tangent = glm::normalize(vertex.tangent);
            vertex.bitangent = glm::normalize(vertex.bitangent);
        }

        indexCount = indices.size(); // Store the count of indices
        positions = uniquePositions;
    }

    void setupMesh() {
        std::cout << "Setting up mesh\n";

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        // Vertex Buffer
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

        // Element Buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        // Vertex Attributes
        // Position
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // Texture Coordinates
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
        // Normals
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
        // Tangents
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));
        // biTangents
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));
        // Bone Index
        glEnableVertexAttribArray(5);
        glVertexAttribIPointer(5, 1, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, boneIndex));

        glBindVertexArray(0);
    }

    void loadTexture() {
        if (!texturePath.empty()) {
            material.loadTexture(texturePath);
        }
		else material.texture.fillTexture(glm::vec4(1.0f));
		if (!normalMapPath.empty()) {
			material.loadNormalMap(normalMapPath);
		}
        else material.texture.fillTexture(glm::vec4(1.0f));
        if (!specularMapPath.empty()) {
            material.loadSpecularMap(specularMapPath);
        }
        else material.texture.fillTexture(glm::vec4(1.0f));
		if (!displacementMapPath.empty()) {
			material.loadDisplacementMap(displacementMapPath);
		}
        else material.texture.fillTexture(glm::vec4(1.0f));
		if (!roughnessMapPath.empty()) {
			material.loadRoughnessMap(roughnessMapPath);
		}
        else material.texture.fillTexture(glm::vec4(1.0f));
		if (!aoMapPath.empty()) {
			material.loadAOMap(aoMapPath);
		}
        else material.texture.fillTexture(glm::vec4(1.0f));
    }

    void bind(int shaderProgram) {
        if (useBuiltInMesh) {
            glBindVertexArray(VAO);

			material.bindAllTextures(shaderProgram);
        }
        else {
            bindMeshFunction();
        }
    }

    void draw(Shader& shader) {
		shader.setBool("debug", debugMode);
        shader.setInt("diffuseTexture", 0);
        shader.setInt("normalMap", 1);
        shader.setInt("specularMap", 2);
        shader.setInt("displacementMap", 3);
        shader.setInt("roughnessMap", 4);
        shader.setInt("aoMap", 5);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    void draw(Shader& shader, bool debug) {
        shader.setBool("debug", debug);
        shader.setInt("diffuseTexture", 0);
        shader.setInt("normalMap", 1);
        shader.setInt("specularMap", 2);
        shader.setInt("displacementMap", 3);
        shader.setInt("roughnessMap", 4);
        shader.setInt("aoMap", 5);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

std::array<Vertex, 3> createTriangle(std::array<glm::vec3, 3> positions, std::array<glm::vec2, 3> texCoords, std::array<glm::vec3, 3> normals) {
    std::array<Vertex, 3> vertices;
    for (size_t i = 0; i < positions.size(); ++i) {
        Vertex vertex;
        vertex.position = positions[i];
        vertex.texCoords = texCoords[i];
        vertex.normal = normals[i];
        vertices[i] = vertex;
    }

    glm::vec3 edge1 = vertices[1].position - vertices[0].position;
    glm::vec3 edge2 = vertices[2].position - vertices[0].position;
    glm::vec2 deltaUV1 = vertices[1].texCoords - vertices[0].texCoords;
    glm::vec2 deltaUV2 = vertices[2].texCoords - vertices[0].texCoords;

    float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

    glm::vec3 tangent;
    tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
    tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
    tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
    tangent = glm::normalize(tangent);

    glm::vec3 bitangent;
    bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
    bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
    bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
    bitangent = glm::normalize(bitangent);

    for (auto& vertex : vertices) {
        vertex.tangent = tangent;
        vertex.bitangent = bitangent;
    }

    return vertices;
}

std::vector<std::array<Vertex, 3>> createTriangleMesh(std::vector<glm::vec3> positions, std::vector<glm::vec2> texCoords, std::vector<glm::vec3> normals) {
    std::vector<std::array<Vertex, 3>> triangles;
    size_t width = static_cast<size_t>(sqrt(positions.size())) - 1;
    size_t height = width;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t topLeft = y * (width + 1) + x;
            size_t topRight = topLeft + 1;
            size_t bottomLeft = topLeft + (width + 1);
            size_t bottomRight = bottomLeft + 1;

            // First triangle (top-left, bottom-left, bottom-right)
            std::array<glm::vec3, 3> pos1 = { positions[topLeft], positions[bottomLeft], positions[bottomRight] };
            std::array<glm::vec2, 3> tex1 = { texCoords[topLeft], texCoords[bottomLeft], texCoords[bottomRight] };
            std::array<glm::vec3, 3> norm1 = { normals[topLeft], normals[bottomLeft], normals[bottomRight] };
            triangles.push_back(createTriangle(pos1, tex1, norm1));

            // Second triangle (top-left, bottom-right, top-right)
            std::array<glm::vec3, 3> pos2 = { positions[topLeft], positions[bottomRight], positions[topRight] };
            std::array<glm::vec2, 3> tex2 = { texCoords[topLeft], texCoords[bottomRight], texCoords[topRight] };
            std::array<glm::vec3, 3> norm2 = { normals[topLeft], normals[bottomRight], normals[topRight] };
            triangles.push_back(createTriangle(pos2, tex2, norm2));
        }
    }
    return triangles;
}

class ProgrammebleMesh : public Component {
public:
    bool debugMode = false;
    
    ProgrammebleMesh() {};

    void addTriangle(std::array<Vertex, 3> triangle) {
        std::array<uint32_t, 3> newIndices;
        std::array<Vertex*, 3> vertexPtrs;

        for (int i = 0; i < 3; ++i) {
            auto it = std::find(vertices.begin(), vertices.end(), triangle[i]);
            if (it != vertices.end()) {
                newIndices[i] = std::distance(vertices.begin(), it);
            }
            else {
                vertices.push_back(triangle[i]);
                newIndices[i] = vertices.size() - 1;
            }
            vertexPtrs[i] = &vertices[newIndices[i]];
        }

        indices.insert(indices.end(), newIndices.begin(), newIndices.end());

        // Store triangle and edges
        triangles.insert(Triangle(vertexPtrs[0], vertexPtrs[1], vertexPtrs[2]));
        edges.insert(Edge(vertexPtrs[0], vertexPtrs[1]));
        edges.insert(Edge(vertexPtrs[1], vertexPtrs[2]));
        edges.insert(Edge(vertexPtrs[2], vertexPtrs[0]));
    }

    void removeTriangle(std::array<Vertex, 3> triangle) {
        for (size_t i = 0; i < indices.size(); i += 3) {
            if (vertices[indices[i]] == triangle[0] &&
                vertices[indices[i + 1]] == triangle[1] &&
                vertices[indices[i + 2]] == triangle[2]) {

                std::array<Vertex*, 3> vertexPtrs = {
                    &vertices[indices[i]],
                    &vertices[indices[i + 1]],
                    &vertices[indices[i + 2]]
                };

                indices.erase(indices.begin() + i, indices.begin() + i + 3);

                Triangle t(vertexPtrs[0], vertexPtrs[1], vertexPtrs[2]);
                triangles.erase(t);

                Edge e1(vertexPtrs[0], vertexPtrs[1]);
                Edge e2(vertexPtrs[1], vertexPtrs[2]);
                Edge e3(vertexPtrs[2], vertexPtrs[0]);

                edges.erase(e1);
                edges.erase(e2);
                edges.erase(e3);

                break;
            }
        }
    }

    std::vector<Vertex>& getVertices() {
        return vertices;
    }

    void paintTexture(glm::vec4 color) {
		material.texture.fillTexture(color);
    }

    void sendVertexData() {

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        // Vertex Buffer
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

        // Element Buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        // Vertex Attributes
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));
        glEnableVertexAttribArray(5);
        glVertexAttribIPointer(5, 1, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, boneIndex));

        glBindVertexArray(0);
    }

    void bindTexture(int shader) {
		material.bindAllTextures(shader);
    }

    void draw(Shader& shader) {
		shader.setBool("debug", debugMode);
        shader.setInt("diffuseTexture", 0);
        shader.setInt("normalMap", 1);
        shader.setInt("specularMap", 2);
        shader.setInt("displacementMap", 3);
        shader.setInt("roughnessMap", 4);
        shader.setInt("aoMap", 5);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    Material material;
    std::unordered_set<Triangle, std::hash<Triangle>, TriangleEqual> triangles;
private:
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    GLuint VAO, VBO, EBO;
    std::unordered_set<Edge, std::hash<Edge>, EdgeEqual> edges;
    
};

class AnimationMesh;

struct SphereCollider {
    glm::vec3 displayPosition;
    glm::vec3 actualPosition;
    float radius;
    float distance = 1.0f;
    Mesh visualizer = Mesh("Tetrahedron.obj", "", "", "", "", "", "", true);
    bool isIntersectingRay(Ray& ray) {
        return glm::intersectRaySphere(ray.origin, ray.direction, actualPosition, radius * radius, distance);
    }
};

struct Bone {
    Mesh visualizer = Mesh("Tetrahedron.obj", "", "", "", "", "", "", true);
    AnimationMesh* mesh = nullptr;
    Bone* parent = nullptr;
    SphereCollider collider;
    std::string name;
    glm::mat4 offsetMatrix;
    std::vector<Vertex*> vertices;

    Bone(const std::string& name, const glm::mat4& offsetMatrix)
        : name(name), offsetMatrix(offsetMatrix) {
        collider.radius = 0.5f;
    }

    Bone() { collider.radius = 0.5f; }

    // Attach vertex
    void attachVertex(Vertex* vertex);

    glm::mat4 getOffsetMatrix() const {
        if (parent == nullptr) {
            return offsetMatrix;
        }
        return parent->getOffsetMatrix() * offsetMatrix;
    }

    void updateCollider(glm::mat4& model) {
        collider.actualPosition = glm::vec3(getOffsetMatrix() * model * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    }

    bool checkSelect(Ray& ray) {
        return collider.isIntersectingRay(ray);
    }
};

class AnimationMesh : public Mesh {

public:
	bool drawGizmos = false;
    std::vector<SphereCollider> vertexColliders;

    AnimationMesh(const std::string& objFilePath, const std::string& texturePath, const std::string& normalMapPath, const std::string& specularMapPath, const std::string& displacementMapPath, const std::string& roughnessMapPath, const std::string& aoMapPath, bool useBuiltInMesh = true)
    : Mesh(objFilePath, texturePath, normalMapPath, specularMapPath, displacementMapPath, roughnessMapPath, aoMapPath, useBuiltInMesh) {};

	void addBone(const std::string& name, const glm::mat4& offsetMatrix) {
		Bone bone;
		bone.name = name;
		bone.offsetMatrix = offsetMatrix;
		bones.push_back(bone);
	}

    void createVertexColliders() {
        for (auto& vert : positions) {
            auto coll = SphereCollider();
            coll.displayPosition = vert;
            coll.radius = 0.25f;
            vertexColliders.push_back(coll);
        }

        std::cout << "Sphere Collider Count : " << vertexColliders.size() << "\n";
    }

    void drawVertices(Shader& shader, glm::mat4& model) {
        for (auto& collider : vertexColliders) {
            shader.setBool("gizmo", true);
            shader.setVec3("gizmoColor", glm::vec3(1.0f, 0.0f, 1.0f));
            shader.setMat4("model", glm::translate(model, collider.displayPosition));
            shader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));
            collider.visualizer.bind(shader.ID);
            collider.visualizer.draw(shader, false);
        }
    }

    int selectVertex(Ray& ray) {
        for (int i = 0; i < vertexColliders.size(); i++) {
            if (vertexColliders[i].isIntersectingRay(ray)) {
                std::cout << "Collider Index: " << i << "\n";
                return i;
            }
        }
        return -1;
    }

    Bone* selectBone(Ray& ray, glm::mat4& model) {
        for (int i = 0; i < bones.size(); i++) {
            bones[i].updateCollider(model);
            if (bones[i].checkSelect(ray)) {
                std::cout << "Bone Selected: " << i << '\n';
                std::cout << glm::to_string(bones[i].collider.actualPosition) << '\n';
                return &bones[i];
            }
        }
        return nullptr;
    }

    void updateColliderPositions(glm::mat4& model) {
        for (auto& collider : vertexColliders) {
            collider.actualPosition = glm::vec3(model * glm::vec4(collider.displayPosition, 1.0f));
        }
    }

    void attachVertexToBone(int vertexIndex, const std::string& boneName) {
        auto posIndex = vertices[vertexIndex].positionIndex;
        std::vector<Vertex*> verts;
        for (auto& vert : vertices) {
            if (vert.positionIndex == posIndex) {
                verts.push_back(&vert);
            }
        }

        for (auto& bone : bones) {
            if (bone.name == boneName) {
                for (auto& ver : verts) {
                    ver->boneIndex = bones.size() - 1;
                    bones[bones.size() - 1].attachVertex(ver);
                }
                break;
            }
        }
        setupMesh();
    }

    Bone& getBone(const std::string& name) {
        for (auto& bone : bones) {
            if (bone.name == name) {
                return bone;
            }
        }
    };

    void sendBoneData(Shader& shader) {
		shader.setInt("numBones", bones.size());
        for (size_t i = 0; i < bones.size(); ++i) {
            shader.setMat4("bones[" + std::to_string(i) + "]", bones[i].getOffsetMatrix());
        }
    }

    void drawGizmo(Shader& shader, glm::mat4& model) {

		glDisable(GL_DEPTH_TEST);
		for (auto& bone : bones) {
            shader.setBool("gizmo", true);
            shader.setVec3("gizmoColor", glm::vec3(0.0f, 1.0f, 0.0f));
			shader.setMat4("model", model * bone.getOffsetMatrix());
            shader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model * bone.getOffsetMatrix()))));
			bone.visualizer.bind(shader.ID);
			bone.visualizer.draw(shader, false);
		}
        drawVertices(shader, model);
		glEnable(GL_DEPTH_TEST);
        updateColliderPositions(model);
	}

    std::vector<Bone> bones;
};

void Bone::attachVertex(Vertex* vertex) {
    if (mesh) {
        auto& meshVertices = mesh->vertices;
        auto& meshBones = mesh->bones;

        auto vertexIt = std::find(meshVertices.begin(), meshVertices.end(), *vertex);
        auto boneIt = std::find_if(meshBones.begin(), meshBones.end(), [this](const Bone& b) { return b.name == this->name; });

        if (vertexIt != meshVertices.end() && boneIt != meshBones.end()) {
            vertices.push_back(vertex);
        }
    }
}

struct Particle {
	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec3 acceleration;
	float mass;
};

struct Constraint {
	virtual void solve(float compliance) = 0;
};

struct DistanceConstraint: public Constraint {
	Particle* p1;
	Particle* p2;
	float restDistance;

	DistanceConstraint(Particle* p1, Particle* p2, float restDistance) : p1(p1), p2(p2), restDistance(restDistance) {};

    void solve(float compliance) override {
		std::cout << "Solving distance constraint\n";
		auto distance = glm::distance(p1->position, p2->position);
		auto correction = compliance * (distance - restDistance) / distance;
		auto correctionVector = 0.5f * correction * (p1->position - p2->position);
		p1->position -= correctionVector;
		p2->position += correctionVector;
    }
};

struct BendingConstraint : public Constraint {
    Particle* p1;
    Particle* p2;
    Particle* p3;
    Particle* p4;

    float restAngle;

	BendingConstraint(Particle* p1, Particle* p2, Particle* p3, Particle* p4, float restAngle)
		: p1(p1), p2(p2), p3(p3), p4(p4), restAngle(restAngle) {
	};

    void solve(float compliance) override {
        auto v1 = p2->position - p1->position;
        auto v2 = p3->position - p2->position;
        auto v3 = p4->position - p3->position;
        auto n1 = glm::normalize(glm::cross(v1, v2));
        auto n2 = glm::normalize(glm::cross(v2, v3));
        auto angle = glm::acos(glm::dot(n1, n2));
        auto correction = compliance * (angle - restAngle) / angle;
        auto correctionVector1 = 0.5f * correction * glm::cross(n1, v2);
        auto correctionVector2 = 0.5f * correction * (glm::cross(n1, v2) + glm::cross(n2, v2));
        auto correctionVector3 = 0.5f * correction * glm::cross(n2, v2);
        auto correctionVector4 = 0.5f * correction * glm::cross(n2, v2);

        // Project the correction vectors onto the line p2->p3
        auto lineDir = glm::normalize(v2);
        correctionVector1 = glm::dot(correctionVector1, lineDir) * lineDir;
        correctionVector2 = glm::dot(correctionVector2, lineDir) * lineDir;
        correctionVector3 = glm::dot(correctionVector3, lineDir) * lineDir;
        correctionVector4 = glm::dot(correctionVector4, lineDir) * lineDir;

        p1->position += correctionVector1;
        p2->position += correctionVector2;
        p3->position += correctionVector3;
        p4->position += correctionVector4;
    };
};

class Cloth : public Component {
public:
    std::shared_ptr<ProgrammebleMesh> mesh;
    std::vector<Particle> particles;
    std::vector<Constraint*> constraints;

    Cloth(float compliance, float mass, int width, int height) {
        float spacing = 0.1f;
        int numParticles = (width + 1) * (height + 1);
        particles.reserve(numParticles);

		mesh = std::make_shared<ProgrammebleMesh>();

        // Create Particles
        for (int y = 0; y <= height; y++) {
            for (int x = 0; x <= width; x++) {
                Particle p;
                p.position = glm::vec3(x * spacing, y * spacing, 0.0f);
                p.mass = mass;
                particles.push_back(p);
            }
        }

        // Create Distance Constraints
        for (int y = 0; y <= height; y++) {
            for (int x = 0; x <= width; x++) {
                int index = y * (width + 1) + x;
                if (x < width) { // Right neighbor
                    constraints.push_back(new DistanceConstraint{ &particles[index], &particles[index + 1], spacing });
                }
                if (y < height) { // Bottom neighbor
                    constraints.push_back(new DistanceConstraint{ &particles[index], &particles[index + (width + 1)], spacing });
                }
            }
        }
        
        // Create Bending Constraints
        //for (int y = 0; y <= height - 2; y++) {
        //    for (int x = 0; x <= width - 2; x++) {
        //        int index = y * (width + 1) + x;
        //        constraints.push_back(new BendingConstraint{
        //            &particles[index],
        //            &particles[index + 1],
        //            &particles[index + (width + 1)],
        //            &particles[index + (width + 2)],
        //            glm::radians(180.0f) // Default rest angle
        //            });
        //    }
        //}

        // Create Mesh
        std::vector<glm::vec3> positions;
        std::vector<glm::vec2> texCoords;
        std::vector<glm::vec3> normals(particles.size(), glm::vec3(0, 0, 1));

        for (auto& p : particles) {
            positions.push_back(p.position);
            texCoords.emplace_back(p.position.x / (width * spacing), p.position.y / (height * spacing));
        }

        auto triangles = createTriangleMesh(positions, texCoords, normals);
        for (auto& tri : triangles) {
            mesh->addTriangle(tri);
        }
    }

    void init() {
        owner->addComponent(mesh);
        mesh->sendVertexData();
        float compliance = 0.1f;
        for (auto constraint : constraints) {
            constraint->solve(compliance);
        }
    }

	void solve() {
		float compliance = 0.5f;
		for (auto constraint : constraints) {
			constraint->solve(compliance);
		}

		for (int i = 0; i < particles.size(); ++i) {
			mesh->getVertices()[i].position = particles[i].position;
		}
		mesh->sendVertexData();
	}

    ~Cloth() {
        for (auto constraint : constraints) {
            delete constraint;
        }
    }
};

class Physics {
public:
    std::vector<Entity*> physicsBodies;

    void addPhysicsBody(Entity* entity) {
        physicsBodies.push_back(entity);
    };

	~Physics() {
		for (auto body : physicsBodies) {
			delete body;
		}
	}
};

class BoneEditor {
public:
    BoneEditor() {};

    AnimationMesh* mesh;

    std::vector<int> selectedVertices;
	Bone* selectedBone = nullptr;
    std::vector<Bone*> bonesToAttach = { nullptr };

    void selectVertex(Ray& ray) {
        selectedVertices.push_back(mesh->selectVertex(ray));
    };

    void selectBone(Ray& ray, glm::mat4& model) {
        selectedBone = mesh->selectBone(ray, model);
        if (selectedBone != nullptr) {
            std::cout << "selected Bone: " << selectedBone->name << '\n';
        }
	};

	void selectChildBones(Bone* bone) {
		for (auto& b : mesh->bones) {
			if (b.parent == bone) {
				bonesToAttach.push_back(&b);
			}
		}
	};

	void connectBoneToVertices() {
		for (auto& vertex : selectedVertices) {
			if (selectedBone) {
				mesh->attachVertexToBone(vertex, selectedBone->name);
			}
		}
	}

    void connectChildBones() {
        for (auto& bone : bonesToAttach) {
			bone->parent = selectedBone;
        }
    }

	void clearSelections() {
		selectedVertices.clear();
        bonesToAttach.clear();
        selectedBone = nullptr;
	}

    void printSelectedVertices() {
        std::cout << selectedVertices.size() << " Vertices selected: \n";
        for (auto vertex : selectedVertices) {
            std::cout << '\t' << vertex << '\n';
        }
    }
};

class PointLight: public Component{
public:
    glm::vec3 color;
    float intensity;

    PointLight(glm::vec3 color, float intensity) : color(color), intensity(intensity) {};

	void setUniforms(Shader& shader, const std::string& name) {
		shader.setVec3(name + ".position", owner->transform.getPosition());
		shader.setVec3(name + ".color", color);
		shader.setFloat(name + ".intensity", intensity);
	}
};

enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera {
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    float Yaw;
    float Pitch;

    float MovementSpeed;
    float MouseSensitivity;

public:

    glm::vec3 Front;
    glm::vec3 Position;
    float Zoom;
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(25.5f), MouseSensitivity(0.1f), Zoom(45.0f) {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    Camera() {};

    glm::mat4 GetViewMatrix() {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 GetProjectionMatrix(float windowHeight, float windowWidth, float FOV) {
		return glm::perspective(glm::radians(FOV), windowWidth / windowHeight, 0.1f, 100.0f);
    }

    glm::mat4 GetOrthogonalMatrix(float windowHeight, float windowWidth, float FOV) {
        float aspectRatio = windowWidth / windowHeight;
        float orthoHeight = FOV;
        float orthoWidth = FOV * aspectRatio;
        return glm::ortho<float>(-120, 120, -120, 120, -500, 500);
    }

    void ProcessKeyboard(int direction, float deltaTime) {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            Position += Front * velocity;
        if (direction == BACKWARD)
            Position -= Front * velocity;
        if (direction == LEFT)
            Position -= Right * velocity;
        if (direction == RIGHT)
            Position += Right * velocity;
    }

    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch) {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        xoffset = 0.0f;
        yoffset = 0.0f;

        if (constrainPitch) {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        updateCameraVectors();
    }

    void ProcessMouseScroll(float yoffset) {
        Zoom -= yoffset;
        if (Zoom < 1.0f)
            Zoom = 1.0f;
        if (Zoom > 45.0f)
            Zoom = 45.0f;
    }

    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);

        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up = glm::normalize(glm::cross(Right, Front));
    }
};

Ray shootRayFromMouse(Camera& camera, const glm::vec2& windowDims, const glm::vec2& mousePosition) {
	float x = (2.0f * mousePosition.x) / windowDims.x - 1.0f;
	float y = 1.0f - (2.0f * mousePosition.y) / windowDims.y;
	float z = 1.0f;
	glm::vec3 ray_nds = glm::vec3(x, y, z);
	glm::vec4 ray_clip = glm::vec4(ray_nds.x, ray_nds.y, -1.0, 1.0);
	glm::vec4 ray_eye = glm::inverse(camera.GetProjectionMatrix(windowDims.y, windowDims.x, 45.0f)) * ray_clip;
	ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);
	glm::vec3 ray_wor = glm::vec3(glm::inverse(camera.GetViewMatrix()) * ray_eye);
	ray_wor = glm::normalize(ray_wor);
	return Ray{ camera.Position, ray_wor };
}

const char* vertexShaderSource = "#version 330 core\n\nlayout (location = 0) in vec3 aPos;\nlayout (location = 1) in vec2 aTexCoords;\nlayout (location = 2) in vec3 aNormal;\nlayout (location = 3) in vec3 aTangent;\nlayout (location = 4) in vec3 aBitangent;\n\nout vec2 TexCoords;\nout vec3 WorldPos;\nout vec3 Normal;\nout mat3 TBN;\n\nuniform mat4 projection;\nuniform mat4 view;\nuniform mat4 model;\nuniform mat3 normalMatrix;\n\nvoid main()\n{\n    TexCoords = aTexCoords;\n    WorldPos = vec3(model * vec4(aPos, 1.0));\n    Normal = normalMatrix * aNormal;\n\n    vec3 T = normalize(mat3(model) * aTangent);\n    vec3 B = normalize(mat3(model) * aBitangent);\n    vec3 N = normalize(mat3(model) * aNormal);\n\n    TBN = transpose(mat3(T, B, N));\n\n    gl_Position =  projection * view * vec4(WorldPos, 1.0);\n}";

const char* fragmentShaderSource = "#version 330 core\n\nin vec2 TexCoords;\nin vec3 WorldPos;\nin vec3 Normal;\nin mat3 TBN;\n\nstruct PointLight {\n\tvec3 position;\n\tvec3 color;\n\tfloat intensity;\n};\n\nout vec4 FragColor;\n\nuniform sampler2D diffuseTexture;\nuniform sampler2D normalMap;\nuniform sampler2D specularMap;\nuniform sampler2D displacementMap;\nuniform sampler2D roughnessMap;\nuniform sampler2D aoMap;\n\nuniform PointLight pointLights[4];\nuniform vec3 camPos;\n\nconst float PI = 3.14159265359;\n\nvec2 parallaxMapping(vec2 texCoords, vec3 viewDir) {\n    // number of depth layers\n    const float minLayers = 24.0;\n    const float maxLayers = 84.0;\n    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));\n    // calculate the size of each layer\n    float layerDepth = 1.0 / numLayers;\n    // depth of current layer\n    float currentLayerDepth = 0.0;\n    // the amount to shift the texture coordinates per layer (From vector P)\n    vec2 P = viewDir.xy * 0.1f;\n    vec2 deltaTexCoords = P / numLayers;\n\n    // get initial values\n    vec2 currentTexCoords = texCoords;\n    float currentDepthMapValue = textureGrad(displacementMap, currentTexCoords, dFdx(texCoords), dFdy(texCoords)).r;\n    \n    while(currentLayerDepth < currentDepthMapValue)\n    {\n        // shift texture coordinates along direction of P\n        currentTexCoords -= deltaTexCoords;\n        // get depthmap value at current texture coordinates\n        currentDepthMapValue = textureGrad(displacementMap, currentTexCoords, dFdx(texCoords), dFdy(texCoords)).r;\n        // get depth of next layer\n        currentLayerDepth += layerDepth;\n    }\n\n    // Relief Parallax Mapping\n\n    // decrease shift and height of layer by half\n    deltaTexCoords /= 2;\n    layerDepth /= 2;\n\n    // return to the mid point of previous layer\n    currentTexCoords += deltaTexCoords;\n    currentLayerDepth -= layerDepth;\n\n    // binary search to increase precision of Steep Paralax Mapping\n    const int numSearches = 5;\n    for(int i = 0; i < numSearches; ++i)\n    {\n        // decrease shift and height of layer by half\n        deltaTexCoords /= 2;\n        layerDepth /=2;\n        \n        // new depth from heightmap\n        currentDepthMapValue = textureGrad(displacementMap, currentTexCoords, dFdx(texCoords), dFdy(texCoords)).r;\n\n        // shift along or aginas vector ViewDir\n        if(currentDepthMapValue > currentLayerDepth)\n        {\n            currentTexCoords -= deltaTexCoords;\n            currentLayerDepth += layerDepth;\n        }\n        else\n        {\n            currentTexCoords += deltaTexCoords;\n            currentLayerDepth -= layerDepth;\n        }\n    }\n    // get texture coordinates before collision (reverse operations)\n    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;\n    \n    // get depth after and before collision for linear interpolation\n    float afterDepth = currentDepthMapValue - currentLayerDepth;\n    float beforeDepth = textureGrad(displacementMap, prevTexCoords, dFdx(texCoords), dFdy(texCoords)).r - currentLayerDepth + layerDepth;\n\n    // interpolation of texture coordinates\n    float weight = afterDepth / (afterDepth - beforeDepth);\n    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);\n\n    return finalTexCoords;\n}\n\nvec3 getNormalFromMap(vec2 texCoord)\n{\n    vec3 tangentNormal = textureGrad(normalMap, texCoord, dFdx(texCoord), dFdy(texCoord)).xyz * 2.0 - 1.0;\n\n    return normalize(inverse(TBN) * tangentNormal);\n}\n\n// ----------------------------------------------------------------------------\nfloat DistributionGGX(vec3 N, vec3 H, float roughness)\n{\n    float a = roughness*roughness;\n    float a2 = a*a;\n    float NdotH = max(dot(N, H), 0.0);\n    float NdotH2 = NdotH*NdotH;\n\n    float nom   = a2;\n    float denom = (NdotH2 * (a2 - 1.0) + 1.0);\n    denom = PI * denom * denom;\n\n    return nom / denom;\n}\n// ----------------------------------------------------------------------------\nfloat GeometrySchlickGGX(float NdotV, float roughness)\n{\n    float r = (roughness + 1.0);\n    float k = (r*r) / 8.0;\n\n    float nom   = NdotV;\n    float denom = NdotV * (1.0 - k) + k;\n\n    return nom / denom;\n}\n// ----------------------------------------------------------------------------\nfloat GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)\n{\n    float NdotV = max(dot(N, V), 0.0);\n    float NdotL = max(dot(N, L), 0.0);\n    float ggx2 = GeometrySchlickGGX(NdotV, roughness);\n    float ggx1 = GeometrySchlickGGX(NdotL, roughness);\n\n    return ggx1 * ggx2;\n}\n// ----------------------------------------------------------------------------\nvec3 fresnelSchlick(float cosTheta, vec3 F0)\n{\n    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);\n}\n// ----------------------------------------------------------------------------\n\nvoid main(){\n    \n    vec3 viewDir = normalize(TBN * (camPos - WorldPos));\n    vec2 texCoord = parallaxMapping(TexCoords, viewDir);\n    vec3 albedo     = pow(textureGrad(diffuseTexture, texCoord, dFdx(texCoord), dFdy(texCoord)).rgb, vec3(2.2));\n    float metallic  = textureGrad(specularMap, texCoord, dFdx(texCoord), dFdy(texCoord)).r;\n    float roughness = textureGrad(roughnessMap, texCoord, dFdx(texCoord), dFdy(texCoord)).r;\n    float ao        = textureGrad(aoMap, texCoord, dFdx(texCoord), dFdy(texCoord)).r;\n\n    vec3 N = getNormalFromMap(texCoord);\n    vec3 V = normalize(camPos - WorldPos);\n\n    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 \n    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    \n    vec3 F0 = vec3(0.04); \n    F0 = mix(F0, albedo, metallic);\n\n    float shadow = 0.0;\n\n    // reflectance equation\n    vec3 Lo = vec3(0.0);\n    for(int i = 0; i < 4; ++i) \n    {\n\n        if (shadow < 1.0) {\n            // calculate per-light radiance\n            vec3 L = normalize(pointLights[i].position - WorldPos);\n            vec3 H = normalize(V + L);\n            float distance = length(pointLights[i].position - WorldPos);\n            float attenuation = 1.0 / (distance * distance);\n            vec3 radiance = pointLights[i].color * attenuation * 500.0f;\n\n            // Cook-Torrance BRDF\n            float NDF = DistributionGGX(N, H, roughness);   \n            float G   = GeometrySmith(N, V, L, roughness);      \n            vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);\n               \n            vec3 numerator    = NDF * G * F; \n            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero\n            vec3 specular = numerator / denominator;\n            \n            // kS is equal to Fresnel\n            vec3 kS = F;\n            // for energy conservation, the diffuse and specular light can't\n            // be above 1.0 (unless the surface emits light); to preserve this\n            // relationship the diffuse component (kD) should equal 1.0 - kS.\n            vec3 kD = vec3(1.0) - kS;\n            // multiply kD by the inverse metalness such that only non-metals \n            // have diffuse lighting, or a linear blend if partly metal (pure metals\n            // have no diffuse light).\n            kD *= 1.0 - metallic;\t  \n\n            // scale light by NdotL\n            float NdotL = max(dot(N, L), 0.0);\n\n            // add to outgoing radiance Lo\n            Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again\n        }\n    }   \n    \n    // ambient lighting (note that the next IBL tutorial will replace \n    // this ambient lighting with environment lighting).\n    vec3 ambient = vec3(0.03) * albedo * ao;\n    \n    vec3 color = (ambient + Lo);\n\n    // HDR tonemapping\n    color = color / (color + vec3(1.0));\n    // gamma correct\n    color = pow(color, vec3(1.0/2.2));\n    FragColor = vec4(color, 1.0);\n}";

class Scene;

class EntitySelector {
public:
    Entity* selectedEntity = nullptr;
    EntitySelector(){};
    
    void selectEntity(const glm::vec3& cameraPos, const glm::vec3& cameraDir, std::vector<Entity>& entities);
};

class Scene {
public:
    std::vector<Entity> entities;
    Shader shader;
    Camera camera;
    glm::vec2 windowSize;
    bool useBuiltInShaders = true;
    CustomFunction render{};
    bool gizmo = false;
    EntitySelector entitySelector;

    Scene(Camera& camera, const Shader& shader = Shader(vertexShaderSource, fragmentShaderSource, true), bool useBuiltInShaders = true) : shader(shader), camera(camera), useBuiltInShaders(useBuiltInShaders) {};
    Scene() {};

    void renderScene() {

		std::vector<PointLight*> pointLights;

        for (auto& entity : entities) {
			if (entity.hasComponent<PointLight>()) {
				pointLights.push_back(entity.getComponent<PointLight>().get());
			}
        }

        shader.use();
        glViewport(0, 0, windowSize.x, windowSize.y);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), windowSize.x / windowSize.y, 0.01f, 1000.0f);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
		shader.setVec3("camPos", camera.Position);
		for (int i = 0; i < pointLights.size(); i++) {
			pointLights[i]->setUniforms(shader, "pointLights[" + std::to_string(i) + "]");
		}

        // Render the scene
        for (auto& entity : entities) {
            entity.update();
            if (entity.hasComponent<Mesh>()) {
                
                shader.setMat4("model", entity.transform.model);
                shader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(entity.transform.model))));          
                shader.setInt("diffuseTexture", 0);
                shader.setInt("normalMap", 1);
                shader.setInt("specularMap", 2);
                shader.setInt("displacementMap", 3);
                shader.setInt("roughnessMap", 4);
                shader.setInt("aoMap", 5);
                entity.getComponent<Mesh>()->bind(shader.ID);
                glDrawElements(GL_TRIANGLES, entity.getComponent<Mesh>()->indexCount, GL_UNSIGNED_INT, 0);
            }
        }
    }

    template <typename Render, typename... Args>
    void setRenderFunction(Render&& renderFunction, Args&&... args) {
		render.setFunction(renderFunction, args...);
    }

	void addEntity(Entity& entity) {
		entities.push_back(std::move(entity));
	}
	void setWindowSize(glm::vec2 windowSize) {
		this->windowSize = windowSize;
	}
};

void EntitySelector::selectEntity(const glm::vec3& cameraPos, const glm::vec3& cameraDir, std::vector<Entity>& entities) {

    for (auto& entity : entities) {
		float distance;
        std::cout << glm::to_string(entity.transform.getPosition()) << "\n";
        if (glm::intersectRaySphere(cameraPos, cameraDir, entity.transform.getPosition(), 1.0f, distance)) {
			selectedEntity = &entity;
            return;
        }
    }
	selectedEntity = nullptr;
}

float xoffset = 0.0f;
float yoffset = 0.0f;

class Engine {
public:
	int WIDTH;
	int HEIGHT;
    static bool cursorEnabled;

    float deltaTime;
    float lastFrame;
    static Scene scene;
    BoneEditor boneEditor;
    GLFWwindow* window;
	Engine(int WIDTH, int HEIGHT) : WIDTH(WIDTH), HEIGHT(HEIGHT), window(nullptr) {}

    // Store the previous state of the mouse buttons
    std::unordered_map<int, bool> mouseButtonStates;

    bool isMouseButtonPressed(GLFWwindow* window, int button) {
        int currentState = glfwGetMouseButton(window, button);

        // Check if the button was not pressed previously and is now pressed
        if (currentState == GLFW_PRESS && !mouseButtonStates[button]) {
            mouseButtonStates[button] = true;
            return true;
        }
        // If the button is released, remove it from the map
        else if (currentState == GLFW_RELEASE) {
            mouseButtonStates.erase(button);
        }

        return false;
    }

    std::unordered_map<int, bool> releaseMouseButtonStates;

    bool isMouseButtonReleased(GLFWwindow* window, int button) {
        int currentState = glfwGetMouseButton(window, button);

        // Check if the button was pressed previously and is now released
        if (currentState == GLFW_RELEASE && releaseMouseButtonStates[button]) {
            releaseMouseButtonStates[button] = false; // Update the state to released
            return true;
        }
        // If the button is pressed, update the map
        else if (currentState == GLFW_PRESS) {
            releaseMouseButtonStates[button] = true;
        }

        return false;
    }

    bool isMouseButtonPressedDown(GLFWwindow* window, int button) {
        if (glfwGetMouseButton(window, button)) return true;
        else return false;
    }
    
    glm::vec2 getCursorPosition(GLFWwindow* window) {
        double xPos;
        double yPos;
        glfwGetCursorPos(window, &xPos, &yPos);
        return glm::vec2(xPos, yPos);
    }

    std::unordered_map<int, bool> keyStates;

    bool isKeyPressed(GLFWwindow* window, int key) {
        int currentState = glfwGetKey(window, key);

        // Check if key was not pressed previously and is now pressed
        if (currentState == GLFW_PRESS && !keyStates[key]) {
            keyStates[key] = true;
            return true;
        }
        // Update the key state
        else if (currentState == GLFW_RELEASE) {
            keyStates.erase(key);
        }

        return false;
    }

    bool isKeyPressedDown(GLFWwindow* window, int key) {
        if (glfwGetKey(window, key) == GLFW_PRESS) {
            return true;
        }
        else return false;
    }

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        static float lastX = 400, lastY = 300;
        static bool firstMouse = true;
        static float movementInterval = 0.005f; // Set interval duration in seconds (50 ms)
        static float lastMovementTime = 0.0f;  // Store the last time movement was triggered

        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        xoffset = xpos - lastX;
        yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;
        if (!cursorEnabled)
            scene.camera.ProcessMouseMovement(xoffset, yoffset, true);
    }

    void toggleCursor(GLFWwindow* window, bool enableCursor) {
        if (enableCursor) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);  // Show and unlock the cursor
            cursorEnabled = true;
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // Hide and lock the cursor
            cursorEnabled = false;
        }
    }

    void windowInit() {
        if (!glfwInit()) {
            // Initialization failed
            return;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Garbage Engine", nullptr, nullptr);
        if (!window) {
            // Window or OpenGL context creation failed
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glfwSetCursorPosCallback(window, mouse_callback);
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            // GLAD initialization failed
            glfwDestroyWindow(window);
            glfwTerminate();
            return;
        }

        //std::vector<std::pair<bool, std::string>> requiredGLEWExtensions = {
        //{ GLAD_GL_ARB_shader_image_load_store,		"ARB_shader_image_load_store"},
        //{ GLAD_GL_VERSION_4_5,						"GLAD_GL_VERSION_4_5 (OpenGL 4.5)"},
        //{ GLAD_GL_ARB_multisample,					"GLFW MSAA" }
        //};
        //
        //for (const auto& ext : requiredGLEWExtensions) {
        //    if (!ext.first) {
        //        std::cerr << "ERROR: " << ext.second << " not supported! Expect unexpected behaviour." << std::endl;
        //        std::cerr << "Press any key to continue ... " << std::endl;
        //        //getchar();
        //    }
        //}
        //std::cout << "Using OpenGL version " << glGetString(GL_VERSION) << std::endl;

        // Ensure we can capture the escape key being pressed below
        glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        // Enable depth test
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS);

        // Cull triangles which normal is not towards the camera
        glEnable(GL_CULL_FACE);
    }

    void processInput(Camera& camera) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.ProcessKeyboard(RIGHT, deltaTime);
    }

	void runApp() {
		while (!glfwWindowShouldClose(window)) {

            float currentFrame = glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;		

            if (isMouseButtonPressedDown(window, GLFW_MOUSE_BUTTON_2)) {
                toggleCursor(window, false);
            }
            else toggleCursor(window, true);

            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            processInput(scene.camera);
			scene.windowSize = glm::vec2(WIDTH, HEIGHT);
            if (scene.useBuiltInShaders) {
				scene.renderScene();
			}
			else {
				scene.render();
            }

            if (isKeyPressedDown(window, GLFW_KEY_DOWN) && scene.entitySelector.selectedEntity != nullptr) {
                scene.entitySelector.selectedEntity->transform.translate(glm::vec3(0.0f, -0.1f, 0.0f));
            }
            if (isKeyPressedDown(window, GLFW_KEY_UP) && scene.entitySelector.selectedEntity != nullptr) {
                scene.entitySelector.selectedEntity->transform.translate(glm::vec3(0.0f, 0.1f, 0.0f));
            }
			if (isKeyPressedDown(window, GLFW_KEY_LEFT) && scene.entitySelector.selectedEntity != nullptr) {
                scene.entitySelector.selectedEntity->transform.translate(glm::vec3(-0.1f, 0.0f, 0.0f));
			}
			if (isKeyPressedDown(window, GLFW_KEY_RIGHT) && scene.entitySelector.selectedEntity != nullptr) {
                scene.entitySelector.selectedEntity->transform.translate(glm::vec3(0.1f, 0.0f, 0.0f));
			}

			if (!isKeyPressedDown(window, GLFW_KEY_LEFT_SHIFT) && isMouseButtonPressed(window, GLFW_MOUSE_BUTTON_1)) {
				auto ray = shootRayFromMouse(scene.camera, scene.windowSize, getCursorPosition(window));
				scene.entitySelector.selectEntity(ray.origin, ray.direction, scene.entities);
			}

			if (scene.entitySelector.selectedEntity != nullptr && isKeyPressed(window, GLFW_KEY_G)) {
                if (scene.entitySelector.selectedEntity->hasComponent<AnimationMesh>()) {
                    if (scene.entitySelector.selectedEntity->getComponent<AnimationMesh>()->debugMode == true && scene.entitySelector.selectedEntity->getComponent<AnimationMesh>()->drawGizmos == true) {
						scene.entitySelector.selectedEntity->getComponent<AnimationMesh>()->debugMode = false;
                        scene.entitySelector.selectedEntity->getComponent<AnimationMesh>()->drawGizmos = false;
					}
                    else {
                        scene.entitySelector.selectedEntity->getComponent<AnimationMesh>()->debugMode = true;
                        scene.entitySelector.selectedEntity->getComponent<AnimationMesh>()->drawGizmos = true;
                    }
                }
			}

            if (isKeyPressedDown(window, GLFW_KEY_LEFT_SHIFT) && isMouseButtonPressed(window, GLFW_MOUSE_BUTTON_1) && scene.entitySelector.selectedEntity != nullptr) {
                if (scene.entitySelector.selectedEntity->hasComponent<AnimationMesh>()) {
                    boneEditor.mesh = scene.entitySelector.selectedEntity->getComponent<AnimationMesh>().get();
                    std::cout << "trying to select! \n";
                    auto ray = shootRayFromMouse(scene.camera, scene.windowSize, getCursorPosition(window));
                    boneEditor.selectBone(ray, scene.entitySelector.selectedEntity->transform.model);
                    boneEditor.selectVertex(ray);
                    boneEditor.printSelectedVertices();
                }
            }

            if(scene.entitySelector.selectedEntity != nullptr){
                if (scene.entitySelector.selectedEntity->hasComponent<AnimationMesh>()) {
                    
                }
            }
            glfwPollEvents();
			glfwSwapBuffers(window);
		}

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

bool Engine::cursorEnabled = true;
Scene Engine::scene{};
}

#endif // !CALCIUM_ENGINE
