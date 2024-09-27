#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <volk/volk.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{
#		define SHADERDIR_ "assets/src/shaders/"

		constexpr char const* kVertexShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kTextureFragShaderPath = SHADERDIR_ "default.frag.spv";
		constexpr char const* kAlphaMaskFragShaderPath = SHADERDIR_ "alphaMasked.frag.spv";


#		undef SHADERDIR_

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		//More camera settings, useful for debug
		constexpr float kCameraBaseSpeed = 0.01f; //Units/second
		constexpr float kCameraFastMult = 2.f; //Speed multiplier
		constexpr float kCameraSlowMult = 0.05f; //Speed multiplier

		constexpr float kCameraMouseSensitivity = 0.01f; //Radians per pixel
	}

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	// Local types/structures:
	// Uniform data
	namespace glsl
	{
		struct SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;

			glm::vec3 cameraPos;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");
	}

	// Helpers:
	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();

	};

	//Holds all information needed for meshes
	struct MeshDetails
	{
		//Details to pass to the shader
		lut::Buffer positions;
		lut::Buffer texCoords;
		lut::Buffer normals;
		lut::Buffer tangents;

		//Index buffer storing indices
		lut::Buffer indices;

		//Store which material belongs to it
		int materialIndex = 0;

		//Store number of indices in mesh (needed for drawing)
		size_t indexCount = 0;
	};

	struct PushConstants
	{
		int isNormalMapping;
		float lightPosX, lightPosY, lightPosZ;
		float lightColX, lightColY, lightColZ;
	};

	// Local functions:
	void update_user_state(UserState&, float aElapsedTime);

	//Create render pass
	lut::RenderPass create_render_pass(lut::VulkanWindow const&);
	
	//Create mesh
	MeshDetails create_mesh(lut::VulkanContext const& aContext, lut::Allocator const& aAllocator, glm::vec3 aPositions[], glm::vec2 aTexCoords[],
							glm::vec3 aNormals[], std::uint32_t aIndices[], size_t aVertexCount, size_t aIndexCount, int aMaterialIndex, glm::vec4 aTangents[]);

	//Create descriptor sets
	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow);
	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow);

	//Create pipeline layout
	lut::PipelineLayout create_default_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout);

	//Create pipeline
	lut::Pipeline create_default_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout, const char*, const char*, bool);


	//Create depth buffer
	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	//Create swapchain framebuffer
	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	//Update scene uniforms
	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const&
	);

	//Submit commands
	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);

	//Present results
	void present_results(
		VkQueue,
		VkSwapchainKHR,
		std::uint32_t aImageIndex,
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);

	//ImGui Functions
	void init_imgui(lut::VulkanWindow&, VkDescriptorPool&, VkRenderPass&);
	void destroy_imgui();

	lut::RenderPass create_imgui_render_pass(lut::VulkanWindow const&);

	void create_imgui_framebuffers(lut::VulkanWindow const&, VkRenderPass, std::vector<lut::Framebuffer>&);
}

int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	// Configure the GLFW window
	UserState state{};

	glfwSetWindowUserPointer(window.window, &state);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass(window);

	//Create descriptor set layout (Needed for the pipeline layout)
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);

	lut::DescriptorSetLayout materialLayout = create_material_descriptor_layout(window);

	//Create pipeline layout
	lut::PipelineLayout pipeLayout = create_default_pipeline_layout(window, sceneLayout.handle, materialLayout.handle);

	//Create pipeline
	lut::Pipeline pipe = create_default_pipeline(window, renderPass.handle, pipeLayout.handle, cfg::kVertexShaderPath, cfg::kTextureFragShaderPath, false);
	lut::Pipeline alphaPipe = create_default_pipeline(window, renderPass.handle, pipeLayout.handle, cfg::kVertexShaderPath, cfg::kAlphaMaskFragShaderPath, true);
	//Create depth buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);
	
	//Create swapchain framebuffer
	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	//Create command pool
	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	//Allocate buffers and fences
	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); i++)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	//Create semaphores
	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	//Create scene buffer
	lut::Buffer sceneUBO = lut::create_buffer(allocator, sizeof(glsl::SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

	//Load model
	BakedModel model = load_baked_model("assets/src/suntemple.comp5822mesh");
	
	//Make a list of all materials with alpha masks (i.e. those with valid alphaMaskTextureIDs)
	std::vector<uint32_t> alphaTextures;
	
	for (size_t i = 0; i < model.materials.size(); i++)
	{
		if (model.materials[i].alphaMaskTextureId != 0xffffffff)
			alphaTextures.push_back(int(i));
	}

	//Process the model
	std::vector<MeshDetails> meshes;
	std::vector<MeshDetails> alphaMaskedMeshes;

	std::vector<MeshDetails> notAlphaMaskedMeshes;

	for (size_t i = 0; i < model.meshes.size(); i++)
	{
		bool hasAlphaMask = false;

		for (uint32_t& index : alphaTextures)
		{
			if (model.meshes[i].materialId == index)
			{
				hasAlphaMask = true;
			}
		}

		if (hasAlphaMask)
		{
			alphaMaskedMeshes.emplace_back(create_mesh(window, allocator, model.meshes.at(i).positions.data(),
				model.meshes.at(i).texcoords.data(), model.meshes.at(i).normals.data(),
				model.meshes.at(i).indices.data(), model.meshes.at(i).positions.size(),
				model.meshes.at(i).indices.size(), model.meshes.at(i).materialId, model.meshes.at(i).tangents.data()));
		}

		else
		{
			//Create mesh and store in vector
			meshes.emplace_back(create_mesh(window, allocator, model.meshes.at(i).positions.data(),
				model.meshes.at(i).texcoords.data(), model.meshes.at(i).normals.data(),
				model.meshes.at(i).indices.data(), model.meshes.at(i).positions.size(),
				model.meshes.at(i).indices.size(), model.meshes.at(i).materialId, model.meshes.at(i).tangents.data()));

		}

		notAlphaMaskedMeshes.emplace_back(create_mesh(window, allocator, model.meshes.at(i).positions.data(),
			model.meshes.at(i).texcoords.data(), model.meshes.at(i).normals.data(),
			model.meshes.at(i).indices.data(), model.meshes.at(i).positions.size(),
			model.meshes.at(i).indices.size(), model.meshes.at(i).materialId, model.meshes.at(i).tangents.data()));
			}

	//Load every texture in the model, and create image views for each
	//This includes base colour, metallic, roughness and normal maps
	std::vector<lut::Image> images(model.textures.size());
	std::vector<lut::ImageView> imageViews(images.size());

	for (size_t i = 0; i < model.textures.size(); i+=4)
	{
		
		images[i] = lut::load_image_texture2d(model.textures[i].path.c_str(), window, cpool.handle, allocator, VK_FORMAT_R8G8B8A8_SRGB);
		imageViews[i] = lut::create_image_view_texture2d(window, images[i].image, VK_FORMAT_R8G8B8A8_SRGB);

		images[i+1] = lut::load_image_texture2d(model.textures[i+1].path.c_str(), window, cpool.handle, allocator, VK_FORMAT_R8G8B8A8_UNORM);
		imageViews[i+1] = lut::create_image_view_texture2d(window, images[i+1].image, VK_FORMAT_R8G8B8A8_UNORM);

		images[i+2] = lut::load_image_texture2d(model.textures[i+2].path.c_str(), window, cpool.handle, allocator, VK_FORMAT_R8G8B8A8_UNORM);
		imageViews[i+2] = lut::create_image_view_texture2d(window, images[i+2].image, VK_FORMAT_R8G8B8A8_UNORM);

		images[i+3] = lut::load_image_texture2d(model.textures[i+3].path.c_str(), window, cpool.handle, allocator, VK_FORMAT_R8G8B8A8_UNORM);
		imageViews[i+3] = lut::create_image_view_texture2d(window, images[i+3].image, VK_FORMAT_R8G8B8A8_UNORM);

	}

	//Create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);
	
	//Create texture sampler
	lut::Sampler defaultSampler = lut::create_default_sampler(window);

	//Create descriptor set for the scene
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//Create a descriptor set for every material
	std::vector<VkDescriptorSet> meshDescriptorSets(model.materials.size());

	for (size_t i = 0; i < meshDescriptorSets.size(); i++)
	{
		//Allocate the descriptor set
		meshDescriptorSets[i] = lut::alloc_desc_set(window, dpool.handle, materialLayout.handle);

		VkDescriptorImageInfo imageInfo[4]{};

		//Base Colour
		imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[0].imageView = imageViews.at(model.materials[i].baseColorTextureId).handle;
		imageInfo[0].sampler = defaultSampler.handle;

		//Metalness
		imageInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[1].imageView = imageViews.at(model.materials[i].metalnessTextureId).handle;
		imageInfo[1].sampler = defaultSampler.handle;

		//Roughness
		imageInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[2].imageView = imageViews.at(model.materials[i].roughnessTextureId).handle;
		imageInfo[2].sampler = defaultSampler.handle;

		//Normal map
		imageInfo[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo[3].imageView = imageViews.at(model.materials[i].normalMapTextureId).handle;
		imageInfo[3].sampler = defaultSampler.handle;

		//Update the descriptor set
		VkWriteDescriptorSet desc[4]{};

		for (int j = 0; j < 4; j++)
		{
			desc[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[j].dstSet = meshDescriptorSets[i];
			desc[j].dstBinding = j;
			desc[j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[j].descriptorCount = 1;
			desc[j].pImageInfo = &imageInfo[j];
		}

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);

		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//Create buffer to store the light details
	/*Light light;
	light.lightPosition = { -0.2972, 7.3100, -11.9532 };
	light.lightColour = { 1.f, 1.f, 1.f };*/
	
	
	//Setup imgui
	lut::RenderPass imguiRenderPass = create_imgui_render_pass(window);

	lut::Semaphore imguiSemaphore = lut::create_semaphore(window);

	std::vector<lut::Framebuffer> imguiFramebuffers;
	create_imgui_framebuffers(window, imguiRenderPass.handle, imguiFramebuffers);

	std::vector<VkCommandBuffer> imguicbuffers;
	std::vector<lut::Fence> imguicbfences;

	for (std::size_t i = 0; i < imguiFramebuffers.size(); ++i)
	{
		imguicbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		imguicbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	init_imgui(window, dpool.handle, imguiRenderPass.handle);

	PushConstants pushConstants = { 0, //normal mapping
								 -0.2972, 7.3100, -11.9532, //light position
									1.f, 1.f, 1.f, //light colour
								};


	bool alphaMasking = false;
	bool normalMappingEnabled = false;

	float* lightPosition[3] = { &pushConstants.lightPosX, &pushConstants.lightPosY, &pushConstants.lightPosZ };
	float* lightColour[3] = { &pushConstants.lightColX, &pushConstants.lightColY, &pushConstants.lightColZ };
	//RENDERING LOOP
	// Application main loop
	bool recreateSwapchain = false;

	//Record time before main loop starts
	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()
	
	
		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//Wait for the GPU to finish processing
			vkDeviceWaitIdle(window.device);

			//Recreate them
			auto const changes = lut::recreate_swapchain(window);

			if (changes.changedFormat)
			{
				renderPass = create_render_pass(window);
				imguiRenderPass = create_imgui_render_pass(window);

			}
				
			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				pipe = create_default_pipeline(window, renderPass.handle, pipeLayout.handle, cfg::kVertexShaderPath, cfg::kTextureFragShaderPath, false);
				alphaPipe = create_default_pipeline(window, renderPass.handle, pipeLayout.handle, cfg::kVertexShaderPath, cfg::kAlphaMaskFragShaderPath, true);
			}
				
			framebuffers.clear();
			imguiFramebuffers.clear();

			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);
			create_imgui_framebuffers(window, imguiRenderPass.handle, imguiFramebuffers);

			//Recreate semaphore
			imageAvailable = lut::create_semaphore(window);

			recreateSwapchain = false;
			continue;

		}
	
	
		//Acquire next swapchain image
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(window.device, window.swapchain, std::numeric_limits<std::uint64_t>::max(), imageAvailable.handle, VK_NULL_HANDLE, &imageIndex);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire next swapchain image" "vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}

		//Wait for command buffer to be available
		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n" "vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		//Record and submit commands
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());
	
		//Update state
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();

		update_user_state(state, dt);

		//Prepare data for this frame
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);
		
		//Record commands ------------------------------------------------------------------------------------------------------
		//Begin recording commands
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(cbuffers[imageIndex], &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Update unifom buffer
		lut::buffer_barrier(cbuffers[imageIndex], sceneUBO.buffer, VK_ACCESS_UNIFORM_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(cbuffers[imageIndex], sceneUBO.buffer, 0, sizeof(glsl::SceneUniform), &sceneUniforms);

		lut::buffer_barrier(cbuffers[imageIndex], sceneUBO.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		//Begin render pass
		//Clear to a dark gray background
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		clearValues[1].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = renderPass.handle;
		passInfo.framebuffer = framebuffers[imageIndex].handle;
		passInfo.renderArea.offset = VkOffset2D{ 0,0 };
		passInfo.renderArea.extent = VkExtent2D{ window.swapchainExtent.width, window.swapchainExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

	
		vkCmdBeginRenderPass(cbuffers[imageIndex], &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		//Bind the pipeline
		vkCmdBindPipeline(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.handle);

		//Bind the descriptors
		vkCmdBindDescriptorSets(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout.handle, 0, 1, &sceneDescriptors, 0, nullptr);
		
		//Pass PushConstants to shader
		normalMappingEnabled ? pushConstants.isNormalMapping = 1 : pushConstants.isNormalMapping = 0;

		vkCmdPushConstants(cbuffers[imageIndex], pipeLayout.handle, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pushConstants);

		if (alphaMasking)
		{
			for (uint32_t i = 0; i < meshes.size(); i++)
			{
				//Bind the material descriptor set
				vkCmdBindDescriptorSets(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout.handle, 1, 1, &meshDescriptorSets[meshes.at(i).materialIndex], 0, nullptr);

				VkBuffer meshBuffers[4] = { meshes.at(i).positions.buffer, meshes.at(i).texCoords.buffer, meshes.at(i).normals.buffer, meshes.at(i).tangents.buffer };
				VkDeviceSize meshOffsets[4] = {};
				//Bind vertex buffers
				vkCmdBindVertexBuffers(cbuffers[imageIndex], 0, 4, meshBuffers, meshOffsets);

				//Bind index buffer
				vkCmdBindIndexBuffer(cbuffers[imageIndex], meshes.at(i).indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(cbuffers[imageIndex], meshes.at(i).indexCount, 1, 0, 0, 0);
			}

			//Change to alpha masked pipeline
			vkCmdBindPipeline(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, alphaPipe.handle);

			//Pass push constants again
			vkCmdPushConstants(cbuffers[imageIndex], pipeLayout.handle, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pushConstants);

			for (size_t i = 0; i < alphaMaskedMeshes.size(); i++)
			{
				//Bind the material descriptor set
				vkCmdBindDescriptorSets(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout.handle, 1, 1, &meshDescriptorSets[alphaMaskedMeshes.at(i).materialIndex], 0, nullptr);

				VkBuffer alphaMeshBuffers[4] = { alphaMaskedMeshes.at(i).positions.buffer, alphaMaskedMeshes.at(i).texCoords.buffer, alphaMaskedMeshes.at(i).normals.buffer, alphaMaskedMeshes.at(i).tangents.buffer };
				VkDeviceSize alphaMeshOffsets[4] = {};

				//Bind vertex buffers
				vkCmdBindVertexBuffers(cbuffers[imageIndex], 0, 4, alphaMeshBuffers, alphaMeshOffsets);

				//Bind index buffer
				vkCmdBindIndexBuffer(cbuffers[imageIndex], alphaMaskedMeshes.at(i).indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(cbuffers[imageIndex], alphaMaskedMeshes.at(i).indexCount, 1, 0, 0, 0);
			}
		}

		else
		{
			for (uint32_t i = 0; i < notAlphaMaskedMeshes.size(); i++)
			{
				//Bind the material descriptor set
				vkCmdBindDescriptorSets(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout.handle, 1, 1, &meshDescriptorSets[notAlphaMaskedMeshes.at(i).materialIndex], 0, nullptr);

				VkBuffer meshBuffers[4] = { notAlphaMaskedMeshes.at(i).positions.buffer, notAlphaMaskedMeshes.at(i).texCoords.buffer, notAlphaMaskedMeshes.at(i).normals.buffer, notAlphaMaskedMeshes.at(i).tangents.buffer };
				VkDeviceSize meshOffsets[4] = {};
				//Bind vertex buffers
				vkCmdBindVertexBuffers(cbuffers[imageIndex], 0, 4, meshBuffers, meshOffsets);

				//Bind index buffer
				vkCmdBindIndexBuffer(cbuffers[imageIndex], notAlphaMaskedMeshes.at(i).indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(cbuffers[imageIndex], notAlphaMaskedMeshes.at(i).indexCount, 1, 0, 0, 0);
			}
		}
		
		

		//End the render pass
		vkCmdEndRenderPass(cbuffers[imageIndex]);

		//End command recording
		if (auto const res = vkEndCommandBuffer(cbuffers[imageIndex]); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Recording commands ended --------------------------------------------------------------------------------------------------------------
		
		 
		
		//Submit the recorded commands
		submit_commands(window, cbuffers[imageIndex], cbfences[imageIndex].handle, imageAvailable.handle, imguiSemaphore.handle);

		//Prepare for second pass with ImGui
		assert(std::size_t(imageIndex) < imguicbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &imguicbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n" "vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &imguicbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		//It is available, so begin recording
		begInfo = {};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(imguicbuffers[imageIndex], &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkRenderPassBeginInfo imguiPassInfo = passInfo;
		imguiPassInfo.framebuffer = imguiFramebuffers[imageIndex].handle;
		imguiPassInfo.renderPass = imguiRenderPass.handle;
		imguiPassInfo.clearValueCount = 1;
		imguiPassInfo.pClearValues = &clearValues[0];

		vkCmdBeginRenderPass(imguicbuffers[imageIndex], &imguiPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		//Setup new ImGui frame
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::Begin("ImGui Window");
		ImGui::Checkbox("Enable Alpha Masking", &alphaMasking);
		ImGui::Checkbox("Use Normal Mapping", &normalMappingEnabled);

		ImGui::Text("Camera Pos: (%f, %f, %f)", sceneUniforms.cameraPos.x, sceneUniforms.cameraPos.y, sceneUniforms.cameraPos.z);
		
		ImGui::DragFloat3("Light Position (XYZ)", *lightPosition, 0.1f, -20.0f, 20.0f, "%.2f");
		ImGui::ColorEdit3("Light Colour", *lightColour);

		ImGui::End();


		ImGui::Render();
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), imguicbuffers[imageIndex]);

		vkCmdEndRenderPass(imguicbuffers[imageIndex]);

		//End command recording
		if (auto const res = vkEndCommandBuffer(imguicbuffers[imageIndex]); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Submit commands
		submit_commands(window, imguicbuffers[imageIndex], imguicbfences[imageIndex].handle, imguiSemaphore.handle, renderFinished.handle);

		//Present the results
		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);

		//Ensure the command buffers have finished (throws an error if not)
		if (auto const res = vkWaitForFences(window.device, 1, &imguicbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n" "vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

	}

	destroy_imgui();

	vkDeviceWaitIdle(window.device);

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		//Handle camera controls
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;

		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;


		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			//Only update rotation on second frame of mouse navigation
			//Ensures previous X and Y variables are initialised to sensible values
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));

			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;

			aState.wasMousing = true;
		}

		else
		{
			aState.wasMousing = false;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));

		
			
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));
		
			

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		
			
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));
			

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));

		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));

			
	}
}


namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& aState)
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);

		aSceneUniforms.projection[1][1] *= -1.f; //Mirror the y axis
		aSceneUniforms.camera = glm::inverse(aState.camera2world);
		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
	
		glm::vec4 cameraPosition = { 0,0,0, 1.0 };

		aSceneUniforms.cameraPos = aState.camera2world * cameraPosition;

	}
}

namespace
{
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//Declare the single subpass
		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1; //refers to attachments[1]
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		//Introduce a dependency
		VkSubpassDependency deps[2]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[1].dstSubpass = 0;
		deps[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;


		//With declarations in place, we can now create the render pass
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = deps;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	MeshDetails create_mesh(lut::VulkanContext const& aContext, lut::Allocator const& aAllocator, glm::vec3 aPositions[], glm::vec2 aTexCoords[],
		glm::vec3 aNormals[], std::uint32_t aIndices[], size_t aVertexCount, size_t aIndexCount, int aMaterialIndex, glm::vec4 aTangents[])
	{
		//Set required sizes for each detail
		VkDeviceSize posSize = aVertexCount * sizeof(glm::vec3);
		VkDeviceSize texSize = aVertexCount * sizeof(glm::vec2);
		VkDeviceSize normSize = aVertexCount * sizeof(glm::vec3);
		VkDeviceSize indexSize = aIndexCount * sizeof(std::uint32_t);

		VkDeviceSize tangentSize = aVertexCount * sizeof(glm::vec4);


		//Create buffers for position, texcoord, normal
		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			posSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		lut::Buffer vertexTexGPU = lut::create_buffer(
			aAllocator,
			texSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		lut::Buffer vertexNormGPU = lut::create_buffer(
			aAllocator,
			normSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		//Create indexed buffer for indices
		lut::Buffer indexGPU = lut::create_buffer(
			aAllocator,
			indexSize,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);

		lut::Buffer tangentGPU = lut::create_buffer(
			aAllocator,
			tangentSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		//Create staging buffers
		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			posSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer texStaging = lut::create_buffer(
			aAllocator,
			texSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer normStaging = lut::create_buffer(
			aAllocator,
			normSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer indexStaging = lut::create_buffer(
			aAllocator,
			indexSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer tangentStaging = lut::create_buffer(
			aAllocator,
			tangentSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		//Map position memory
		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(posPtr, aPositions, posSize);
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		//Map texcoord memory
		void* texPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, texStaging.allocation, &texPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(texPtr, aTexCoords, texSize);
		vmaUnmapMemory(aAllocator.allocator, texStaging.allocation);

		//Map normal memory
		void* normPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, normStaging.allocation, &normPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(normPtr, aNormals, normSize);
		vmaUnmapMemory(aAllocator.allocator, normStaging.allocation);

		//Map index memory
		void* indexPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, indexStaging.allocation, &indexPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(indexPtr, aIndices, indexSize);
		vmaUnmapMemory(aAllocator.allocator, indexStaging.allocation);

		//Map tangent memory
		void* tangentPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, tangentStaging.allocation, &tangentPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(tangentPtr, aTangents, tangentSize);
		vmaUnmapMemory(aAllocator.allocator, tangentStaging.allocation);
		
		//Prepare for issuing the transfer commands that copy data from staging buffers to final on-GPU buffers
		//First, ensure that Vulkan resources are alive until all transfers are completed
		lut::Fence uploadComplete = lut::create_fence(aContext);

		//Queue data uploads from staging buffers to final buffers
		//Use a separate command pool for simplicity
		lut::CommandPool uploadPool = lut::create_command_pool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		//Record copy commands into command buffer
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Copy commands into buffer
		VkBufferCopy pcopy{};
		pcopy.size = posSize;

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy tcopy{};
		tcopy.size = texSize;

		vkCmdCopyBuffer(uploadCmd, texStaging.buffer, vertexTexGPU.buffer, 1, &tcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexTexGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ncopy{};
		ncopy.size = normSize;

		vkCmdCopyBuffer(uploadCmd, normStaging.buffer, vertexNormGPU.buffer, 1, &ncopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexNormGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy icopy{};
		icopy.size = indexSize;

		vkCmdCopyBuffer(uploadCmd, indexStaging.buffer, indexGPU.buffer, 1, &icopy);

		lut::buffer_barrier(
			uploadCmd,
			indexGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);


		VkBufferCopy tangentcopy{};
		tangentcopy.size = tangentSize;

		vkCmdCopyBuffer(uploadCmd, tangentStaging.buffer, tangentGPU.buffer, 1, &tangentcopy);

		lut::buffer_barrier(
			uploadCmd,
			tangentGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);
		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}

		//Wait for commands to finish before destroying temporary resources needed for transfers
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}

		MeshDetails ret;
		ret.positions = std::move(vertexPosGPU);
		ret.texCoords = std::move(vertexTexGPU);
		ret.normals = std::move(vertexNormGPU);
		ret.indices = std::move(indexGPU);
		ret.tangents = std::move(tangentGPU);
		ret.materialIndex = aMaterialIndex;
		ret.indexCount = aIndexCount;

		return ret;
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		//Set up bindings
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		
		//With bindings set, finish up the descriptor set layout properties
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		//Finally, create descriptor set layout
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);

	}

	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		//Set up the bindings
		VkDescriptorSetLayoutBinding bindings[4]{};

		//First binding - base colour
		bindings[0].binding = 0; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		//Second binding - metalness
		bindings[1].binding = 1; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		//Third binding - roughness
		bindings[2].binding = 2; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		//Fourth binding - normal map
		bindings[3].binding = 3; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[3].descriptorCount = 1;
		bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		//With bindings set, finish up the descriptor set layout properties
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		//Finally, create descriptor set layout
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}


	//Create "default" pipeline - i.e. the main pipeline that draws most objects (draws all initially)
	lut::PipelineLayout create_default_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aMaterialLayout)
	{
		VkDescriptorSetLayout layouts[] =
		{
			aSceneLayout,
			aMaterialLayout,
		};

		//Create push constants
		VkPushConstantRange pushConstantRange;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(PushConstants);

		//Finish up the pipeline layout properties
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		//Create the pipeline layout
		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create textured pipeline layout\n" "vkCreatePipelineLayout returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline create_default_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, const char* vertexPath, const char* fragPath, bool isAlpha)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, vertexPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, fragPath);

		//Define shader stages in the pipeline
		//2 stages, vertex then fragment stage
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		//Define vertex input attributes
		VkVertexInputBindingDescription vertexInputs[4]{};

		//First input - vertex position
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//Second input - texture coordinates
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//Third input - vertex normals
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//Fourth input - tangents
		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(float) * 4;
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//Describe the vertex input attributes
		VkVertexInputAttributeDescription vertexAttributes[4]{};

		//Vertex Positions
		vertexAttributes[0].binding = 0; //Must match binding above
		vertexAttributes[0].location = 0; //Must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		//Texture Coordinates
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		//Normals
		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;

		//Tangents
		vertexAttributes[3].binding = 3;
		vertexAttributes[3].location = 3;
		vertexAttributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		vertexAttributes[3].offset = 0;

		//Summarize the shader's input details
		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 4;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 4;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		//Next, define which primitive the input is assembled into for rasterization (spoiler alert - it's triangles)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Define viewport and scissor regions, then combine them together
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Define rasterisation options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		if(!isAlpha)
			rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Define multi sampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//Define blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Define depth stencil state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//Set up the pipeline properties
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;


		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		//We can now create the pipeline
		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}




	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] =
			{
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n" "vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n" "vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		//Create image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n" "vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}


	void submit_commands(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);
		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}

		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable to present swapchain image %u\n" "vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}
}

//Imgui functions
namespace
{
	void init_imgui(lut::VulkanWindow& aWindow, VkDescriptorPool& aDpool, VkRenderPass& aRenderPass)
	{
		IMGUI_CHECKVERSION();

		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForVulkan(aWindow.window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = aWindow.instance;
		init_info.PhysicalDevice = aWindow.physicalDevice;
		init_info.Device = aWindow.device;
		init_info.QueueFamily = aWindow.graphicsFamilyIndex;
		init_info.Queue = aWindow.graphicsQueue;
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = aDpool;
		init_info.Allocator = nullptr;
		init_info.RenderPass = aRenderPass;

		//Get image count
		std::uint32_t imageCount;
		vkGetSwapchainImagesKHR(aWindow.device, aWindow.swapchain, &imageCount, nullptr);

		init_info.MinImageCount = imageCount;
		init_info.ImageCount = imageCount;

		ImGui_ImplVulkan_Init(&init_info);

		//Load fonts
		ImGui_ImplVulkan_CreateFontsTexture();
	}

	void destroy_imgui()
	{
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	//Create a renderpass
	lut::RenderPass create_imgui_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colourAttachment = {};
		colourAttachment.attachment = 0;
		colourAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = &colourAttachment;

		//Introduce a dependency
		VkSubpassDependency deps = {};
		deps.srcSubpass = VK_SUBPASS_EXTERNAL;
		deps.srcAccessMask = 0;
		deps.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps.dstSubpass = 0;
		deps.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;



		//With declarations in place, we can now create the render pass
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 1;
		passInfo.pDependencies = &deps;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	//Create framebuffers (they differ from the swapchain framebuffer since they don't have depth)
	void create_imgui_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[] =
			{
				aWindow.swapViews[i]
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 1;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create imgui framebuffer for swap chain image %zu\n" "vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

}



//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
