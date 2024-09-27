# Vulkan with Lighting
## Introduction
This application uses the Vulkan graphics API to render the famous Sponza scene

## Compilation
This application was created using C++ and Vulkan. Specifically, the Volk loader was used, which can be found [here](https://github.com/zeux/volk)

Execute the premake file  

    .\premake5.exe vs2022

Then, open the solution file (`vulkanLighting.sln`)

First of all, set `bake` to be the startup project (as seen in the following screenshot) and run in the `release` configuration.

![Set Startup Project](assets-src/set%20startup%20project.jpg)

After the bake is completed, set `vulkanLighting` as the startup project and run in the release configuration.

## Controls
W - Move Camera Forward
S - Move Camera Backward
A - Move Camera to the Left
D - Move Camera to the Right
E - Move Camera Upwards
Q - Move Camera Downward
Right Click - Toggle Mouse

## Interface
The interface allows you to change the following settings:
- Enable Alpha Masking
- Enable Normal Mapping
- Change Light Position and Colour


![Image](assets-src/lighting%20vulkan.jpg)
