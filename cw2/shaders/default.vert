#version 450

layout (location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;
layout(location = 2) in vec3 iNormal;
layout(location = 3) in vec4 iTangent;

layout (set = 0, binding = 0, std140) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;

	vec3 cameraPos;

}	uScene;

layout(location = 0) out vec2 v2fTexCoord;
layout(location = 1) out vec3 oNormal;
layout(location = 2) out vec3 fragPos;
layout(location = 3) out mat3 tbn;


void main()
{
	v2fTexCoord = iTexCoord;
	oNormal = iNormal;
	gl_Position = uScene.projCam * vec4(iPosition, 1.f);
	
	//Pass the position to the fragment
	fragPos = iPosition;

	//Create TBN matrix
	//First calculate bitangent
	vec3 bitangent = iTangent.w * cross(iNormal, iTangent.xyz);

	//Create TBN matrix
	mat3 tbnMatrix = mat3(iTangent.xyz, bitangent, iNormal);

	tbn = tbnMatrix;

}
