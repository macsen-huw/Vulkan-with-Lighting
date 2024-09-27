#version 450

//Set pi
#define PI 3.141592653589;

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 oNormal;
layout (location = 2) in vec3 fragPos;
layout (location = 3) in mat3 tbn;

layout (set = 0, binding = 0, std140) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;

	vec3 cameraPos;
}	uScene;

layout(set = 1, binding = 0) uniform sampler2D uTexColor;
layout(set = 1, binding = 1) uniform sampler2D uMetalness;
layout(set = 1, binding = 2) uniform sampler2D uRoughness;
layout(set = 1, binding = 3) uniform sampler2D uNormal;

layout( push_constant ) uniform PushConstants {
	int normalMapEnabled;
	float lightPosX, lightPosY, lightPosZ;
	float lightColX, lightColY, lightColZ;

} pushConstants;

layout(location = 0) out vec4 oColor;

void main()
{
	float pi = PI;

	vec3 lPos = { -0.2972, 7.3100, -11.9532 };
	vec3 lCol = { 1.f, 1.f, 1.f };

	vec3 lightPosition = {pushConstants.lightPosX, pushConstants.lightPosY, pushConstants.lightPosZ}; 
	vec3 lightColour = {pushConstants.lightColX, pushConstants.lightColY, pushConstants.lightColZ}; 

	//Get all the parameters needed for light calculation
	vec4 materialColour = vec4(texture(uTexColor, v2fTexCoord).rgb, 1.f);

	//Get roughness and metalness from the respective maps
	float roughness = texture(uRoughness, v2fTexCoord).r;
	float metalness = texture(uMetalness, v2fTexCoord).r;

	vec3 mappedNormals = texture(uNormal, v2fTexCoord).rgb;

	//Transform to global space using the tbn matrix
	vec3 transformedNormals = normalize(tbn * mappedNormals);

	vec3 normal = (pushConstants.normalMapEnabled * transformedNormals) + (int(!(bool(pushConstants.normalMapEnabled))) * oNormal);

	//Follow the screenshots
	float globalAmbient = 0.02;

	//Beckmann roughness is equivalent to texture roughness squared
	float beckmannRoughness = pow(roughness, 2);

	/* 
	Get the fragment position to calculate light and view directions
	The object space is the same as the world space, so no transformations needed
	Calculate the directions in world space (the chosen shading space) 
	*/

	vec3 camera = uScene.camera[3].xyz;
	vec3 lightDirection = normalize(lightPosition - fragPos);
	vec3 viewDirection = normalize(uScene.cameraPos - fragPos);

	//Get half vector from lightDirection and viewDirection
	vec3 halfVector = normalize(lightDirection + viewDirection);

	//Calculate L_ambient
	vec4 L_ambient = globalAmbient * materialColour;

	//Calculate dot products that'll be reused in different functions
	//Some of them are clamped, others aren't (this is intentional)
	float nDotH = max(0, dot(normal, halfVector));
	float nDotV = max(0, dot(normal, viewDirection));
	float nDotL = max(0, dot(normal, lightDirection));
	float vDotH = dot(viewDirection, halfVector);

	//Calculate masking term using Cook-Torrence model
	float innerBracket1 = 2 * (nDotH * nDotV / vDotH);
	float innerBracket2 = 2 * (nDotH * nDotL / vDotH);

	float G = min(1, min(innerBracket1, innerBracket2));

	//Calculate normal distribution function D
	float eNumerator = pow(nDotH, 2) - 1;
	float eDenominator = pow(beckmannRoughness, 2) * pow(nDotH, 2);

	float dNumerator = exp(eNumerator / eDenominator);
	float dDenominator = pi * pow(beckmannRoughness, 2) * pow(nDotH, 4);

	float D = dNumerator / dDenominator;

	//Calculate specular reflection (using Fresnel term F)
	//First of all, calculate F_0
	vec3 F0 = ((1 - metalness) * vec3(0.04, 0.04, 0.04)) + (metalness * materialColour.rgb);

	//Then, calculate F using Schlick approximation
	vec3 F = F0 + ((1 - F0) * pow(1 - vDotH, 5));

	//Use F to calculate the diffuse light
	//L_diffuse consists of a tensor product of 2 sides
	vec3 lDiffuseLeft = materialColour.rgb / pi;
	vec3 lDiffuseRight = (vec3(1,1,1) - F) * (1 - metalness);
	vec3 L_diffuse = vec3(lDiffuseLeft.x * lDiffuseRight.x, lDiffuseLeft.y * lDiffuseRight.y, lDiffuseLeft.z * lDiffuseRight.z);

	vec3 DFG = D * F * G;
	//Finally, we have everything we need for the BRDF microfacet model
	vec3 BRDF = L_diffuse + (DFG / (4 * nDotV * nDotL)); 

	vec3 finalLightColour = L_ambient.rgb + (BRDF * lightColour * nDotL);
	//oColor = vec4(light.lightColour, 1.f);
	//oColor = vec4(lightDirection, 1.f);
	//oColor = vec4(viewDirection, 1.f);
	//oColor = materialColour;
	//oColor = vec4(abs(oNormal), 1.f);
	//oColor = vec4(abs(transformedNormals), 1.f);
	//oColor = vec4(L_diffuse, 1.f);
	//oColor = vec4(lightDirection, 1.f);
	//oColor = vec4(G,G,G, 1.f);
	//oColor = vec4(D,D,D, 1.f);
	//oColor = vec4(F, 1.f);
	//oColor = vec4(metalness, metalness, metalness, 1.f);
	//oColor = vec4(innerBracket1, innerBracket1, innerBracket1, 1.f);
	//oColor = vec4(DFG, 1.f);
	//oColor = vec4(fragPos, 1.f);
	//oColor = vec4(innerBracket1, innerBracket1, innerBracket1, 1.f);

	oColor = vec4(finalLightColour, 1.f);
}
