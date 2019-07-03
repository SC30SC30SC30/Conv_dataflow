__kernel void convolution(
	__global const float* input, __global const float* weight, __global float* output)
{
	int idx = get_global_id(0);
	output[idx] = input[idx] * weight[idx];
}