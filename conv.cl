__kernel void convolution(
	__global const float* input, __global const float* weight, __global float* output, int i_base_addr, int w_base_addr, int o_base_addr)
{
	int idx = get_global_id(0);
	output[idx] = input[idx] * weight[idx];
}