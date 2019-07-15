__kernel void convolution(
	__global const float* input, 
	__global const float* weight, 
	__global float* output, 
	int i_offset, 
	int w_offset, 
	int o_offset,
	int i_size)
{
	int idx = get_global_id(0);

	int tmp = idx/13;
	int i_idx = idx + 2*tmp;

	float sum1 = *(input + i_offset + i_idx) * *(weight + w_offset);
	float sum2 = *(input + i_offset + i_idx + 1) * *(weight + w_offset + 1);
	float sum3 = *(input + i_offset + i_idx + 2) * *(weight + w_offset + 2);
	float sum4 = *(input + i_offset + i_idx + i_size) * *(weight + w_offset + 3);
	float sum5 = *(input + i_offset + i_idx + i_size + 1) * *(weight + w_offset + 4);
	float sum6 = *(input + i_offset + i_idx + i_size + 2) * *(weight + w_offset + 5);
	float sum7 = *(input + i_offset + i_idx + i_size*2) * *(weight + w_offset + 6);
	float sum8 = *(input + i_offset + i_idx + i_size*2 + 1) * *(weight + w_offset + 7);
	float sum9 = *(input + i_offset + i_idx + i_size*2 + 2) * *(weight + w_offset + 8);

	float result = (sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9);

	sum1 = *(input + i_size*i_size + i_offset + i_idx) * *(weight + w_offset + 3*3);
	sum2 = *(input + i_size*i_size + i_offset + i_idx + 1) * *(weight + w_offset + 1 + 3*3);
	sum3 = *(input + i_size*i_size + i_offset + i_idx + 2) * *(weight + w_offset + 2 + 3*3);
	sum4 = *(input + i_size*i_size + i_offset + i_idx + i_size) * *(weight + w_offset + 3 + 3*3);
	sum5 = *(input + i_size*i_size + i_offset + i_idx + i_size + 1) * *(weight + w_offset + 4 + 3*3);
	sum6 = *(input + i_size*i_size + i_offset + i_idx + i_size + 2) * *(weight + w_offset + 5 + 3*3);
    sum7 = *(input + i_size*i_size + i_offset + i_idx + i_size*2) * *(weight + w_offset + 6 + 3*3);
	sum8 = *(input + i_size*i_size + i_offset + i_idx + i_size*2 + 1) * *(weight + w_offset + 7 + 3*3);
	sum9 = *(input + i_size*i_size + i_offset + i_idx + i_size*2 + 2) * *(weight + w_offset + 8 + 3*3);

	result += (sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9);

	sum1 = *(input + 2*i_size*i_size + i_offset + i_idx) * *(weight + w_offset + 18);
	sum2 = *(input + 2*i_size*i_size + i_offset + i_idx + 1) * *(weight + w_offset + 1 + 18);
	sum3 = *(input + 2*i_size*i_size + i_offset + i_idx + 2) * *(weight + w_offset + 2 + 18);
	sum4 = *(input + 2*i_size*i_size + i_offset + i_idx + i_size) * *(weight + w_offset + 3 + 18);
	sum5 = *(input + 2*i_size*i_size + i_offset + i_idx + i_size + 1) * *(weight + w_offset + 4 + 18);
	sum6 = *(input + 2*i_size*i_size + i_offset + i_idx + i_size + 2) * *(weight + w_offset + 5 + 18);
	sum7 = *(input + 2*i_size*i_size + i_offset + i_idx + i_size*2) * *(weight + w_offset + 6 + 18);
	sum8 = *(input + 2*i_size*i_size + i_offset + i_idx + i_size*2 + 1) * *(weight + w_offset + 7 + 18);
	sum9 = *(input + 2*i_size*i_size + i_offset + i_idx + i_size*2 + 2) * *(weight + w_offset + 8 + 18);

	result += (sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9);

	sum1 = *(input + 3*i_size*i_size + i_offset + i_idx) * *(weight + w_offset + 27);
	sum2 = *(input + 3*i_size*i_size + i_offset + i_idx + 1) * *(weight + w_offset + 1 + 27);
	sum3 = *(input + 3*i_size*i_size + i_offset + i_idx + 2) * *(weight + w_offset + 2 + 27);
	sum4 = *(input + 3*i_size*i_size + i_offset + i_idx + i_size) * *(weight + w_offset + 3 + 27);
	sum5 = *(input + 3*i_size*i_size + i_offset + i_idx + i_size + 1) * *(weight + w_offset + 4 + 27);
	sum6 = *(input + 3*i_size*i_size + i_offset + i_idx + i_size + 2) * *(weight + w_offset + 5 + 27);
	sum7 = *(input + 3*i_size*i_size + i_offset + i_idx + i_size*2) * *(weight + w_offset + 6 + 27);
	sum8 = *(input + 3*i_size*i_size + i_offset + i_idx + i_size*2 + 1) * *(weight + w_offset + 7 + 27);
	sum9 = *(input + 3*i_size*i_size + i_offset + i_idx + i_size*2 + 2) * *(weight + w_offset + 8 + 27);

	result += (sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9);

	output[o_offset + idx] += result;
}