__kernel void convolution(
	__global const float* input, 
	__global const float* weight, 
	__global float* partsum, 
	__global float* output, 
	int i_offset, 
	int w_offset, 
	int o_offset
	)
{
	int global_id = get_global_id(0);
	int o_id = global_id / 9;

	int o_channel_id = o_id / 169;
	int ohow_id = o_id % 169;
	int ihiw_id = ohow_id + (ohow_id/13)*2;
	int w_id = global_id % 9;

	float sum = 0.0;
	int i = 0;
	for(; i < 192; i++)
	{
		float img = *(input + i_offset + i*225 + ihiw_id + (w_id/3)*15 + (w_id%3));
		float k = *(weight + w_offset + o_channel_id*1728 + i*9 + w_id);
		sum += (img*k);
	}
	*(partsum + global_id) = sum;

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(w_id == 0)
	{
		sum = *(partsum + o_id*9) + *(partsum + o_id*9 + 1) + *(partsum + o_id*9 + 2) + 
		      *(partsum + o_id*9 + 3) + *(partsum + o_id*9 + 4) + *(partsum + o_id*9 + 5) + 
			  *(partsum + o_id*9 + 6) + *(partsum + o_id*9 + 7) + *(partsum + o_id*9 + 8);

		*(output + o_offset + o_channel_id*169 + ohow_id) += sum;
	}
}