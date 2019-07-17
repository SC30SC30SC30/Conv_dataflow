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
	int local_id_1 = global_id % 216;
	int local_id_2 = global_id / 216;

	int o_channel_id = local_id_2 / 169;
	int ohow_id = local_id_2 % 169;
	int i_channel_id = local_id_1 / 9;
	int ihiw_id = ohow_id + (ohow_id/13)*2;
	int w_id = local_id_1 % 9;

	float img = *(input + i_offset + i_channel_id*225 + ihiw_id + (w_id/3)*15 + (w_id%3));
	float k = *(weight + w_offset + o_channel_id*1728 + i_channel_id*9 + w_id);
	*(partsum + global_id) = img * k;

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(local_id_1 == 0)
	{
		float sum = 0.0;
		int i = 0;
		for(; i < 216; i++)
		{
			sum += *(partsum+local_id_2*216+i);
		}

		*(output + o_offset + o_channel_id*169 + ohow_id) += sum;
	}
}