__kernel void conv_unroll_all(
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
	int local_id_1 = global_id % 36;
	int local_id_2 = global_id / 36;

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
		for(; i < 36; i++)
		{
			sum += *(partsum+local_id_2*36+i);
		}

		*(output + o_offset + o_channel_id*169 + ohow_id) += sum;
	}
}

__kernel void conv_unroll_ohow(
	__global const float* input, 
	__global const float* weight, 
	__global float* partsum, 
	__global float* output, 
	int i_offset, 
	int w_offset, 
	int o_offset
	)
{
	int o_channel_id = get_group_id(0);
	int ohow_id = get_local_id(0);
	int ihiw_id = ohow_id + (ohow_id/13)*2;

	printf("%d. o_channel_id=%d\tohow_id=%d\tihiw_id=%d\n", get_global_id(0), o_channel_id, ohow_id, ihiw_id);

	float sum = 0.0;
	int ic = 0;
	for(; ic < 32; ic++)
	{
		sum += *(input + i_offset + ic*225 + ihiw_id + 0) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 0) +
		 *(input + i_offset + ic*225 + ihiw_id + 1) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 1) +
		 *(input + i_offset + ic*225 + ihiw_id + 2) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 2) +
		 *(input + i_offset + ic*225 + ihiw_id + 15) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 3) +
		 *(input + i_offset + ic*225 + ihiw_id + 16) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 4) +
		 *(input + i_offset + ic*225 + ihiw_id + 17) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 5) +
		 *(input + i_offset + ic*225 + ihiw_id + 30) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 6) +
		 *(input + i_offset + ic*225 + ihiw_id + 31) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 7) + 
		 *(input + i_offset + ic*225 + ihiw_id + 32) * *(weight + w_offset + o_channel_id*1728 + ic*9 + 8);
	}

	*(output + o_offset + o_channel_id*169 + ohow_id) += sum;
}