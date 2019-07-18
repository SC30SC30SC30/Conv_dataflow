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
	int local_id = global_id % 96;
	int o_id = global_id / 96;

	int o_channel_id = 	o_id / 169;
	int ohow_id = o_id % 169;
	int i_channel_id = local_id / 3;
	int ihiw_id = ohow_id + (ohow_id/13)*2;
	int wh_id = local_id % 3;

	float sum1 = *(input + i_offset + i_channel_id*225 + ihiw_id + wh_id*15) * *(weight + w_offset + o_channel_id*1728 + i_channel_id*9 + wh_id*3);
	float sum2 = *(input + i_offset + i_channel_id*225 + ihiw_id + wh_id*15 + 1) * *(weight + w_offset + o_channel_id*1728 + i_channel_id*9 + wh_id*3 + 1);
	float sum3 = *(input + i_offset + i_channel_id*225 + ihiw_id + wh_id*15 + 2) * *(weight + w_offset + o_channel_id*1728 + i_channel_id*9 + wh_id*3 + 2);

	*(partsum + global_id) = (sum1 + sum2 + sum3);

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(local_id == 0)
	{
		float sum = 0.0;
		int i = 0;
		for(; i < 96; i++)
		{
			sum += *(partsum + o_id*96 + i);
		}
		*(output + o_offset + o_channel_id*169 + ohow_id) += sum;
	}
}