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
	int o_channel_id = get_group_id(0);
	int ohow_id = get_local_id(0);
	int ihiw_id = ohow_id + (ohow_id/13)*2;

	printf("%d. o_channel_id=%d\tohow_id=%d\tihiw_id=%d\n", get_global_id(0), o_channel_id, ohow_id, ihiw_id);

	float sum = 0.0;
	int ic = 0;
	for(; ic < 32; ic++)
	{
		float4 img1 = (float4)(*(input+i_offset+ic*225+ihiw_id+0), *(input+i_offset+ic*225+ihiw_id+1), *(input+i_offset+ic*225+ihiw_id+2), *(input+i_offset+ic*225+ihiw_id+15));
		float4 img2 = (float4)(*(input+i_offset+ic*225+ihiw_id+16), *(input+i_offset+ic*225+ihiw_id+17), *(input+i_offset+ic*225+ihiw_id+30), *(input+i_offset+ic*225+ihiw_id+31));
		float4 k1 = vload4(0, (weight + w_offset + o_channel_id*1728 + ic*9));
		float4 k2 = vload4(0, (weight + w_offset + o_channel_id*1728 + ic*9 + 4));
		float sum = dot(img1, k1) + dot(img2, k2) + (*(input+i_offset+ic*225+ihiw_id+32) * *(weight+w_offset+o_channel_id*1728+ic*9+8));
	}

	*(output + o_offset + o_channel_id*169 + ohow_id) += sum;
}