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
	int oc_id = get_group_id(0) / 169;
	int ohow_id = get_group_id(0) % 169;
	int ic_id = get_local_id(0);
	int ihiw_id = ohow_id + (ohow_id/13)*2;

	float sum1 = *(input + i_offset + ic_id*225 + ihiw_id) * *(weight + w_offset + oc_id*1728 + ic_id*9);
	float sum2 = *(input + i_offset + ic_id*225 + ihiw_id + 1) * *(weight + w_offset +  oc_id*1728 + ic_id*9 + 1);
	float sum3 = *(input + i_offset + ic_id*225 + ihiw_id + 2) * *(weight + w_offset + oc_id*1728 + ic_id*9 + 2);
	float sum4 = *(input + i_offset + ic_id*225 + ihiw_id + 15) * *(weight + w_offset + oc_id*1728 + ic_id*9 + 3);
	float sum5 = *(input + i_offset + ic_id*225 + ihiw_id + 16) * *(weight + w_offset + oc_id*1728 + ic_id*9 + 4);
	float sum6 = *(input + i_offset +ic_id*225 + ihiw_id + 17) * *(weight + w_offset + oc_id*1728 + ic_id*9 + 5);
	float sum7 = *(input + i_offset +ic_id*225 + ihiw_id + 30) * *(weight + w_offset + oc_id*1728 + ic_id*9 + 6);
	float sum8 = *(input + i_offset +ic_id*225 + ihiw_id + 31) * *(weight + w_offset + oc_id*1728 + ic_id*9 + 7);
	float sum9 = *(input + i_offset +ic_id*225 + ihiw_id + 32) * *(weight + w_offset + oc_id*1728 + ic_id*9 + 8);

	*(partsum + get_group_id(0)/4 + get_local_id(0)*9) = (sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9);

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(ic_id == 0)
	{
		int i = 0;
		for(; i < 1728; i++)
		{
			*(output + o_offset + oc_id*169 + ohow_id) += *(partsum + get_group_id(0)/4 + i);
		}
	}
}