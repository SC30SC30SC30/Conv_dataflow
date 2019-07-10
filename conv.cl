__kernel void convolution(
	__global const float* input, 
	__global const float* weight, 
	__global float* partsum, 
	__global float* output, 
	int i_offset, 
	int w_offset, 
	int o_offset)
{
	int global_id = get_global_id(0);

	int channel_id = global_id / 1521;
	int ohow = global_id % 1521;
	int ohow_id = get_group_id(0);
	int ihiw_id = ohow_id + (ohow_id/13)*2;
	int w_id = ohow % 9;

	float img = *(input + i_offset + channel_id*225 + ihiw_id + (w_id/3)*15 + (w_id%3));
	float k = *(weight + w_offset + channel_id*9 + w_id);
	*(partsum + global_id) = img * k;

	barrier(CLK_GLOBAL_MEM_FENCE);

	int multiple = global_id / 9;
	if((global_id <= 1512) && (global_id % 9 == 0))
	{
		float sum = *(partsum+9*multiple)+*(partsum+9*multiple+1)+*(partsum+9*multiple+2)+*(partsum+9*multiple+3)+*(partsum+9*multiple+4)+*(partsum+9*multiple+5)+*(partsum+9*multiple+6)+*(partsum+9*multiple+7)+*(partsum+9*multiple+8)+
					*(partsum+9*multiple+1521)+*(partsum+9*multiple+1522)+*(partsum+9*multiple+1523)+*(partsum+9*multiple+1524)+*(partsum+9*multiple+1525)+*(partsum+9*multiple+1526)+*(partsum+9*multiple+1527)+*(partsum+9*multiple+1528)+*(partsum+9*multiple+1529)+
					*(partsum+9*multiple+3042)+*(partsum+9*multiple+3043)+*(partsum+9*multiple+3044)+*(partsum+9*multiple+3045)+*(partsum+9*multiple+3046)+*(partsum+9*multiple+3047)+*(partsum+9*multiple+3048)+*(partsum+9*multiple+3049)+*(partsum+9*multiple+3050)+
					*(partsum+9*multiple+4563)+*(partsum+9*multiple+4564)+*(partsum+9*multiple+4565)+*(partsum+9*multiple+4566)+*(partsum+9*multiple+4567)+*(partsum+9*multiple+4568)+*(partsum+9*multiple+4569)+*(partsum+9*multiple+4570)+*(partsum+9*multiple+4571);

		*(output + o_offset + ohow_id) = sum; 
	}
}