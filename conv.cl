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
	int local_id = get_local_id(0);

	int channel_id = local_id / 9;
	int ohow_id = get_group_id(0);
	int ihiw_id = ohow_id + (ohow_id/13)*2;
	int w_id = local_id % 9;

	float img = *(input + i_offset + channel_id*225 + ihiw_id + (w_id/3)*15 + (w_id%3));
	float k = *(weight + w_offset + channel_id*9 + w_id);
	*(partsum + global_id) = img * k;

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(local_id == 0)
	{
		float sum = *(partsum+ohow_id*36)+*(partsum+ohow_id*36+1)+*(partsum+ohow_id*36+2)+*(partsum+ohow_id*36+3)+*(partsum+ohow_id*36+4)+*(partsum+ohow_id*36+5)+*(partsum+ohow_id*36+6)+*(partsum+ohow_id*36+7)+*(partsum+ohow_id*36+8)+
					*(partsum+ohow_id*36+9)+*(partsum+ohow_id*36+10)+*(partsum+ohow_id*36+11)+*(partsum+ohow_id*36+12)+*(partsum+ohow_id*36+13)+*(partsum+ohow_id*36+14)+*(partsum+ohow_id*36+15)+*(partsum+ohow_id*36+16)+*(partsum+ohow_id*36+17)+
					*(partsum+ohow_id*36+18)+*(partsum+ohow_id*36+19)+*(partsum+ohow_id*36+20)+*(partsum+ohow_id*36+21)+*(partsum+ohow_id*36+22)+*(partsum+ohow_id*36+23)+*(partsum+ohow_id*36+24)+*(partsum+ohow_id*36+25)+*(partsum+ohow_id*36+26)+
					*(partsum+ohow_id*36+27)+*(partsum+ohow_id*36+28)+*(partsum+ohow_id*36+29)+*(partsum+ohow_id*36+30)+*(partsum+ohow_id*36+31)+*(partsum+ohow_id*36+32)+*(partsum+ohow_id*36+33)+*(partsum+ohow_id*36+34)+*(partsum+ohow_id*36+35);

		*(output + ohow_id) = sum;
	}
}