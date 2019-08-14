#include "conv_mali.h"

void run_tile_conv(cl_param* cl_gpu, config* data, tile_param* tile, size_t* global_work_size, size_t* local_work_size)
{
	float* I = create_data(total_value(data, 'I'));
	float* W = create_data(total_value(data, 'W'));
	float* O = create_data(total_value(data, 'O'));
	clear_data(O, total_value(data, 'O'));
	printf("total_I=%d\ttotal_W=%d\ttotal_O=%d\n", total_value(data, 'I'), total_value(data, 'W'), total_value(data, 'O'));

	prepare_to_execute(cl_gpu, false, "OpenCL/conv_IR_seq.cl", "convolution");

	float* gpu_I = malloc_gpu_space(cl_gpu, sizeof(float)*total_value(data, 'I'));
	for(int i = 0; i < total_value(data, 'I'); i++)
	{
		*(gpu_I+i) = *(I+i);
	}
	float* gpu_W = malloc_gpu_space(cl_gpu, sizeof(float)*total_value(data, 'W'));
	for(int i = 0; i < total_value(data, 'W'); i++)
	{
		*(gpu_W+i) = *(W+i);
	}
	float* gpu_O = malloc_gpu_space(cl_gpu, sizeof(float)*total_value(data, 'O'));
	clear_data(gpu_O, total_value(data, 'O'));

	tile_conv(cl_gpu, gpu_I, gpu_W, gpu_O, data, tile, global_work_size, local_work_size);

	// the result of cpu computation
	//==================================================================================================//
	clock_t start, end;
	start = clock();
	for(int oc = 0; oc < 64; oc++)
	{
		for(int ic = 0; ic < 3; ic++)
		{
			for(int oh = 0; oh < 224; oh++)
			{
				for(int ow = 0; ow < 224; ow++)
				{
					for(int kh = 0; kh < 3; kh++)
					{
						for(int kw = 0; kw < 3; kw++)
						{
							float partsum = *(I+ic*226*226+oh*226+ow+kh*226+kw) * *(W+oc*3*3*3+ic*3*3+kh*3+kw);
							*(O+oc*224*224+oh*224+ow) += partsum;
						}
					}
				}
			}
		}
	}
	end = clock();
	printf("cpu execution time = %ld ms\n", (end-start)*1000/CLOCKS_PER_SEC);
	//==================================================================================================//

	// compare cpu and gpu result whether correct
	//==================================================================================================//
	for(int i = 0; i < 20; i++)
	{
		printf("%d.\tcpu=%f\tgpu_O=%f\n", i, *(O+i), *(gpu_O+i));
	}
	//==================================================================================================//

	clean_objects(cl_gpu, gpu_I, gpu_W, gpu_O);
	free(O);
	free(W);
	free(I);
}

int main(int argc, char* argv[])
{
	cl_param* cl_gpu = (cl_param*)malloc(sizeof(cl_param));
	config* data = (config*)malloc(sizeof(config));
	tile_param* tile = (tile_param*)malloc(sizeof(tile_param));

	int a[10] = {226, 3, 3, 224, 64, 28, 28, 3, 2, 1};   // simple test 
	set_configuration(data, tile, a);

	size_t local_work_size = 1;
	size_t global_work_size = 1;
	printf("global_work_size=%ld\n", global_work_size);
	printf("local_work_size=%ld\n", local_work_size);
	printf("work_group_num=%ld\n", (global_work_size/local_work_size));

	run_tile_conv(cl_gpu, data, tile, &global_work_size, &local_work_size);

	free(tile);
	free(data);
	free(cl_gpu);
	return 0;
}