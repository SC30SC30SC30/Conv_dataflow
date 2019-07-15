#include <time.h>
#include "conv_mali.h"

void run_tile_conv(cl_param* cl_gpu, config* data, tile_param* tile, size_t* global_work_size, size_t* local_work_size)
{
	float* I = create_data(total_value(data, 'I'));
	float* W = create_data(total_value(data, 'W'));
	float* O = create_data(total_value(data, 'O'));
	clear_data(O, total_value(data, 'O'));
	printf("total_I=%d\ttotal_W=%d\ttotal_O=%d\n", total_value(data, 'I'), total_value(data, 'W'), total_value(data, 'O'));

	// prepare_to_execute(cl_gpu, false, "conv.cl", "convolution");
	prepare_to_execute(cl_gpu, false, "conv.cl", "convolution");

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
	int partsum_num = tile->tm*tile->tn*tile->tr*tile->tc*data->weight_size*data->weight_size;
	printf("partsum_num=%d\n", partsum_num);
	float* gpu_partsum = malloc_gpu_space(cl_gpu, sizeof(float)*partsum_num);
	float* gpu_O = malloc_gpu_space(cl_gpu, sizeof(float)*total_value(data, 'O'));
	for(int i = 0; i < total_value(data, 'O'); i++)
	{
		*(gpu_O+i) = *(O+i);
	}

	clock_t start, end;
	start = clock();
	tile_conv(cl_gpu, gpu_I, gpu_W, gpu_partsum, gpu_O, data, tile, global_work_size, local_work_size);
	end = clock();
	printf("gpu execution time = %ld ms\n", (end-start)*1000/CLOCKS_PER_SEC);

	// the result of cpu computation
	//==================================================================================================//
	start = clock();
	direct_conv_oc_ic_oh_ow(I, W, O, data, tile, false, false, NULL, NULL, NULL);
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

	int a[10] = {15, 192, 3, 13, 384, 13, 13, 4, 4, 1};   // simple test 
	set_configuration(data, tile, a);

	size_t local_work_size = data->weight_size * data->weight_size * tile->tn;
	size_t global_work_size = local_work_size * data->output_size * data->output_size * tile->tm;
	printf("global_work_size=%ld\n", global_work_size);
	printf("local_work_size=%ld\n", local_work_size);
	printf("work_group_num=%ld\n", (global_work_size/local_work_size));
	run_tile_conv(cl_gpu, data, tile, &global_work_size, &local_work_size);

	free(tile);
	free(data);
	free(cl_gpu);
	return 0;
}