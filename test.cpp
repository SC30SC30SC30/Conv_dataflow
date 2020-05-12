#include "dataflow.h"

bool tile_smaller_cache(int tr, int tc, int tn, int tm, int k_size, int cache_size)
{
	int tile_ih = tr - 1 + k_size;
	int tile_iw = tc - 1 + k_size;
	int total_I = tile_ih * tile_iw * tn;
	int total_W = k_size * k_size * tn * tm;
	int total_O = tr * tc * tm;

	return ((total_I + total_W + total_O) < cache_size);
}

void run_one_case(config* data, tile_param* tile, int cache_size)
{
	float* I = create_data(total_value(data, 'I'));
	float* W = create_data(total_value(data, 'W'));
	float* O = create_data(total_value(data, 'O'));
	clear_data(O, total_value(data, 'O'));
	int total_access = total_access_times(data);

	printf("total_I=%d\ttotal_W=%d\ttotal_O=%d\ttotal_access=%d\n", total_value(data,'I'), total_value(data,'W'), total_value(data,'O'), total_access);

	char* data_type = (char*)malloc(total_access * sizeof(char));
	uint64_t* data_addr = (uint64_t*)malloc(total_access * sizeof(uint64_t));
	int addr_idx = 0;

	if(tile_smaller_cache(tile->tr, tile->tc, tile->tn, tile->tm, data->weight_size, cache_size))
	{
		printf("<tr, tc, tn, tm> = <%d, %d, %d, %d>\n", tile->tr, tile->tc, tile->tn, tile->tm);
		int tile_I = (tile->tr-1+data->weight_size) * (tile->tr-1+data->weight_size) * (tile->tn);
		int tile_W = data->weight_size * data->weight_size * tile->tn * tile->tm;
		int tile_O = tile->tr * tile->tc * tile->tm;
		printf("tile_I=%d\ttile_W=%d\ttile_O=%d\n", tile_I, tile_W, tile_O);

		tile_conv_oh_ow_ic_oc(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);//IR
		// tile_conv_oc_ic_oh_ow(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);//WR
		// tile_conv_oh_ow_oc_ic(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);//OR
		// direct_conv_oc_ic_oh_ow(I, W, O, data, tile, false, true, data_type, data_addr, &addr_idx);

		// compute_rd(true, tile, data_type, data_addr, total_access);

		write_trace_result("mem_trace.out", data_type, data_addr, total_access);

		clear_data(O, total_value(data, 'O'));
		// reorder_data_layout(I, W, data, tile);
	}
	else
	{
		printf("tile size doesn't smaller than cache_size\n");
	}

	free(data_addr);
	free(data_type);
	free(O);
	free(W);
	free(I);
}

// Deep Convolutional Neural Network Architecture With Reconfigurable Computation Patterns, 2017
void verify_IR(config* data, tile_param* tile, int cache_size, int block_size)
{
	float* I = create_data(total_value(data, 'I'));
	float* W = create_data(total_value(data, 'W'));
	float* O = create_data(total_value(data, 'O'));
	clear_data(O, total_value(data, 'O'));
	int total_access = total_access_times(data);

	char* data_type = (char*)malloc(total_access * sizeof(char));
	uint64_t* data_addr = (uint64_t*)malloc(total_access * sizeof(uint64_t));
	int addr_idx = 0;

	// exhaustively search
	for(int TR = block_size; TR <= (data->output_size); TR++)
	{
		if(((data->output_size) % TR) == 0)
		{
			for(int TM = 1; TM <= (data->output_c); TM++)
			{
				if(((data->output_c) % TM) == 0)
				{
					for(int TN = 1; TN <= (data->input_c); TN++)
					{
						if(((data->input_c) % TN) == 0)
						{
							tile->tr = TR;
							tile->tc = TR;
							tile->tn = TN;
							tile->tm = TM;
							if(tile_smaller_cache(tile->tr, tile->tc, tile->tn, tile->tm, data->weight_size, cache_size))
							{
								printf("<tr, tc, tn, tm> = <%d, %d, %d, %d>\n", tile->tr, tile->tc, tile->tn, tile->tm);

								tile_conv_oh_ow_ic_oc(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);
								compute_rd(true, tile, data_type, data_addr, total_access);
								clear_data(O, total_value(data, 'O'));
								addr_idx = 0;
							}
						}
					}
				}
			}
		}
	}

	free(data_addr);
	free(data_type);
	free(O);
	free(W);
	free(I);
}

void verify_WR(config* data, tile_param* tile, int cache_size, int block_size)
{
	float* I = create_data(total_value(data, 'I'));
	float* W = create_data(total_value(data, 'W'));
	float* O = create_data(total_value(data, 'O'));
	clear_data(O, total_value(data, 'O'));
	int total_access = total_access_times(data);

	char* data_type = (char*)malloc(total_access * sizeof(char));
	uint64_t* data_addr = (uint64_t*)malloc(total_access * sizeof(uint64_t));
	int addr_idx = 0;

	for(int TR = block_size; TR <= (data->output_size); TR++)
	{
		if(((data->output_size) % TR) == 0)
		{
			for(int TM = 1; TM <= (data->output_c); TM++)
			{
				if(((data->output_c) % TM) == 0)
				{
					for(int TN = 1; TN <= (data->input_c); TN++)
					{
						if(((data->input_c) % TN) == 0)
						{
							tile->tr = TR;
							tile->tc = TR;
							tile->tn = TN;
							tile->tm = TM;
							if(tile_smaller_cache(tile->tr, tile->tc, tile->tn, tile->tm, data->weight_size, cache_size))
							{
								printf("<tr, tc, tn, tm> = <%d, %d, %d, %d>\n", tile->tr, tile->tc, tile->tn, tile->tm);

								tile_conv_oc_ic_oh_ow(I, W, O, data, tile, 4, true, data_type, data_addr, &addr_idx);
								// compute_rd(true, tile, data_type, data_addr, total_access);
								clear_data(O, total_value(data, 'O'));
								addr_idx = 0;
							}
						}
					}
				}
			}
		}
	}

	free(data_addr);
	free(data_type);
	free(O);
	free(W);
	free(I);
}

void matrix_multiplication(int* config)
{
	int row1 = config[4];
	int col1 = config[2] * config[2] * config[1];
	int row2 = col1;
	int col2 = config[3] * config[3];
	int row3 = row1;
	int col3 = col2;

	// produce three matrix
	float* matrix_1 = (float*)malloc((row1*col1) * sizeof(float));
	srand((unsigned)time(NULL));
	for(int i=0; i<(row1*col1); i++)
	{
		*(matrix_1 + i) = rand() / (RAND_MAX + 1.0);
	}

	float* matrix_2 = (float*)malloc((row2*col2) * sizeof(float));
	srand((unsigned)time(NULL));
	for(int i=0; i<(row2*col2); i++)
	{
		*(matrix_2 + i) = rand() / (RAND_MAX + 1.0);
	}

	float* matrix_3 = (float*)malloc((row3*col3) * sizeof(float));
	for(int i=0; i<(row3*col3); i++)
	{
		*(matrix_3 + i) = 0.0;
	}

	// record memory access
	int total_access = 0;
	for(int i = 0; i < row3; i++)
	{
		for(int j = 0; j < col3; j++)
		{
			for(int k = 0; k < col1; k++)
			{
				total_access += 3;
			}
		}
	}
	printf("total_access = %d\n", total_access);
	char* data_type = (char*)malloc(total_access * sizeof(char));
	uint64_t* data_addr = (uint64_t*)malloc(total_access * sizeof(uint64_t));
	int addr_idx = 0;

	// matrix multiplication
	for(int i = 0; i < row3; i++)
	{
		for(int j = 0; j < col3; j++)
		{
			for(int k = 0; k < col1; k++)
			{
				*(matrix_3 + i*col3 + j) += (*(matrix_1 + i*col1 + k) * *(matrix_2 + k*col2 + j));
				*(data_type + addr_idx) = 'i';
				*(data_addr + addr_idx) = (uint64_t)(matrix_1 + i*col1 + k);
				*(data_type + addr_idx + 1) = 'w';
				*(data_addr + addr_idx + 1) = (uint64_t)(matrix_2 + k*col2 + j);
				*(data_type + addr_idx + 2) = 'o';
				*(data_addr + addr_idx + 2) = (uint64_t)(matrix_3 + i*col3 + j);
				addr_idx += 3;
			}
		}
	}
	printf("addr_idx = %d\ttotal_access = %d\n", addr_idx, total_access);
	write_trace_result("mem_trace.out", data_type, data_addr, total_access);
	// compute_rd(false, NULL, data_type, data_addr, total_access);

	printf("%d\t%d\t%d\t%d\t%d\t%d\n", row1, col1, row2, col2, row3, col3);

	free(data_addr);
	free(data_type);
	free(matrix_3);
	free(matrix_2);
	free(matrix_1);
}

int main(int argc, char* argv[])
{
	config* data = (config*)malloc(sizeof(config));
	tile_param* tile = (tile_param*)malloc(sizeof(tile_param));
	// int a[10] = {31, 48, 5, 27, 256, 16, 16, 4, 4, 1};   // AlexNet的CONV2
	// int a[10] = {15, 256, 3, 13, 384, 13, 13, 256, 3, 1};   // AlexNet的CONV3
	int a[10] = {15, 192, 3, 13, 384, 13, 13, 8, 192, 1};   // AlexNet的CONV4
	// int a[10] = {15, 192, 3, 13, 256, 13, 13, 12, 4, 1};   // AlexNet的CONV5
	// int a[10] = {8, 8, 1, 8, 16, 2, 2, 4, 2, 1};   // simple test 
	set_configuration(data, tile, a);

	// matrix_multiplication(a);

	printf("Input : %d x %d x %d\n", data->input_size, data->input_size, data->input_c);
	printf("Weight : %d x %d x %d x %d\n", data->weight_size, data->weight_size, data->input_c, data->output_c);
	printf("Output : %d x %d x %d\n", data->output_size, data->output_size, data->output_c);

	run_one_case(data, tile, (4*1024/4));
	
	free(data);
	free(tile);

	return 0;
}