#include <time.h>
#include "dataflow.h"

bool greater50(int tr, int tc, int tn, int tm, int k_size, char reuse_type)
{
	int tile_ih = tr - 1 + k_size;
	int tile_iw = tc - 1 + k_size;
	int total_I = tile_ih * tile_iw * tn;
	int total_W = k_size * k_size * tn * tm;
	int total_O = tr * tc * tm;

	if(reuse_type == 'I')
	{
		int sum = total_W + total_O;
		if(total_I >= sum)
			return true;
		else
			return false;
	}
	else if(reuse_type == 'W')
	{
		int sum = total_I + total_O;
		if(total_W >= sum)
			return true;
		else 
			return false;
	}
	else if(reuse_type == 'O')
	{
		int sum = total_I + total_W;
		if(total_O >= sum)
			return true;
		else 
			return false;
	}
}

void run_one_case(config* data, tile_param* tile)
{
	float* I = create_data(data->input_size*data->input_size*data->input_c);
	float* W = create_data(data->weight_size*data->weight_size*data->input_c*data->output_c);
	float* O = create_data(data->output_size*data->output_size*data->output_c);
	int total_O = total_value(data, 'O');
	clear_data(O, total_O);
	int total_access = total_access_times(data);

	printf("total_I=%d\ttotal_W=%d\ttotal_O=%d\ttotal_access=%d\n", total_value(data,'I'), total_value(data,'W'), total_value(data,'O'), total_access);

	char* data_type = (char*)malloc(total_access * sizeof(char));
	uint64_t* data_addr = (uint64_t*)malloc(total_access * sizeof(uint64_t));
	int addr_idx = 0;

<<<<<<< Updated upstream
	if(greater50(tile->tr, tile->tc, tile->tn, tile->tm, data->weight_size, 'W'))
	{
		// clock_t start, end;
		// start = clock();
		// tile_conv_oh_ow_ic_oc(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);//IR
		tile_conv_oc_ic_oh_ow(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);//WR
		// direct_conv_oc_ic_oh_ow(I, W, O, data, tile, false, true, data_type, data_addr, &addr_idx);
		// end = clock();
		// printf("The time = %ld ms\n", (end-start)/**1000/CLOCKS_PER_SEC*/);
		compute_rd(true, tile, data_type, data_addr, total_access);

		// write_trace_result("conv_3x3_trace.out", data_type, data_addr, total_access);

		// clear_data(O, total_O);
		// reorder_data_layout(I, W, data, tile);
	}
=======
	// clock_t start, end;
	// start = clock();
	tile_conv_oh_ow_ic_oc(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);
	// direct_conv_oc_ic_oh_ow(I, W, O, data, tile, false, true, data_type, data_addr, &addr_idx);
	// end = clock();
	// printf("The time = %ld ms\n", (end-start)/**1000/CLOCKS_PER_SEC*/);
	compute_rd(true, tile, data_type, data_addr, total_access);

	// write_trace_result("conv_3x3_trace.out", data_type, data_addr, total_access);

	// clear_data(O, total_O);
	// reorder_data_layout(I, W, data, tile);
>>>>>>> Stashed changes

	free(data_addr);
	free(data_type);
	free(O);
	free(W);
	free(I);
}

// Deep Convolutional Neural Network Architecture With Reconfigurable Computation Patterns, 2017
void verify_IR(config* data, tile_param* tile)
{
	float* I = create_data(data->input_size*data->input_size*data->input_c);
	float* W = create_data(data->weight_size*data->weight_size*data->input_c*data->output_c);
	float* O = create_data(data->output_size*data->output_size*data->output_c);
	int total_O = data->output_size * data->output_size * data->output_c;
	clear_data(O, total_O);
	int total_access = total_access_times(data);

	char* data_type = (char*)malloc(total_access * sizeof(char));
	uint64_t* data_addr = (uint64_t*)malloc(total_access * sizeof(uint64_t));
	int addr_idx = 0;

	// exhaustively search
	// for(int TC = 1; TC <= (data->output_size); TC++)
	// {
	// 	if(((data->output_size) % TC) == 0)
	// 	{
			for(int TR = 1; TR <= (data->output_size); TR++)
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
									if(greater50(tile->tr, tile->tc, tile->tn, tile->tm, data->weight_size, 'I'))
									{
										printf("<tr, tc, tn, tm> = <%d, %d, %d, %d>\n", tile->tr, tile->tc, tile->tn, tile->tm);

										tile_conv_oh_ow_ic_oc(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);
										compute_rd(true, tile, data_type, data_addr, total_access);
										clear_data(O, total_O);
										addr_idx = 0;
									}
								}
							}
						}
					}
				}
			}
	// 	}
	// }

	free(data_addr);
	free(data_type);
	free(O);
	free(W);
	free(I);
}

// Deep Convolutional Neural Network Architecture With Reconfigurable Computation Patterns, 2017
void verify_WR(config* data, tile_param* tile)
{
	float* I = create_data(data->input_size*data->input_size*data->input_c);
	float* W = create_data(data->weight_size*data->weight_size*data->input_c*data->output_c);
	float* O = create_data(data->output_size*data->output_size*data->output_c);
	int total_O = data->output_size * data->output_size * data->output_c;
	clear_data(O, total_O);
	int total_access = total_access_times(data);

	char* data_type = (char*)malloc(total_access * sizeof(char));
	uint64_t* data_addr = (uint64_t*)malloc(total_access * sizeof(uint64_t));
	int addr_idx = 0;

	for(int TR = 1; TR <= (data->output_size); TR++)
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
							if(greater50(tile->tr, tile->tc, tile->tn, tile->tm, data->weight_size, 'W'))
							{
								printf("<tr, tc, tn, tm> = <%d, %d, %d, %d>\n", tile->tr, tile->tc, tile->tn, tile->tm);

								tile_conv_oc_ic_oh_ow(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);
								compute_rd(true, tile, data_type, data_addr, total_access);
								clear_data(O, total_O);
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

int main(int argc, char* argv[])
{
	config* data = (config*)malloc(sizeof(config));
	tile_param* tile = (tile_param*)malloc(sizeof(tile_param));
	// int a[10] = {31, 48, 5, 27, 256, 16, 16, 4, 4, 1};   // AlexNet的CONV2
	// int a[10] = {15, 256, 3, 13, 384, 4, 4, 64, 3, 1};   // AlexNet的CONV3
	// int a[10] = {15, 192, 3, 13, 384, 13, 13, 4, 4, 1};   // AlexNet的CONV4
	// int a[10] = {15, 192, 3, 13, 256, 13, 13, 4, 4, 1};   // AlexNet的CONV5
	int a[10] = {4, 4, 1, 4, 8, 2, 2, 2, 4, 1};   // simple test 
	set_configuration(data, tile, a);

	printf("Input : %d x %d x %d\n", data->input_size, data->input_size, data->input_c);
	printf("Weight : %d x %d x %d x %d\n", data->weight_size, data->weight_size, data->input_c, data->output_c);
	printf("Output : %d x %d x %d\n", data->output_size, data->output_size, data->output_c);

	// verify_IR(data, tile);
	// verify_WR(data, tile);
	run_one_case(data, tile);
	
	free(data);
	free(tile);

	return 0;
}