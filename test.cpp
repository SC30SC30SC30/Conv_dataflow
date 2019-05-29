#include <time.h>
#include "dataflow.h"

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

	// clock_t start, end;
	// start = clock();
	tile_conv_oh_ow_ic_oc(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);
	// direct_conv_oc_ic_oh_ow(I, W, O, data, tile, false, true, data_type, data_addr, &addr_idx);
	// end = clock();
	// printf("The time = %ld ms\n", (end-start)/**1000/CLOCKS_PER_SEC*/);
	compute_rd(true, tile, data_type, data_addr, total_access);

	write_trace_result("conv_3x3_trace.out", data_type, data_addr, total_access);

	// clear_data(O, total_O);
	// reorder_data_layout(I, W, data, tile);

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
	//for(int TC = 1; TC <= (data->output_size); TC++)
	//{
		//if(((data->output_size) % TC) == 0)
		//{
			//for(int TR = 1; TR <= (data->output_size); TR++)
			//{
				//if(((data->output_size) % TR) == 0)
				//{
					//for(int TM = 1; TM <= (data->output_c); TM++)
					//{
						//if(((data->output_c) % TM) == 0)
						//{
							for(int TN = 1; TN <= (data->input_c); TN++)
							{
								if(((data->input_c) % TN) == 0)
								{
									tile->tr = 13;
									tile->tc = 13;
									tile->tn = TN;
									tile->tm = 4;
									printf("<tr, tc, tn, tm> = <%d, %d, %d, %d>\n", tile->tr, tile->tc, tile->tn, tile->tm);

									tile_conv_oh_ow_ic_oc(I, W, O, data, tile, 1, true, data_type, data_addr, &addr_idx);
									compute_rd(true, tile, data_type, data_addr, total_access);
									clear_data(O, total_O);
									addr_idx = 0;
								}
							}
						//}
					//}
				//}
			//}
		//}
	//}

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
	int a[10] = {6, 4, 3, 4, 8, 2, 2, 2, 4, 1};   // AlexNet的CONV4
	set_configuration(data, tile, a);

	// verify_IR(data, tile);
	run_one_case(data, tile);
	
	free(data);
	free(tile);

	return 0;
}