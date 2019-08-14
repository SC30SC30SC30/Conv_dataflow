#include "splay_tree.h"
#include "dataflow.h"

std::map<float*, int> data_block_id;
int block_id = 1;
int file_count = 1;

void set_configuration(config* data, tile_param* tile, int* param)
{
	data->input_size = param[0];
	data->input_c = param[1];
	data->weight_size = param[2];
	data->output_size = param[3];
	data->output_c = param[4];

	tile->tr = param[5];
	tile->tc = param[6];
	tile->tn = param[7];
	tile->tm = param[8];
	tile->block_size = param[9];
}

uint64_t total_access_times(config* data)
{
	uint64_t count = 0;
	for(int oc = 0; oc < data->output_c; oc++)
	{
		for(int ic = 0; ic < data->input_c; ic++)
		{
			for(int oh = 0; oh < data->output_size; oh++)
			{
				for(int ow = 0; ow < data->output_size; ow++)
				{
					for(int kh = 0; kh < data->weight_size; kh++)
					{
						for(int kw = 0; kw < data->weight_size; kw++)
						{
							count += 3;
						}
					}
				}
			}
		}
	}
	return count;
}

int total_value(config* data, char data_type)
{
	int result = 0;
	if(data_type == 'I')
		result = data->input_size * data->input_size * data->input_c;
	else if(data_type == 'W')
		result = data->weight_size * data->weight_size * data->input_c * data->output_c;
	else if(data_type == 'O')
		result = data->output_size * data->output_size * data->output_c;
	return result;
}

float* create_data(size_t size)
{
	float* data = (float*)malloc(size * sizeof(float));

	srand((unsigned)time(NULL));
	for(int i=0; i<size; i++)
	{
		*(data + i) = rand() / (RAND_MAX + 1.0);
	}

	return data;
}

// 重排func_type = 7的資料分佈
void reorder_data_layout(float* input, float* kernel, config* data, tile_param* tile)
{
	float* new_input = create_data(data->input_size*data->input_size*data->input_c);
	float* new_weight = create_data(data->weight_size*data->weight_size*data->input_c*data->output_c);

	int total_I = data->input_size * data->input_size * data->input_c;
	int total_W = data->weight_size * data->weight_size * data->input_c * data->output_c;
	int total_O = data->output_size * data->output_size * data->output_c;
	printf("total_I = %d\ttotal_W = %d\ttotal_O = %d\n", total_I, total_W, total_O);

	printf("Old Input : \n");
	for(int i = 0; i < total_I; i++)
	{
		printf("%d\t%f\n", i+1, *(input+i));
	}
	printf("Old Weight : \n");
	for(int i = 0; i < total_W; i++)
	{
		printf("%d\t%f\n", i+1, *(kernel+i));
	}

	// reorder input
	int idx = 0;
	for(int ic = 0; ic < data->input_c; ic += tile->tn)
	{
		for(int ih = 0; ih < data->input_size; ih++)
		{
			for(int iw = 0; iw < data->input_size; iw++)
			{
				for(int t_ic = 0; t_ic < tile->tn; t_ic++)
				{
					float value = *(input + 
									ic*data->input_size*data->input_size + 
									ih*data->input_size + 
									iw + 
									t_ic*data->input_size*data->input_size);
					*(new_input + idx) = value;
					idx++;
				}
			}
		}
	}
	printf("Reorder input data finish (idx = %d)\n", idx);

	idx = 0;
	for (int oc = 0; oc < data->output_c; oc += tile->tm)
	{
		for(int ic = 0; ic < data->input_c; ic += tile->tn)
		{
			for(int kh = 0; kh < data->weight_size; kh++)
			{
				for(int kw = 0; kw < data->weight_size; kw++)
				{
					for(int t_oc = 0; t_oc < tile->tm; t_oc++)
					{
						for(int t_ic = 0; t_ic < tile->tn; t_ic++)
						{
							float value = *(kernel + 
									oc*data->weight_size*data->weight_size*data->input_c + 
									ic*data->weight_size*data->weight_size + 
									kh*data->weight_size + 
									kw + 
									t_oc*data->weight_size*data->weight_size*data->input_c + 
									t_ic*data->weight_size*data->weight_size);
							*(new_weight + idx) = value;
							idx++;
						}
					}
				}
			}
		}
	}
	printf("Reorder weight data finish (idx = %d)\n", idx);

	for(int i = 0; i < total_W; i++)
	{
		*(kernel + i) = *(new_weight + i);
	}
	free(new_weight);

	for(int i = 0; i < total_I; i++)
	{
		*(input + i) = *(new_input + i);
	}
	free(new_input);

	printf("New Input : \n");
	for(int i = 0; i < total_I; i++)
	{
		printf("%d\t%f\n", i+1, *(input+i));
	}
	printf("New Weight : \n");
	for(int i = 0; i < total_W; i++)
	{
		printf("%d\t%f\n", i+1, *(kernel+i));
	}
}

void label_block_id(float* data, int total_data, int block_size)
{
	int count = 1;
	for (int i = 0; i < total_data; i++)
	{
		data_block_id[(data + i)] = block_id;
		if(count == block_size)
		{
			count = 0;
			block_id++;
		}
		count++;
		printf("%p\t%d\n", (data+i), data_block_id[(data + i)]);
	}
	block_id++;
}

void memory_trace(int block_size, char type, float* addr, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	*(data_type + (*addr_idx)) = type;
	if(block_size == 1)
	{
		*(data_addr + (*addr_idx)) = (uint64_t)addr;
	}
	else
	{
		*(data_addr + (*addr_idx)) = (uint64_t)data_block_id[addr];
	}
	(*addr_idx)++;
}

void write_trace_result(char* file_name, char* data_type, uint64_t* data_addr, int addr_count)
{
	printf("Write %s Begin !!!\n", file_name);
	FILE *fp;
	fp = fopen(file_name, "w");
	for(int i = 0; i < addr_count; i++)
	{
		if(*(data_type + i) == 'o')
			fprintf(fp, "1 %lx\n", *(data_addr + i));
		else
			fprintf(fp, "0 %lx\n", *(data_addr + i));

		// fprintf(fp, "%c\t%lx\n", *(data_type + i), *(data_addr + i));
	}
	fclose(fp);
	printf("Write %s Finish !!!\n", file_name);
}

void compute_rd(bool write_file, tile_param* tile, char* data_type, uint64_t* data_addr, int addr_count)
{
	// Variable section
	TREE_NODE *tree = NULL;// splay tree
	std::map<uint64_t, uint64_t> HASH_TABLE; // store last access time
	uint64_t now_t = 1;
	std::map<uint64_t,uint64_t> ReuseDistance;
	uint64_t infinite_dist=0, max_dist=0; 	

	// Compute reuse distance
	for(int i = 0; i < addr_count; i++)
	{
		// printf("%c\t%p\n", *(data_type+i), *(data_addr+i));
		uint64_t dist=0;
		uint64_t last_t = HASH_TABLE[*(data_addr+i)];

		// dist = infinite 
		if( last_t == 0 ){
			infinite_dist++;
			tree_insert(&tree, now_t, *(data_addr+i));
		}
		else{ 
			dist = distance_compute(&tree, last_t);
			tree_insert(&tree, now_t, *(data_addr+i));
			ReuseDistance[dist]++;
			if( dist > max_dist)
				max_dist = dist;
		}
		HASH_TABLE[*(data_addr+i)] = now_t;
		now_t++;
	}
	
	//==========Output the result to file==========
	if(write_file)
	{
		char file_name[50] = "reuse_dist_";
		char temp[30];
		sprintf(temp, "%d_%d_%d_%d_%dth", tile->tr, tile->tc, tile->tn, tile->tm, file_count);
		strcat(file_name, temp);
		printf("Write %s Begin !!!\n", file_name);
		FILE *fp = fopen(file_name, "w");
		for(uint64_t i= 0; i <= max_dist ; i++)
		{
			if(ReuseDistance[i] != 0)
				fprintf(fp,"%-6lu %-6lu\n", i, ReuseDistance[i]);
		}
		fprintf(fp,"-1 %-6lu\n", infinite_dist);
		fclose(fp);
		printf("Write %s Finish !!!\n", file_name);
		file_count++;
	}
	else
	{
		for(uint64_t i= 0; i <= max_dist ; i++)
		{
			if(ReuseDistance[i] != 0)
				printf("%-6lu %-6lu\n", i, ReuseDistance[i]);
		}
		printf("-1 %-6lu\n", infinite_dist);
	}
}

void a_2d_conv(float* input, float* kernel, float* output, config* data, tile_param* tile, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	for(int kh = 0; kh < (data->weight_size); kh++)
	{
		for(int kw = 0; kw < (data->weight_size); kw++)
		{
			float* i_addr = input + kh*(data->input_size) + kw;
			float* w_addr = kernel + kh*(data->weight_size) + kw;
			*(output) += (*(i_addr) * *(w_addr));
			if(trace_flag)
			{
				memory_trace(tile->block_size, 'i', i_addr, data_type, data_addr, addr_idx);
				memory_trace(tile->block_size, 'w', w_addr, data_type, data_addr, addr_idx);
				memory_trace(tile->block_size, 'o', output, data_type, data_addr, addr_idx);
			}
		}
	}
}

// func_type = 1
void direct_conv_oc_ic_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tm);
		loop_num.push_back(tile->tn);
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
	}
	else
	{
		loop_num.push_back(data->output_c);
		loop_num.push_back(data->input_c);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
	}

	for(int oc = 0; oc < loop_num[0]; oc++)
	{
		for(int ic = 0; ic < loop_num[1]; ic++)
		{
			for(int oh = 0; oh < loop_num[2]; oh++)
			{
				for(int ow = 0; ow < loop_num[3]; ow++)
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					a_2d_conv(i_addr, w_addr, o_addr, data, tile, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}
}

// func_type = 2
void direct_conv_ic_oc_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tn);
		loop_num.push_back(tile->tm);
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
	}
	else
	{
		loop_num.push_back(data->input_c);
		loop_num.push_back(data->output_c);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
	}

	for(int ic = 0; ic < loop_num[0]; ic++)
	{
		for(int oc = 0; oc < loop_num[1]; oc++)
		{
			for(int oh = 0; oh < loop_num[2]; oh++)
			{
				for(int ow = 0; ow < loop_num[3]; ow++)
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					a_2d_conv(i_addr, w_addr, o_addr, data, tile, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}
}

// func_type = 3
void direct_conv_oc_oh_ow_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tm);
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
		loop_num.push_back(tile->tn);
	}
	else
	{
		loop_num.push_back(data->output_c);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->input_c);
	}

	for(int oc = 0; oc < loop_num[0]; oc++)
	{
		for(int oh = 0; oh < loop_num[1]; oh++)
		{
			for(int ow = 0; ow < loop_num[2]; ow++)
			{
				for(int ic = 0; ic < loop_num[3]; ic++)
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					a_2d_conv(i_addr, w_addr, o_addr, data, tile, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}
}

// func_type = 4
void direct_conv_oh_ow_oc_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
		loop_num.push_back(tile->tm);
		loop_num.push_back(tile->tn);
	}
	else
	{
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_c);
		loop_num.push_back(data->input_c);
	}

	for(int oh = 0; oh < loop_num[0]; oh++)
	{
		for(int ow = 0; ow < loop_num[1]; ow++)
		{
			for(int oc = 0; oc < loop_num[2]; oc++)
			{
				for(int ic = 0; ic < loop_num[3]; ic++)
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					a_2d_conv(i_addr, w_addr, o_addr, data, tile, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}
}

// func_type = 5
void direct_conv_ic_oh_ow_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tn);
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
		loop_num.push_back(tile->tm);
	}
	else
	{
		loop_num.push_back(data->input_c);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_c);
	}

	for(int ic = 0; ic < loop_num[0]; ic++)
	{
		for(int oh = 0; oh < loop_num[1]; oh++)
		{
			for(int ow = 0; ow < loop_num[2]; ow++)
			{
				for(int oc = 0; oc < loop_num[3]; oc++)
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					a_2d_conv(i_addr, w_addr, o_addr, data, tile, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}
}

// func_type = 6
void direct_conv_oh_ow_ic_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
		loop_num.push_back(tile->tn);
		loop_num.push_back(tile->tm);
	}
	else
	{
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->input_c);
		loop_num.push_back(data->output_c);
	}

	for(int oh = 0; oh < loop_num[0]; oh++)
	{
		for(int ow = 0; ow < loop_num[1]; ow++)
		{
			for(int ic = 0; ic < loop_num[2]; ic++)
			{
				for(int oc = 0; oc < loop_num[3]; oc++)
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					a_2d_conv(i_addr, w_addr, o_addr, data, tile, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}
}

// func_type = 7
void direct_conv_kh_kw_oh_ow_oc_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
		loop_num.push_back(tile->tm);
		loop_num.push_back(tile->tn);
	}
	else
	{
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_c);
		loop_num.push_back(data->input_c);
	}

	for(int kh = 0; kh < (data->weight_size); kh++)
	{
		for(int kw = 0; kw < (data->weight_size); kw++)
		{
			for(int oh = 0; oh < loop_num[0]; oh++)
			{
				for(int ow = 0; ow < loop_num[1]; ow++)
				{
					for(int oc = 0; oc < loop_num[2]; oc++)
					{
						for(int ic = 0; ic < loop_num[3]; ic++)
						{
							float* i_addr = input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow + kh*(data->input_size) + kw;
							float* w_addr = kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size) + kh*(data->weight_size) + kw;
							float* o_addr = output + oc*data->output_size*data->output_size + oh*data->output_size + ow;

							*(o_addr) += (*(i_addr) * *(w_addr));

							if(trace_flag)
							{
								memory_trace(tile->block_size, 'i', i_addr, data_type, data_addr, addr_idx);
								memory_trace(tile->block_size, 'w', w_addr, data_type, data_addr, addr_idx);
								memory_trace(tile->block_size, 'o', o_addr, data_type, data_addr, addr_idx);
							}
						}
					}
				}
			}
		}
	}

	// printf("direct_conv_kh_kw_oh_ow_oc_ic function finish !!!\n");
}

void direct_conv_kh_kw_oh_ow_oc_ic_reorder(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	std::vector<int> loop_num;
	if(tile_flag)
	{
		loop_num.push_back(tile->tr);
		loop_num.push_back(tile->tc);
		loop_num.push_back(tile->tm);
		loop_num.push_back(tile->tn);
	}
	else
	{
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_size);
		loop_num.push_back(data->output_c);
		loop_num.push_back(data->input_c);
	}

	for(int kh = 0; kh < (data->weight_size); kh++)
	{
		for(int kw = 0; kw < (data->weight_size); kw++)
		{
			for(int oh = 0; oh < loop_num[0]; oh++)
			{
				for(int ow = 0; ow < loop_num[1]; ow++)
				{
					for(int oc = 0; oc < loop_num[2]; oc++)
					{
						for(int ic = 0; ic < loop_num[3]; ic++)
						{
							float* i_addr = input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow + kh*(data->input_size) + kw;
							float* w_addr = kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size) + kh*(data->weight_size) + kw;
							float* o_addr = output + oc*data->output_size*data->output_size + oh*data->output_size + ow;

							*(o_addr) += (*(i_addr) * *(w_addr));

							if(trace_flag)
							{
								memory_trace(tile->block_size, 'i', i_addr, data_type, data_addr, addr_idx);
								memory_trace(tile->block_size, 'w', w_addr, data_type, data_addr, addr_idx);
								memory_trace(tile->block_size, 'o', o_addr, data_type, data_addr, addr_idx);
							}
						}
					}
				}
			}
		}
	}

	// printf("direct_conv_kh_kw_oh_ow_oc_ic_reorder function finish !!!\n");
}

void tile_conv_oh_ow_ic_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	if(((data->output_size % tile->tr) != 0) || ((data->output_size % tile->tc) != 0) || ((data->input_c % tile->tn) != 0) || ((data->output_c % tile->tm) != 0))
	{
		printf("tile paramater error !!!\n");
		exit(1);
	}

	for(int oh = 0; (oh+(tile->tr)-1) < data->output_size; oh += (tile->tr))
	{
		for(int ow = 0; (ow+(tile->tc)-1) < data->output_size; ow += (tile->tc))
		{
			for(int ic = 0; ic < data->input_c; ic += (tile->tn))
			{
				for(int oc = 0; oc < data->output_c; oc += (tile->tm))
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					if(func_type == 1)
						direct_conv_oc_ic_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 2)
						direct_conv_ic_oc_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 3)
						direct_conv_oc_oh_ow_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 4)
						direct_conv_oh_ow_oc_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 5)
						direct_conv_ic_oh_ow_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 6)
						direct_conv_oh_ow_ic_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}

	printf("tile_conv_oh_ow_ic_oc function finish !!!\n");
}

void tile_conv_ic_oh_ow_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	if(((data->output_size % tile->tr) != 0) || ((data->output_size % tile->tc) != 0) || ((data->input_c % tile->tn) != 0) || ((data->output_c % tile->tm) != 0))
	{
		printf("tile paramater error !!!\n");
		exit(1);
	}

	for(int ic = 0; ic < data->input_c; ic += tile->tn)
	{
		for(int oh = 0; (oh+tile->tr-1) < data->output_size; oh += tile->tr)
		{
			for(int ow = 0; (ow+tile->tc-1) < data->output_size; ow+= tile->tc)
			{
				for(int oc = 0; oc < data->output_c; oc += tile->tm)
				{
					float* i_addr = (input + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_addr = (kernel + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					if(func_type == 1)
						direct_conv_oc_ic_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 2)
						direct_conv_ic_oc_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 3)
						direct_conv_oc_oh_ow_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 4)
						direct_conv_oh_ow_oc_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 5)
						direct_conv_ic_oh_ow_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 6)
						direct_conv_oh_ow_ic_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}

	printf("tile_conv_ic_oh_ow_oc function finish !!!\n");
}

void tile_conv_oc_ic_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	if(((data->output_size % tile->tr) != 0) || ((data->output_size % tile->tc) != 0) || ((data->input_c % tile->tn) != 0) || ((data->output_c % tile->tm) != 0))
	{
		printf("tile paramater error !!!\n");
		exit(1);
	}

	for(int oc = 0; oc < data->output_c; oc += tile->tm)
	{
		for(int ic = 0; ic < data->input_c; ic += tile->tn)
		{
			for(int oh = 0; (oh+tile->tr-1) < data->output_size; oh += tile->tr)
			{
				for(int ow = 0; (ow+tile->tc-1) < data->output_size; ow += tile->tc)
				{
					float* i_addr = (input + ic*data->input_size*data->input_size + oh*data->input_size + ow);
					float* w_addr = (kernel + oc*data->weight_size*data->weight_size*data->input_c + ic*data->weight_size*data->weight_size);
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					if(func_type == 1)
						direct_conv_oc_ic_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 2)
						direct_conv_ic_oc_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 3)
						direct_conv_oc_oh_ow_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 4)
						direct_conv_oh_ow_oc_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 5)
						direct_conv_ic_oh_ow_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 6)
						direct_conv_oh_ow_ic_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}

	printf("tile_conv_oc_ic_oh_ow function finish !!!\n");
}

void tile_conv_ic_oc_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	if(((data->output_size % tile->tr) != 0) || ((data->output_size % tile->tc) != 0) || ((data->input_c % tile->tn) != 0) || ((data->output_c % tile->tm) != 0))
	{
		printf("tile paramater error !!!\n");
		exit(1);
	}

	for(int ic = 0; ic < data->input_c; ic += tile->tn)
	{
		for(int oc = 0; oc < data->output_c; oc += tile->tm)
		{
			for(int oh = 0; (oh+tile->tr-1) < data->output_size; oh += tile->tr)
			{
				for(int ow = 0; (ow+tile->tc-1) < data->output_size; ow+= tile->tc)
				{
					float* i_addr = (input + ic*data->input_size*data->input_size + oh*data->input_size + ow);
					float* w_addr = (kernel + oc*data->weight_size*data->weight_size*data->input_c + ic*data->weight_size*data->weight_size);
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					if(func_type == 1)
						direct_conv_oc_ic_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 2)
						direct_conv_ic_oc_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 3)
						direct_conv_oc_oh_ow_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 4)
						direct_conv_oh_ow_oc_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 5)
						direct_conv_ic_oh_ow_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 6)
						direct_conv_oh_ow_ic_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}

	printf("tile_conv_ic_oc_oh_ow function finish !!!\n");
}

void tile_conv_oc_oh_ow_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	if(((data->output_size % tile->tr) != 0) || ((data->output_size % tile->tc) != 0) || ((data->input_c % tile->tn) != 0) || ((data->output_c % tile->tm) != 0))
	{
		printf("tile paramater error !!!\n");
		exit(1);
	}

	for(int oc = 0; oc < data->output_c; oc += tile->tm)
	{
		for(int oh = 0; (oh+tile->tr-1) < data->output_size; oh += tile->tr)
		{
			for(int ow = 0; (ow+tile->tc-1) < data->output_size; ow+= tile->tc)
			{
				for(int ic = 0; ic < data->input_c; ic += tile->tn)
				{
					float* i_addr = (input + ic*data->input_size*data->input_size + oh*data->input_size + ow);
					float* w_addr = (kernel + oc*data->weight_size*data->weight_size*data->input_c + ic*data->weight_size*data->weight_size);
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					if(func_type == 1)
						direct_conv_oc_ic_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 2)
						direct_conv_ic_oc_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 3)
						direct_conv_oc_oh_ow_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 4)
						direct_conv_oh_ow_oc_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 5)
						direct_conv_ic_oh_ow_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 6)
						direct_conv_oh_ow_ic_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}

	printf("tile_conv_oc_oh_ow_ic function finish !!!\n");
}

void tile_conv_oh_ow_oc_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx)
{
	if(((data->output_size % tile->tr) != 0) || ((data->output_size % tile->tc) != 0) || ((data->input_c % tile->tn) != 0) || ((data->output_c % tile->tm) != 0))
	{
		printf("tile paramater error !!!\n");
		exit(1);
	}

	for(int oh = 0; (oh+tile->tr-1) < data->output_size; oh += tile->tr)
	{
		for(int ow = 0; (ow+tile->tc-1) < data->output_size; ow += tile->tc)
		{
			for(int oc = 0; oc < data->output_c; oc += tile->tm)
			{
				for(int ic = 0; ic < data->input_c; ic += tile->tn)
				{
					float* i_addr = (input + ic*data->input_size*data->input_size + oh*data->input_size + ow);
					float* w_addr = (kernel + oc*data->weight_size*data->weight_size*data->input_c + ic*data->weight_size*data->weight_size);
					float* o_addr = (output + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					if(func_type == 1)
						direct_conv_oc_ic_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 2)
						direct_conv_ic_oc_oh_ow(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 3)
						direct_conv_oc_oh_ow_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 4)
						direct_conv_oh_ow_oc_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 5)
						direct_conv_ic_oh_ow_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 6)
						direct_conv_oh_ow_ic_oc(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
					else if(func_type == 7)
						direct_conv_kh_kw_oh_ow_oc_ic(i_addr, w_addr, o_addr, data, tile, true, trace_flag, data_type, data_addr, addr_idx);
				}
			}
		}
	}

	printf("tile_conv_oh_ow_oc_ic function finish !!!\n");
}

void correct_result(float* output_1, float* output_2, int size)
{
	std::vector <int> idx;
	std::vector <float> value_1;
	std::vector <float> value_2;
	int correct = 1;

	for(int a = 0; a < size; a++)
	{
	 	float result1 = *(output_1 + a);
	 	float result2 = *(output_2 + a);
	 	if(result1 != result2)
	 	{
	 		correct = 0;
	 		idx.push_back(a);
	 		value_1.push_back(result1);
	 		value_2.push_back(result2);
	 	}
	}

	if(correct == 1)
		printf("Correct !!!\n");
	else
	{
		printf("Error !!!\n");
		// for(int a = 0; a < 10; a++)
		// {
		// 	// printf("idx:%d  %.*f\n", idx.at(a), 30, (value_1.at(a)-value_2.at(a)));
		// 	printf("idx:%d  %f  %f\n", idx.at(a), value_1.at(a), value_2.at(a));
		// }
	}
}

void clear_data(float* data_ptr, int data_size)
{
	for(int i = 0; i < data_size; i++)
	{
		*(data_ptr + i) = 0.0;
	}
}