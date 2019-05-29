#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <string.h>

struct config
{
	int input_size;
	int input_c;
	int weight_size;
	int output_size;
	int output_c;
};

struct tile_param
{
	int tr;
	int tc;
	int tn;
	int tm;
	int block_size;
};

void set_configuration(config* data, tile_param* tile, int* param);
uint64_t total_access_times(config* data);
int total_value(config* data, char data_type);

// 產生data
float* create_data(size_t size);
void reorder_data_layout(float* input, float* kernel, config* data, tile_param* tile);

// 設定不同的block size
void set_block_id(int size);
void label_block_id(float* data, int size);

// 記錄input、weight、output的data access
void memory_trace(char type, float* addr, char* data_type, uint64_t* data_addr, int* addr_idx);
void write_trace_result(char* file_name, char* data_type, uint64_t* data_addr, int addr_count);

// 算出data access的reuse distance
void compute_rd(bool write_file, tile_param* tile, char* data_type, uint64_t* data_addr, int addr_count);

// (2d input) x (2d weight) 
void a_2d_conv(float* input, float* kernel, float* output, config* data, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);

// 不同的convolution dataflow的執行順序
void direct_conv_oc_ic_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);
void direct_conv_ic_oc_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);//2
void direct_conv_oc_oh_ow_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);//3
void direct_conv_oh_ow_oc_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);//4
void direct_conv_ic_oh_ow_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);//5
void direct_conv_oh_ow_ic_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);//6
void direct_conv_kh_kw_oh_ow_oc_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);//7
void direct_conv_kh_kw_oh_ow_oc_ic_reorder(float* input, float* kernel, float* output, config* data, tile_param* tile, bool tile_flag, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);//8

// 當tile要更換時，不同的執行順序
void tile_conv_oh_ow_ic_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);
void tile_conv_ic_oh_ow_oc(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);
void tile_conv_oc_ic_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);
void tile_conv_ic_oc_oh_ow(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);
void tile_conv_oc_oh_ow_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);
void tile_conv_oh_ow_oc_ic(float* input, float* kernel, float* output, config* data, tile_param* tile, int func_type, bool trace_flag, char* data_type, uint64_t* data_addr, int* addr_idx);

void correct_result(float* output_1, float* output_2, int size);
void clear_data(float* data_ptr, int data_size);