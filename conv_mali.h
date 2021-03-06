#include <iostream>
#include <fstream>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "dataflow.h"

struct cl_param
{
	cl_device_id gpu_device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
};

void print_error_message(cl_int err);
void print_device_info(cl_param* cl_gpu);

void setup_gpu(cl_param* cl_gpu);
cl_program read_cl_file(cl_param* cl_gpu, const char* file_name);
void compile_gpu_program(cl_param* cl_gpu, const char* kernel_file_name, const char* kernel_func);
void prepare_to_execute(cl_param* cl_gpu, bool get_device_info, const char* kernel_file_name, const char* kernel_func);

float* malloc_gpu_space(cl_param* cl_gpu, size_t sz);
cl_mem ptr_to_clmem_unmap(cl_param* cl_gpu, void* ptr);
float* ptr_to_clmem_map(cl_param* cl_gpu, void* ptr);

void tile_conv(cl_param* cl_gpu, float* I, float* W, float* O, config* data, tile_param* tile, size_t* global_work_size, size_t* local_work_size);
void direct_conv(cl_param* cl_gpu, size_t* global_work_size, size_t* local_work_size, float* I, float* W, float* partsum, float* O);

void clean_objects(cl_param* cl_gpu, float* I, float* W, float* O);