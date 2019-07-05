#include <iostream>
#include <fstream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "dataflow.h"

void print_error_message(cl_int err);
void setup_gpu();
void print_device_info();
cl_program read_cl_file(const char* file_name);
void compile_gpu_program(const char* kernel_file_name, const char* kernel_func);
void run_gpu_program(size_t global_work_size, float* I, float* W, float* O, config* data);
void run();