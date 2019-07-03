#include <iostream>
#include <fstream>
#include "dataflow.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

cl_device_id gpu_device;
cl_context context;
cl_command_queue queue;
cl_kernel kernel;
cl_mem buffer_O;

void print_error_message(cl_int err)
{
	cout << "Error Message :" << err << endl;
	exit(-1);
} 

void setup_gpu()
{
	cl_int err;
	cl_uint num;
	err = clGetPlatformIDs(0, NULL, &num);
	if(err != CL_SUCCESS)
		print_error_message(err);
	cout << "The number of platforms : " << num << endl;

	vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 0, NULL, &num);
	if(err != CL_SUCCESS)
		print_error_message(err);
	cout << "The number of devices : " << num << endl;

	vector<cl_device_id> devices(num);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, num, &devices[0], NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);

	gpu_device = devices[0];

	context = clCreateContext(NULL, num, &devices[0], NULL, NULL, &err);
	if(err != CL_SUCCESS)
		print_error_message(err);

	queue = clCreateCommandQueue(context, devices[0], 0, &err);
	if(err != CL_SUCCESS)
		print_error_message(err);
}

void print_device_info()
{
	printf("//--------------------DEVICE INFO--------------------//\n");
	size_t num;

	cl_ulong global_mem_cache_size;
	clGetDeviceInfo(gpu_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &global_mem_cache_size, NULL);
	cout << "global memory cache : " << global_mem_cache_size << " Bytes" << endl;

	cl_device_mem_cache_type global_mem_cache_type;
	clGetDeviceInfo(gpu_device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &global_mem_cache_type, NULL);
	cout << "global memory cache type : " << global_mem_cache_type << endl;

	cl_uint global_mem_cacheline_size;
	clGetDeviceInfo(gpu_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &global_mem_cacheline_size, NULL);
	cout << "global memory cache line : " << global_mem_cacheline_size << " Bytes" << endl;

	cl_ulong global_mem_size;
	clGetDeviceInfo(gpu_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
	cout << "global device memory : " << global_mem_size << " Bytes" << endl;

	cl_ulong local_mem_size;
	clGetDeviceInfo(gpu_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	cout << "local memory : " << local_mem_size << " Bytes" << endl;

	cl_device_local_mem_type local_mem_type;
	clGetDeviceInfo(gpu_device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &local_mem_type, NULL);
	cout << "local memory type : " << local_mem_type << endl;

	cl_uint compute_units;
	clGetDeviceInfo(gpu_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
	cout << "The number of parallel compute cores on the OpenCL device : " << compute_units << endl;

	size_t max_work_group_size;
	clGetDeviceInfo(gpu_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &num);
	clGetDeviceInfo(gpu_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, num, &max_work_group_size, NULL);
	cout << "Maximum number of work-items in a work-group : " << max_work_group_size << endl;

	cl_uint max_work_item_dim;
	clGetDeviceInfo(gpu_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dim, NULL);
	cout << "Maximum dimensions : " << max_work_item_dim << endl;

	cl_device_type device_type;
	clGetDeviceInfo(gpu_device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	cout << "OpenCL device type : " << device_type << endl;

	string devname;
	clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, 0, NULL, &num);
	devname.resize(num);
	clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, num, &devname[0], NULL);
	cout << "Device name : " << devname.c_str() << endl;

	clGetDeviceInfo(gpu_device, CL_DEVICE_VERSION, 0, NULL, &num);
	devname.resize(num);
	clGetDeviceInfo(gpu_device, CL_DEVICE_VERSION, num, &devname[0], NULL);
	cout << "The OpenCL version supported by the device : " << devname.c_str() << endl;

	clGetDeviceInfo(gpu_device, CL_DRIVER_VERSION, 0, NULL, &num);
	devname.resize(num);
	clGetDeviceInfo(gpu_device, CL_DRIVER_VERSION, num, &devname[0], NULL);
	cout << "OpenCL software driver version :  " << devname.c_str() << endl;

	printf("//--------------------DEVICE INFO--------------------//\n");
}

cl_program read_cl_file(char* file_name)
{
	ifstream in(file_name, ios::binary);
	// get file length
	in.seekg(0, ios_base::end);
	size_t length = in.tellg();
	in.seekg(0, ios_base::beg);
	// read program source
	vector<char> data(length+1);
	in.read(&data[0], length);
	data[length] = 0;

	const char* source = &data[0];
	cl_program program;
	program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);

	return program;
}

void compile_gpu_program(float* I, float* W, float* O, config* data)
{
	cl_int err;
	cl_program program = read_cl_file("conv.cl");
	err = clBuildProgram(program, 1, &gpu_device, NULL, NULL, NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);
 
	kernel = clCreateKernel(program, "convolution", &err);
	if(err != CL_SUCCESS)
		print_error_message(err);

	cl_mem buffer_I = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*4, I, NULL);
	cl_mem buffer_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*4, W, NULL);
	buffer_O = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*4, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_I);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_W);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_O);
}

void run_gpu_program(size_t global_work_size, float* O)
{
	cl_int err;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);
	err = clEnqueueReadBuffer(queue, buffer_O, CL_TRUE, 0, sizeof(float)*4, O, 0, NULL, NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);
	err = clFinish(queue);
	if(err != CL_SUCCESS)
		print_error_message(err);
}

void run()
{
	config* data = (config*)malloc(sizeof(config));
	data->input_size = 0;
	data->input_c = 0;
	data->weight_size = 0;
	data->output_size = 0;
	data->output_c = 0;
	float* I = create_data(4);
	float* W = create_data(4);
	float* O = create_data(4);
	clear_data(O, 4);

	setup_gpu();
	print_device_info();
	compile_gpu_program(I, W, O, data);
	run_gpu_program(4, O);

	for(int i = 0; i < 4; i++)
	{
		float result = 0.0;
		result = (*(I+i)) * (*(W+i));
		printf("cpu=%f\tgpu=%f\n", result, *(O+i));
	}

	free(O);
	free(W);
	free(I);
	free(data);
}

int main(int argc, char* argv[])
{
	run();
	return 0;
}