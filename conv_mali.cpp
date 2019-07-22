#include "conv_mali.h"

using namespace std;

map<void*, cl_mem> map_ptr_to_cl_mem;
map<void*, size_t> map_ptr_to_size;

void print_error_message(cl_int err)
{
	cout << "Error Message :" << err << endl;
	exit(-1);
} 

void setup_gpu(cl_param* cl_gpu)
{
	cl_int err;
	cl_uint num;
	err = clGetPlatformIDs(0, NULL, &num);
	if(err != CL_SUCCESS)
		print_error_message(err);
	// cout << "The number of platforms : " << num << endl;

	vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 0, NULL, &num);
	if(err != CL_SUCCESS)
		print_error_message(err);
	// cout << "The number of devices : " << num << endl;

	vector<cl_device_id> devices(num);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, num, &devices[0], NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);

	cl_gpu->gpu_device = devices[0];

	cl_gpu->context = clCreateContext(NULL, num, &devices[0], NULL, NULL, &err);
	if(err != CL_SUCCESS)
		print_error_message(err);

	cl_gpu->queue = clCreateCommandQueue(cl_gpu->context, devices[0], 0, &err);
	if(err != CL_SUCCESS)
		print_error_message(err);
}

void print_device_info(cl_param* cl_gpu)
{
	printf("//--------------------DEVICE INFO--------------------//\n");
	size_t num;

	cl_ulong global_mem_cache_size;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &global_mem_cache_size, NULL);
	cout << "\tglobal memory cache : " << global_mem_cache_size << " Bytes" << endl;

	cl_device_mem_cache_type global_mem_cache_type;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &global_mem_cache_type, NULL);
	cout << "\tglobal memory cache type : " << global_mem_cache_type << endl;

	cl_uint global_mem_cacheline_size;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &global_mem_cacheline_size, NULL);
	cout << "\tglobal memory cache line : " << global_mem_cacheline_size << " Bytes" << endl;

	cl_ulong global_mem_size;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
	cout << "\tglobal device memory : " << global_mem_size << " Bytes" << endl;

	cl_ulong local_mem_size;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	cout << "\tlocal memory : " << local_mem_size << " Bytes" << endl;

	cl_device_local_mem_type local_mem_type;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &local_mem_type, NULL);
	cout << "\tlocal memory type : " << local_mem_type << endl;

	cl_ulong mem_alloc_size;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_alloc_size, NULL);
	cout << "\tMaximum memory object allocation : " << mem_alloc_size << " Bytes" << endl;

	cl_uint compute_units;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
	cout << "\tThe number of parallel compute cores on the OpenCL device : " << compute_units << endl;

	size_t max_work_group_size;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &num);
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, num, &max_work_group_size, NULL);
	cout << "\tMaximum number of work-items in a work-group : " << max_work_group_size << endl;

	cl_uint max_work_item_dim;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dim, NULL);
	cout << "\tMaximum dimensions : " << max_work_item_dim << endl;

	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &num);
	vector<size_t> max_work_item_size(num);
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, num, &max_work_item_size[0], NULL);
	cout << "\tMaximum work item for each dimensions : " << "(" << max_work_item_size[0] << ", " << max_work_item_size[1] << ", " << max_work_item_size[2] << ")" << endl;

	cl_uint max_clock_frequency;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_frequency, NULL);
	cout << "\tMaximum clock frequency : " << max_clock_frequency << " MHz" << endl;

	cl_device_type device_type;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	cout << "\tOpenCL device type : " << device_type << endl;

	string devname;
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_NAME, 0, NULL, &num);
	devname.resize(num);
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_NAME, num, &devname[0], NULL);
	cout << "\tDevice name : " << devname.c_str() << endl;

	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_VERSION, 0, NULL, &num);
	devname.resize(num);
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DEVICE_VERSION, num, &devname[0], NULL);
	cout << "\tThe OpenCL version supported by the device : " << devname.c_str() << endl;

	clGetDeviceInfo(cl_gpu->gpu_device, CL_DRIVER_VERSION, 0, NULL, &num);
	devname.resize(num);
	clGetDeviceInfo(cl_gpu->gpu_device, CL_DRIVER_VERSION, num, &devname[0], NULL);
	cout << "\tOpenCL software driver version :  " << devname.c_str() << endl;

	printf("//--------------------DEVICE INFO--------------------//\n");
}

cl_program read_cl_file(cl_param* cl_gpu, const char* file_name)
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
	program = clCreateProgramWithSource(cl_gpu->context, 1, &source, NULL, NULL);

	return program;
}

void compile_gpu_program(cl_param* cl_gpu, const char* kernel_file_name, const char* kernel_func)
{
	cl_int err;
	cl_gpu->program = read_cl_file(cl_gpu, kernel_file_name);
	err = clBuildProgram(cl_gpu->program, 1, &(cl_gpu->gpu_device), NULL, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		// print_error_message(err);

		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(cl_gpu->program, cl_gpu->gpu_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(cl_gpu->program, cl_gpu->gpu_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);

	}
 
	cl_gpu->kernel = clCreateKernel(cl_gpu->program, kernel_func, &err);
	if(err != CL_SUCCESS)
		print_error_message(err);
}

void prepare_to_execute(cl_param* cl_gpu, bool get_device_info, const char* kernel_file_name, const char* kernel_func)
{
	setup_gpu(cl_gpu);
	if(get_device_info)
	{
		printf("\n");
		print_device_info(cl_gpu);
		printf("\n");
	}
	compile_gpu_program(cl_gpu, kernel_file_name, kernel_func);
}

float* malloc_gpu_space(cl_param* cl_gpu, size_t sz)
{
	cl_int err;
	cl_mem buffer = clCreateBuffer(cl_gpu->context, CL_MEM_ALLOC_HOST_PTR, sz, NULL, NULL);
	float* ptr = (float*)clEnqueueMapBuffer(cl_gpu->queue, buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
	if(err != CL_SUCCESS)
		print_error_message(err);
	map_ptr_to_cl_mem[(void*)ptr] = buffer;
	map_ptr_to_size[(void*)ptr] = sz;
	return ptr;
}

cl_mem ptr_to_clmem_unmap(cl_param* cl_gpu, void* ptr)
{	
	cl_mem buffer  = map_ptr_to_cl_mem[ptr];
	clEnqueueUnmapMemObject(cl_gpu->queue, buffer, ptr, 0, NULL, NULL);	
	return buffer;
}

float* ptr_to_clmem_map(cl_param* cl_gpu, void* ptr)
{	
	cl_mem buffer  = map_ptr_to_cl_mem[ptr];
	size_t sz = map_ptr_to_size[ptr];
	float* new_ptr = (cl_float*)clEnqueueMapBuffer(cl_gpu->queue, buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sz, 0, NULL, NULL, NULL);
	return new_ptr;
}

void run_gpu_program(cl_param* cl_gpu, size_t* global_work_size, size_t* local_work_size, float* I, float* W, float* partsum, float* O, int i_offset, int w_offset, int o_offset)
{
	cl_int err;

	cl_mem buffer_I = ptr_to_clmem_unmap(cl_gpu, (void*)I);
	cl_mem buffer_W = ptr_to_clmem_unmap(cl_gpu, (void*)W);
	cl_mem buffer_partsum = ptr_to_clmem_unmap(cl_gpu, (void*)partsum);
	cl_mem buffer_O = ptr_to_clmem_unmap(cl_gpu, (void*)O);

	clSetKernelArg(cl_gpu->kernel, 0, sizeof(cl_mem), &buffer_I);
	clSetKernelArg(cl_gpu->kernel, 1, sizeof(cl_mem), &buffer_W);
	clSetKernelArg(cl_gpu->kernel, 2, sizeof(cl_mem), &buffer_partsum);
	clSetKernelArg(cl_gpu->kernel, 3, sizeof(cl_mem), &buffer_O);
	clSetKernelArg(cl_gpu->kernel, 4, sizeof(int), &i_offset);
	clSetKernelArg(cl_gpu->kernel, 5, sizeof(int), &w_offset);
	clSetKernelArg(cl_gpu->kernel, 6, sizeof(int), &o_offset);

	err = clEnqueueNDRangeKernel(cl_gpu->queue, cl_gpu->kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	if(err != CL_SUCCESS)
		print_error_message(err);

	err = clFinish(cl_gpu->queue);
	if(err != CL_SUCCESS)
		print_error_message(err);

	I = ptr_to_clmem_map(cl_gpu, (void*)I);
	W = ptr_to_clmem_map(cl_gpu, (void*)W);
	partsum = ptr_to_clmem_map(cl_gpu, (void*)partsum);
	O = ptr_to_clmem_map(cl_gpu, (void*)O);

	err = clFinish(cl_gpu->queue);
	if(err != CL_SUCCESS)
		print_error_message(err);
}

void tile_conv(cl_param* cl_gpu, float* I, float* W, float* partsum, float* O, config* data, tile_param* tile, size_t* global_work_size, size_t* local_work_size)
{
	for(int oh = 0; (oh+(tile->tr)-1) < data->output_size; oh += (tile->tr))
	{
		for(int ow = 0; (ow+(tile->tc)-1) < data->output_size; ow += (tile->tc))
		{
			for(int ic = 0; ic < data->input_c; ic += (tile->tn))
			{
				for(int oc = 0; oc < data->output_c; oc += (tile->tm))
				{
					float* i_base_addr = (I + ic*(data->input_size)*(data->input_size) + oh*(data->input_size) + ow);
					float* w_base_addr = (W + oc*(data->weight_size)*(data->weight_size)*data->input_c + ic*(data->weight_size)*(data->weight_size));
					float* o_base_addr = (O + oc*data->output_size*data->output_size + oh*data->output_size + ow);

					int i_offset = ((uint64_t)i_base_addr - (uint64_t)I)/4;
					int w_offset = ((uint64_t)w_base_addr - (uint64_t)W)/4;
					int o_offset = ((uint64_t)o_base_addr - (uint64_t)O)/4;

					// printf("i_offset=%d\tw_offset=%d\to_offset=%d\n", i_offset, w_offset, o_offset);
					// printf("oh=%d\tow=%d\tic=%d\toc=%d\n", oh, ow, ic, oc);

					run_gpu_program(cl_gpu, global_work_size, local_work_size, I, W, partsum, O, i_offset, w_offset, o_offset);
				}
			}
		}
	}
}

void direct_conv(cl_param* cl_gpu, float* I, float* W, float* partsum, float* O, config* data, size_t* global_work_size, size_t* local_work_size)
{
	run_gpu_program(cl_gpu, global_work_size, local_work_size, I, W, partsum, O, 0, 0, 0);
}

void clean_objects(cl_param* cl_gpu, float* I, float* W, float* O)
{
	cl_int err;

	cl_mem buffer_I = ptr_to_clmem_unmap(cl_gpu, (void*)I);
	err = clReleaseMemObject(buffer_I);
	if(err != CL_SUCCESS)
		print_error_message(err);
	cl_mem buffer_W = ptr_to_clmem_unmap(cl_gpu, (void*)W);
	err = clReleaseMemObject(buffer_W);
	if(err != CL_SUCCESS)
		print_error_message(err);
	cl_mem buffer_O = ptr_to_clmem_unmap(cl_gpu, (void*)O);
	err = clReleaseMemObject(buffer_O);
	if(err != CL_SUCCESS)
		print_error_message(err);

	err = clReleaseKernel(cl_gpu->kernel);
	if(err != CL_SUCCESS)
		print_error_message(err);

	err = clReleaseProgram(cl_gpu->program);
	if(err != CL_SUCCESS)
		print_error_message(err);

	err = clReleaseCommandQueue(cl_gpu->queue);
	if(err != CL_SUCCESS)
		print_error_message(err);

	err = clReleaseContext(cl_gpu->context);
	if(err != CL_SUCCESS)
		print_error_message(err);
}