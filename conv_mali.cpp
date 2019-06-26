#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

cl_program read_cl_file(char* file_name)
{
	fstream file;
	file.read(file_name, ios::in);
	cl_program program;
	program = clCreateProgramWithSource(context, );
} 

bool setup_gpu()
{
	cl_int err;
	cl_uint num;
	err = clGetPlatformIDs(0, NULL, &num);
	if(err != CL_SUCCESS)
	{
		printf("clGetPlatformIDs() Error !!!\n");
		cout << "Error Message :" << err << endl;
		return false;
	}
	cout << "Platform number :" << num << endl;

	vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], NULL);
	if(err != CL_SUCCESS)
	{
		printf("clGetPlatformIDs() Error !!!\n");
		cout << "Error Message :" << err << endl;
		return false;
	}

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 0, NULL, &num);
	if(err != CL_SUCCESS)
	{
		printf("clGetDeviceIDs() Error !!!\n");
		cout << "Error Message :" << err << endl;
		return false;
	}
	cout << "Device number :" << num << endl;

	vector<cl_device_id> devices(num);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, num, &devices[0], NULL);
	if(err != CL_SUCCESS)
	{
		printf("clGetDeviceIDs() Error !!!\n");
		cout << "Error Message :" << err << endl;
		return false;
	}

	for(int i = 0; i < num; i++)
	{
		size_t str_num;
		string devname;
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &str_num);
		devname.resize(str_num);
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, str_num, &devname[0], NULL);
		cout << devname.c_str() << endl;
	}

	cl_context context;
	context = clCreateContext(NULL, num, &devices[0], NULL, NULL, &err);
	if(err != CL_SUCCESS)
	{
		printf("clCreateContext() Error !!!\n");
		cout << "Error Message :" << err << endl;
		return false;
	}

	cl_command_queue queue;
	queue = clCreateCommandQueue(context, devices[0], 0, &err);
	if(err != CL_SUCCESS)
	{
		printf("clCreateCommandQueue() Error !!!\n");
		cout << "Error Message :" << err << endl;
		return false;
	}

	cl_program program;
	program = read_cl_file();
}

int main(int argc, char* argv[])
{
	setup_gpu();
	return 0;
}