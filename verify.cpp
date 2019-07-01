#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>

using namespace std;

int conv_config[5] = {8, 8, 1, 8, 16};
int tile[4] = {2, 2, 4, 2};

void initialization(int* rd, int* num, int size)
{
	for(int i = 0; i < size; i++)
	{
		*(rd+i) = 1000000;
		*(num+i) = 1000000;
	}
}

void get_tile_inside_rd(int* tile, char reuse_type, int* rd)
{
	if(reuse_type == 'I')
	{
		int t_isize = tile[0]-1+conv_config[2];
		*(rd) = t_isize*t_isize*tile[2] + 
				conv_config[2]*conv_config[2]*tile[2] + 
				tile[0]*tile[1];
	}
	else if(reuse_type == 'W')
	{
		*(rd+1) = conv_config[2]*conv_config[2] + 
				  conv_config[2]*conv_config[2] + 
				  1;
	}
	else if(reuse_type == 'O')
	{
		int t_isize = tile[0]-1+conv_config[2];
		*(rd+2) = t_isize*t_isize + 
				  conv_config[2]*conv_config[2] + 
				  tile[0]*tile[1];
	}
}

void get_access_number(int* tile, char two_type, char data_type, int* num)
{
	int result;

	if(two_type == 'o')
	{
		if(data_type == 'I')
			*(num+3) = (conv_config[4]/tile[3])-1;
		else if(data_type == 'W')
			*(num+4) = (conv_config[3]/tile[0])*(conv_config[3]/tile[1])-1;
		else if(data_type == 'O')
			*(num+5) = (conv_config[1]/tile[2])-1;
	}
	else if(two_type == 'i')
	{
		if(data_type == 'I')
			*(num) = (conv_config[4]-1)-*(num+3);
		else if(data_type == 'W')
			*(num+1) = (conv_config[3]*conv_config[3]-1)-*(num+4);
		else if(data_type == 'O')
			*(num+2) = (conv_config[1]-1)-*(num+5);
	}
}

void IR(int* tile, int* rd, int* num)
{
	int t_isize = tile[0]-1+conv_config[2];

	// reuse distance
	get_tile_inside_rd(tile, 'I', rd);
	get_tile_inside_rd(tile, 'W', rd);
	get_tile_inside_rd(tile, 'O', rd);
	*(rd+3) = -1;
	*(rd+4) = t_isize*t_isize*conv_config[1] + 
			  conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4] + 
			  tile[0]*tile[1]*conv_config[4];
	*(rd+5) = t_isize*t_isize*tile[2] + 
			  conv_config[2]*conv_config[2]*tile[2]*conv_config[4] + 
			  tile[0]*tile[1]*conv_config[4];

	// number
	*(num+3) = 0;
	get_access_number(tile, 'o', 'W', num);
	get_access_number(tile, 'o', 'O', num);
	get_access_number(tile, 'i', 'I', num);
	*(num) = (*(num))*conv_config[0]*conv_config[0]*conv_config[1];
	get_access_number(tile, 'i', 'W', num);
	get_access_number(tile, 'i', 'O', num);
}

void run()
{
	int* rd = (int*)malloc(6 * sizeof(int));
	int* num = (int*)malloc(6 * sizeof(int));

	initialization(rd, num, 6);
	IR(&tile[0], rd, num);

	for(int i = 0; i < 6; i++)
	{
		printf("rd = %d\tnum = %d\n", *(rd+i), *(num+i));
	}

	free(num);
	free(rd);
}

int main(int argc, char* argv[])
{
	run(); 
	return 0;
}