#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <time.h>

using namespace std;

// AlexNet_CONV2 : {31, 48, 5, 27, 256};
// AlexNet_CONV3 : {15, 256, 3, 13, 384};
// AlexNet_CONV4 : {15, 192, 3, 13, 384};
// AlexNet_CONV5 : {15, 192, 3, 13, 256};

// VGG_CONV1 : {226, 3, 3, 224, 64}
// VGG_CONV2 : {226, 64, 3, 224, 64}
// VGG_CONV3 : {114, 64, 3, 112, 128}
// VGG_CONV4 : {114, 128, 3, 112, 128}
// VGG_CONV5 : {58, 128, 3, 56, 256}
// VGG_CONV6&CONV7 : {58, 256, 3, 56, 256}
// VGG_CONV8 : {30, 256, 3, 28, 512}
// VGG_CONV9&CONV10 : {30, 512, 3, 28, 512}
// VGG_CONV11&CONV12&CONV13 : {16, 512, 3, 14, 512}

int conv_config[5] = {15, 192, 3, 13, 256};
int tile[4] = {0, 0, 0, 0};

void initialization(int* rd, uint64_t* num, int size)
{
	for(int i = 0; i < size; i++)
	{
		*(rd+i) = 1000000;
		*(num+i) = 1000000;
	}
}

// oc -> ic -> oh -> ow -> kh -> kw
void get_tile_inside_rd_IR(int* tile, int loop_type, char reuse_type, int* rd)
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
		if(loop_type == 1)
		{
			*(rd+1) = conv_config[2]*conv_config[2] + 
					  conv_config[2]*conv_config[2] + 
					  1;
		}
		else if((loop_type == 2) || (loop_type == 5))
		{
			*(rd+1) = 3;
		}
		else if((loop_type == 3) || (loop_type == 4))
		{
			*(rd+1) = conv_config[2]*conv_config[2]*tile[2] + 
					  conv_config[2]*conv_config[2]*tile[2] + 
					  1;
		}
		else if(loop_type == 6)
		{
			*(rd+1) = tile[2] + tile[2] + 1;
		}
	}
	else if(reuse_type == 'O')
	{
		if((loop_type == 1) || (loop_type == 2))
		{
			int t_isize = tile[0]-1+conv_config[2];
			*(rd+2) = t_isize*t_isize + 
					  conv_config[2]*conv_config[2] + 
					  tile[0]*tile[1];
		}
		else if((loop_type == 3) || (loop_type == 4) || (loop_type == 6))
		{
			*(rd+2) = 3;
		}
		else if(loop_type == 5)
		{
			int t_isize = tile[0]-1+conv_config[2];
			*(rd+2) = t_isize*t_isize + 1 + tile[0]*tile[1];
		}
	}
}

void get_access_number(int* tile, char two_type, char data_type, uint64_t* num)
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
			*(num) = (conv_config[4]-1)-(*(num+3));
		else if(data_type == 'W')
			*(num+1) = (conv_config[3]*conv_config[3]-1)-(*(num+4));
		else if(data_type == 'O')
			*(num+2) = (conv_config[1]-1)-(*(num+5));
	}
}
bool tile_smaller_cache(int tr, int tc, int tn, int tm, int k_size, int cache_size)
{
	int tile_I = (tr-1+k_size) * (tc-1+k_size) * tn;
	int tile_W = k_size * k_size * tn * tm;
	int tile_O = tr * tc * tm;

	return ((tile_I + tile_W + tile_O) < cache_size);
}

void IR(int* tile, int* rd, uint64_t* num)
{
	int t_isize = tile[0]-1+conv_config[2];

	// reuse distance
	get_tile_inside_rd_IR(tile, 1, 'I', rd);
	get_tile_inside_rd_IR(tile, 1, 'W', rd);
	get_tile_inside_rd_IR(tile, 1, 'O', rd);
	*(rd+3) = -1;
	*(rd+4) = t_isize*t_isize*conv_config[1] + 
			  conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4] + 
			  tile[0]*tile[1]*conv_config[4];
	*(rd+5) = t_isize*t_isize*tile[2] + 
			  conv_config[2]*conv_config[2]*tile[2]*conv_config[4] + 
			  tile[0]*tile[1]*conv_config[4];

	for(int i = 0; i < 6; i++)
		*(rd+i) = *(rd+i) / 16;

	// number
	*(num+3) = 0;
	get_access_number(tile, 'o', 'W', num);
	get_access_number(tile, 'o', 'O', num);
	get_access_number(tile, 'i', 'I', num);
	get_access_number(tile, 'i', 'W', num);
	get_access_number(tile, 'i', 'O', num);
	*(num+4) = (*(num+4))*conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4];
	*(num+5) = (*(num+5))*conv_config[3]*conv_config[3]*conv_config[4];
	*(num) = (*(num))*conv_config[0]*conv_config[0]*conv_config[1];
	*(num+1) = (*(num+1))*conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4];
	*(num+2) = (*(num+2))*conv_config[3]*conv_config[3]*conv_config[4];

	// print
	for(int i = 0; i < 6; i++)
	{
		printf("rd=%d\tnum=%lu\n", *(rd+i), *(num+i));
	}
}

void OR(int* tile, int* rd, uint64_t* num)
{
	int t_isize = tile[0]-1+conv_config[2];

	// reuse distance
	*(rd) = t_isize*t_isize*tile[2] + 
			conv_config[2]*conv_config[2]*tile[2] + 
			tile[0]*tile[1];
	*(rd+1) = conv_config[2]*conv_config[2] + 
			  conv_config[2]*conv_config[2] + 
			  1;
	*(rd+2) = t_isize*t_isize + 
			  conv_config[2]*conv_config[2] + 
			  tile[0]*tile[1];
	*(rd+3) = t_isize*t_isize*conv_config[1] + 
			  conv_config[2]*conv_config[2]*conv_config[1]*tile[3] + 
			  tile[0]*tile[1]*tile[3];
	*(rd+4) = t_isize*t_isize*conv_config[1] + 
			  conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4] + 
			  tile[0]*tile[1]*conv_config[4];
	*(rd+5) = t_isize*t_isize*tile[2] + 
			  conv_config[2]*conv_config[2]*tile[2]*tile[3] + 
			  tile[0]*tile[1]*tile[3];

	// number
	get_access_number(tile, 'o', 'I', num);
	get_access_number(tile, 'o', 'W', num);
	get_access_number(tile, 'o', 'O', num);
	get_access_number(tile, 'i', 'I', num);
	get_access_number(tile, 'i', 'W', num);
	get_access_number(tile, 'i', 'O', num);
	*(num) = (*(num))*conv_config[0]*conv_config[0]*conv_config[1];
	*(num+1) = (*(num+1))*conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4];
	*(num+2) = (*(num+2))*conv_config[3]*conv_config[3]*conv_config[4];
	*(num+3) = (*(num+3))*conv_config[0]*conv_config[0]*conv_config[1];
	*(num+4) = (*(num+4))*conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4];
	*(num+5) = (*(num+5))*conv_config[3]*conv_config[3]*conv_config[4];

	// print
	for(int i = 0; i < 6; i++)
	{
		printf("rd=%d\tnum=%lu\n", *(rd+i), *(num+i));
	}
}

void WR(int* tile, int* rd, uint64_t* num)
{
	int t_isize = tile[0]-1+conv_config[2];

	// reuse distance
	*(rd) = t_isize*t_isize*tile[2] + 
			conv_config[2]*conv_config[2]*tile[2] + 
			tile[0]*tile[1];
	*(rd+1) = conv_config[2]*conv_config[2] + 
			  conv_config[2]*conv_config[2] + 
			  1;
	*(rd+2) = t_isize*t_isize + 
			  conv_config[2]*conv_config[2] + 
			  tile[0]*tile[1];
	*(rd+3) = conv_config[0]*conv_config[0]*conv_config[1] + 
			  conv_config[2]*conv_config[2]*conv_config[1]*tile[3] + 
			  conv_config[3]*conv_config[3]*tile[3];
	*(rd+4) = t_isize*t_isize*tile[2] + 
			  conv_config[2]*conv_config[2]*tile[2]*tile[3] + 
			  tile[0]*tile[1]*tile[3];
	*(rd+5) = conv_config[0]*conv_config[0]*tile[2] + 
			  conv_config[2]*conv_config[2]*tile[2]*tile[3] + 
			  conv_config[3]*conv_config[3]*tile[3];

	// number
	get_access_number(tile, 'o', 'I', num);
	get_access_number(tile, 'o', 'W', num);
	get_access_number(tile, 'o', 'O', num);
	get_access_number(tile, 'i', 'I', num);
	get_access_number(tile, 'i', 'W', num);
	get_access_number(tile, 'i', 'O', num);
	*(num) = (*(num))*conv_config[0]*conv_config[0]*conv_config[1];
	*(num+1) = (*(num+1))*conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4];
	*(num+2) = (*(num+2))*conv_config[3]*conv_config[3]*conv_config[4];
	*(num+3) = (*(num+3))*conv_config[0]*conv_config[0]*conv_config[1];
	*(num+4) = (*(num+4))*conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4];
	*(num+5) = (*(num+5))*conv_config[3]*conv_config[3]*conv_config[4];

	// print
	for(int i = 0; i < 6; i++)
	{
		printf("rd=%d\tnum=%lu\n", *(rd+i), *(num+i));
	}
}

void compute_hit_miss(int* rd, uint64_t* num, int cache_size)
{
	uint64_t hit_sum = 0;
	uint64_t miss_sum = 0;

	for(int i = 0; i < 6; i++)
	{
		if(*(rd+i) < cache_size)
		{
			hit_sum += *(num+i);
		}
		else
		{
			miss_sum += *(num+i);
		}
	}

	int tile_I = (tile[0]-1+conv_config[2]) * (tile[1]-1+conv_config[2]) * tile[2];
	int tile_W = conv_config[2] * conv_config[2] * tile[2] * tile[3];
	int tile_O = tile[0] * tile[1] * tile[3];
	int tile_size = tile_I + tile_W + tile_O;

	printf("============> tile_size = %d\thit_sum = %lu\tmiss_sum = %lu\n", tile_size, hit_sum, miss_sum);
}

void run()
{
	int* rd = (int*)malloc(6 * sizeof(int));
	uint64_t* num = (uint64_t*)malloc(6 * sizeof(uint64_t));
	int cache_block_size = 4/4;
	int cache_size = /*256*/7*1024/4;
	int count = 1;

	printf("on-chip buffer can hold %d data\n\n", (cache_size/cache_block_size));

	for(int tr = 1; tr <= conv_config[3]; tr++)
	{
		if((conv_config[3]%tr) == 0)
		{
				for(int tn = 1; tn <= conv_config[1]; tn++)
				{
					if((conv_config[1]%tn) == 0)
					{
						for (int tm = 1; tm <= conv_config[4]; tm++)
						{
							if((conv_config[4]%tm) == 0)
							{
								if(tile_smaller_cache(tr, tr, tn, tm, conv_config[2], cache_size))
								{
									initialization(rd, num, 6);
									tile[0] = tr;
									tile[1] = tr;
									tile[2] = tn;
									tile[3] = tm;
									printf("%d\t<tr, tc, tn, tm> = <%d, %d, %d, %d>\n", count, tr, tr, tn, tm);
									WR(&tile[0], rd, num);
									compute_hit_miss(rd, num, cache_size/cache_block_size);
									printf("\n");
									count++;
								}
							}
						}
					}
				}
		}
	}


	free(num);
	free(rd);
}

int main(int argc, char* argv[])
{
	// clock_t start, end;
	// start = clock();
	run();
	// end = clock();
	// printf("The time = %ld ms\n", (end-start)*1000000000/CLOCKS_PER_SEC);
	return 0;
}