#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>

void initialization(int* rd, uint64_t* num, int size);

// oc -> ic -> oh -> ow -> kh -> kw
void get_tile_inside_rd(int* tile, char reuse_type, int* rd);

void get_access_number(int* tile, char two_type, char data_type, uint64_t* num);

void IR(int* tile, int* rd, uint64_t* num);

bool greater50(int tr, int tc, int tn, int tm, int k_size, char reuse_type);
void run();