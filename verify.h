#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>

void initialization(int* rd, uint64_t* num, int size);

// oc -> ic -> oh -> ow -> kh -> kw
void get_tile_inside_rd_IR1(int* tile, char reuse_type, int* rd);
// oc -> ic -> kh -> kw -> oh -> ow
void get_tile_inside_rd_IR2(int* tile, char reuse_type, int* rd);
// oc -> oh -> ow -> ic -> kh -> kw
void get_tile_inside_rd_IR3(int* tile, char reuse_type, int* rd);
// oc -> oh -> ow -> kh -> kw -> ic
void get_tile_inside_rd_IR4(int* tile, char reuse_type, int* rd);
// oc -> kh -> kw -> ic -> oh -> ow
void get_tile_inside_rd_IR5(int* tile, char reuse_type, int* rd);
// oc -> kh -> kw -> oh -> ow -> ic
void get_tile_inside_rd_IR6(int* tile, char reuse_type, int* rd);

void get_access_number(int* tile, char two_type, char data_type, uint64_t* num);

void IR(int* tile, int* rd, uint64_t* num);

bool greater50(int tr, int tc, int tn, int tm, int k_size, char reuse_type);
void run();