#ifndef SPLAY_TREE_H
#define SPLAY_TREE_H
#include<iostream>
#include<stdint.h>
#include<map>
/*if l < r => return < 0*/
/*if l = r => return = 0*/
/*if l > r => return > 0*/
#define compare(l,r) ((l->time)-(r->time))

struct TREE_NODE {
    TREE_NODE *left, *right, *parent;
    uint64_t time;/* last access time*/
    uint64_t addr; /* access address */
    uint64_t size;/* maintained to be the number of nodes rooted here */
};

TREE_NODE* tree_create(uint64_t i_time,float* i_addr);
TREE_NODE* tree_rotate(TREE_NODE **check_node);//do rotation
uint64_t distance_compute(TREE_NODE **root, uint64_t f_time);
void tree_insert(TREE_NODE **root,uint64_t i_time, uint64_t i_addr);
void tree_delete(TREE_NODE **check_node);
void tree_print(TREE_NODE **t);
void tree_free(TREE_NODE **t);


#endif
