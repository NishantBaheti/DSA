#include <stdio.h>
#include <stdlib.h>

typedef struct MyArray
{
    /* 
    * total_size : total size of the array
    * used_size : used size of the array
    * ptr : pointer to the index address
    */
    int total_size;
    int used_size;
    int *ptr;
} Array;

void create_array(Array *a, int total_size, int used_size)
{
    /*
    * a : pointer the the array (when we want value of an address we use *)
    * total_size : total size to the array
    * used_size : size we want to use now    
    */

    // noob approach
    // (*a).total_size = total_size;
    // (*a).used_size = used_size;
    // (*a).ptr = (int *)malloc(total_size*sizeof(int));

    a->total_size = total_size;
    a->used_size = used_size;
    a->ptr = (int *)malloc(total_size * sizeof(int));
}

void show(Array *a)
{
    /*
    * a : pointer the the array (when we want value of an address we use *)
    */
    for (int i = 0; i < a->used_size; i++)
    {
        printf("%d\n", (a->ptr)[i]);
    }
}

void set_values(Array *a)
{
    /*
    * a : pointer the the array (when we want value of an address we use *)
    */
    int n;
    for (int i = 0; i < a->used_size; i++)
    {

        printf("Enter element for index: %d\n", i);
        scanf("%d", &n);
        (a->ptr)[i] = n;
    }
}

int main()
{
    Array marks;
    create_array(&marks, 20, 2); // passing the address of Array marks so pointer can catch it
    set_values(&marks);
    show(&marks);
    return 0;
}