#include "stdio.h"
#include "stdlib.h"

void print_array(int *arr, int n)
{
    printf("*-------------------------- Array -----------------------------*\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d\t", arr[i]);
    }
    printf("\n");
    printf("*--------------------------------------------------------------*\n");
}

void *bubble_sort(int *arr, int n)
{
    int temp;
    for (int pass = 0; pass < n; pass++)
    {

        for (int i = 0; i <= n - 1 - pass; i++)
        {
            printf("pass : %d element : %d\n", pass + 1, i + 1);
            print_array(arr, n);
            if (arr[i] > arr[i + 1])
            {
                temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
    }
}

void *bubble_sort_opt(int *arr, int n)
{
    int temp;
    int is_sorted =0;
    for (int pass = 0; pass < n; pass++)
    {
        is_sorted = 1;
        for (int i = 0; i <= n - 1 - pass; i++)
        {
            printf("pass : %d element : %d\n", pass + 1, i + 1);
            print_array(arr, n);
            if (arr[i] > arr[i + 1])
            {
                temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                is_sorted = 0;
            }
        }

        if(is_sorted){
            printf("breaking early");
            return NULL;
        }
    }
}

int main()
{
    int a[] = {3, 33, 512, 5, 64};
    int n = sizeof(a) / sizeof(int);

    // print_array(a, n);
    bubble_sort(a, n);
    // print_array(a,n);

    bubble_sort_opt(a, n);

    return 0;
}