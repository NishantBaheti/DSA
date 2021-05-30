#include "stdio.h"
#include "stdlib.h"

/**
 * Print array elements
 * 
 * @param arr array pointer
 * @n size of array
 * **/
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

/**
 * Bubble sort
 * 
 * @note
 * 1. in every pass compare adjescent elements and sort them out.
 * 
 * @param arr array pointer
 * @param n size of the array
 * 
 **/
void bubble_sort(int *arr, int n)
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
    print_array(arr, n);
}

/**
 * Bubble sort optimized method
 * 
 * @note
 * Breaking early if any of the pass is not needed to be sorted.
 * 
 * @param arr array pointer
 * @param n size of the array
 * 
 **/
void bubble_sort_opt(int *arr, int n)
{
    int temp;
    int is_sorted = 0;
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

        if (is_sorted)
        {
            printf("breaking early");
            return;
        }
    }
    print_array(arr, n);
}

/**
 * Selection sort algorithm
 * 
 * @note
 * select minimum element and replace it with current index element 
 * 
 * @param arr array pointer
 * @param n size of array
 ***/
void selection_sort(int *arr, int n)
{
    int temp, min;
    for (int i = 0; i < n - 1; i++)
    {
        print_array(arr, n);
        // find minimum element's index
        min = i;
        for (int j = i + 1; j < n; j++)
        {
            if (arr[min] > arr[j])
            {
                min = j;
            }
        }
        // select minimum element and replace it with current index element
        temp = arr[i];
        arr[i] = arr[min];
        arr[min] = temp;
    }
    print_array(arr, n);
}

/**
 * Insertion sort algorithm
 * 
 * @note
 * select element in the index range and insert it at the index at sorted position
 * 
 * @param arr array pointer
 * @param n size of array
 ***/
void insertion_sort(int *arr, int n)
{
    int ins_value, i, j;
    for (i = 1; i < n; i++)
    {
        print_array(arr, n);
        ins_value = arr[i]; // store the element of current position
        j = i; 
        // iterate to prev indices to see if there is an element that is 
        // smaller than the insertion element
        
        // while loop will break either at the 0th index 
        // or it reaches a smaller element
        while (arr[j - 1] > ins_value && j >= 1)
        {
            arr[j] = arr[j - 1];
            j--;
        }
        // insert at the index 
        arr[j] = ins_value;
    }

    print_array(arr, n);
}


void shell_sort(int *arr, int n)
{
    
}



int main()
{
    int a[] = {3, 33, 512, 5, 64};
    int n = sizeof(a) / sizeof(int);

    // bubble_sort(a, n);
    // bubble_sort_opt(a, n);
    // selection_sort(a,n);

    insertion_sort(a, n);

    return 0;
}