#include <stdio.h>

#define ARRAY_CAP 100

int delete_element(int arr[], int index, int size)
{
    /*
    * arr : array
    * index : index to which element needs to be inserted
    * size : size of the array
    */

    if (size <= 0 || size > ARRAY_CAP)
    {
        return -1;
    }
    else
    {

        for (int i = index; i <= size - 1; i++)
        {
            arr[i] = arr[i + 1];
        }
        return index;
    }
}

void dislay_elements(int arr[], int size)
{
    /*
    * arr : array
    * size : size of the array
    */
    for (int i = 0; i < size; i++)
    {
        printf("%d\n", arr[i]);
    }
}

int main()
{

    int a[ARRAY_CAP] = {1, 3, 4, 5, 7};
    int size = 5;

    int result = delete_element(a, 1, size);
    if (result != -1)
    {
        size -= 1;
    }
    dislay_elements(a, size);

    return 0;
}