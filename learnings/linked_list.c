#include <stdio.h>
#include <stdlib.h>

/************************************************
 * Linked list node structure
*************************************************/
typedef struct ListNode
{
    int data;
    struct ListNode *next;
} ListNode;

/************************************************
 * traverse linked list fuction
 * @param ptr pointer to the node in the linked list
 ***********************************************/
void traverse_linked_list(ListNode *ptr)
{
    while (ptr != NULL)
    {
        printf("Element : %d \n", ptr->data);
        ptr = ptr->next;
    };
}

/*********************************************************************************************
 * Insert element at the first position of the linked list
 * 
 * @param head head pointer of the ListNode type repr of the linked list
 * @param element element that needs to be inserted at the first
 * 
 * @returns new_head new head node pointer of the linked list type of ListNode
 * 
*********************************************************************************************/
ListNode *insert_at_first(ListNode *head, int element)
{
    ListNode *new_head;
    new_head = (ListNode *)malloc(sizeof(ListNode));

    new_head->data = element;
    new_head->next = head;
    return new_head;
}

int main()
{
    ListNode *head;
    ListNode *second;
    ListNode *third;

    head = (ListNode *)malloc(sizeof(ListNode));
    second = (ListNode *)malloc(sizeof(ListNode));
    third = (ListNode *)malloc(sizeof(ListNode));

    head->data = 7;
    head->next = second;

    second->data = 10;
    second->next = third;

    third->data = 14;
    third->next = NULL;

    traverse_linked_list(head);
    head = insert_at_first(head, 3);
    traverse_linked_list(head);
    return 0;
}