#include <stdio.h>
#include <stdlib.h>

/*************************************************
 * Linked list node structure.
 *************************************************/
typedef struct ListNode
{
    int data;
    struct ListNode *next;
} ListNode;

/*******************************************************
 * traverse linked list fuction.
 * @param ptr: pointer to the node in the linked list.
 ******************************************************/
void traverse_linked_list(ListNode *ptr)
{
    while (ptr != NULL)
    {
        printf("Element : %d \n", ptr->data);
        ptr = ptr->next;
    };
    printf("*----------------------------------------------------*\n");
}

/*********************************************************************************************
 * Insert element at the first position of the linked list.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * @param element: element that needs to be inserted at the first.
 * 
 * @returns new_head: new head node pointer of the linked list type of ListNode.
 * 
 ********************************************************************************************/
ListNode *insert_at_first(ListNode *head, int element)
{
    ListNode *new_head = (ListNode *)malloc(sizeof(ListNode));

    new_head->data = element;
    new_head->next = head;
    return new_head;
}

/*********************************************************************************************
 * Insert element at the last position of the linked list.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * @param element: element that needs to be inserted at the last.
 * 
 * @returns new_head: new head node pointer of the linked list type of ListNode.
 * 
 ********************************************************************************************/
ListNode *insert_at_last(ListNode *head, int element)
{
    ListNode *new_node = (ListNode *)malloc(sizeof(ListNode));
    new_node->data = element;
    new_node->next = NULL;
    ListNode *p = head;

    while (p->next != NULL)
    {
        p = p->next;
    }

    p->next = new_node;

    return head;
}

/*******************************************************************************
 * insert element at a index.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * @param element: element that needs to be inserted at the first.
 * @param idx: index.
 * 
 * @return head: head of the linked list.
 ******************************************************************************/
ListNode *insert_at_index(ListNode *head, int element, int idx)
{
    ListNode *new_node = (ListNode *)malloc(sizeof(ListNode));
    ListNode *p = head;
    int cursor = 1;

    while (cursor <= (idx - 1))
    {

        if (p->next == NULL)
        {
            break;
        }
        p = p->next;
        cursor++;
    }
    new_node->data = element;
    new_node->next = p->next;
    p->next = new_node;
    return head;
}

/*********************************************************************************************
 * delete element at the first position of the linked list.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * 
 * @returns head: new head node pointer of the linked list type of ListNode.
 * 
 ********************************************************************************************/
ListNode *delete_at_first(ListNode *head)
{
    ListNode *t_node = head;
    head = head->next;
    free(t_node);
    return head;
}

/*********************************************************************************************
 * delete element at the last position of the linked list.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * 
 * @returns head: new head node pointer of the linked list type of ListNode.
 ********************************************************************************************/
ListNode *delete_at_last(ListNode *head)
{
    ListNode *p = head;
    ListNode *q = head->next;

    while (q->next != NULL)
    {
        p = p->next;
        q = q->next;
    }

    p->next = NULL;
    free(q);
    return head;
}

/*********************************************************************************************
 * delete element at the indexed position of the linked list.
 * 
 * @param head: head pointer of the ListNode type repr of the linked list.
 * @param idx: index
 * 
 * @returns head: new head node pointer of the linked list type of ListNode.
 ********************************************************************************************/
ListNode *delete_at_index(ListNode *head, int idx)
{
    ListNode *t1 = head;
    ListNode *t2 = head->next;

    int cursor = 1;
    while (cursor <= (idx - 1))
    {
        if (t2->next == NULL){
            break;
        }
        t1 = t1->next;
        t2 = t2->next;
        cursor++;
    }

    t1->next = t2->next;
    free(t2);
    return head;
}

int main()
{
    ListNode *head = (ListNode *)malloc(sizeof(ListNode));
    ListNode *second = (ListNode *)malloc(sizeof(ListNode));
    ListNode *third = (ListNode *)malloc(sizeof(ListNode));

    head->data = 7;
    head->next = second;

    second->data = 10;
    second->next = third;

    third->data = 14;
    third->next = NULL;

    traverse_linked_list(head);

    head = insert_at_first(head, 3);
    traverse_linked_list(head);

    head = insert_at_index(head, 13, 2);
    traverse_linked_list(head);

    head = insert_at_last(head, 100);
    traverse_linked_list(head);

    head = delete_at_first(head);
    traverse_linked_list(head);

    head = delete_at_last(head);
    traverse_linked_list(head);

    head = delete_at_index(head,1);
    traverse_linked_list(head);

    return 0;
}