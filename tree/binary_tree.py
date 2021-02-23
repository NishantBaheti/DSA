"""package for binary tree
"""


class Node:
    """A node in binary tree
    """

    def __init__(self, data):
        """
        """
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:
    """Binary Tree
    """

    def __init__(self):
        """
        """
        self._started = None
        self.root = None
        self._last_insert_point = None

    def start_tree(self,root_node_data):
        if root_node_data is not None:
            self.root = Node(root_node_data)
            self._started = True
        else:
            raise ValueError("root node data cant be None")     
        
    
    def _inorder_traverse(self, node, values=[]):
        """
        """
        if node:
            data = node.data
            left = node.left
            right = node.right

            self._inorder_traverse(left, values)
            values.append(data)
            self._inorder_traverse(right, values)
        return values

    def _preorder_traverse(self, node, values=[]):
        """
        """
        if node:
            data = node.data
            left = node.left
            right = node.right

            values.append(data)
            self._preorder_traverse(left, values)
            self._preorder_traverse(right, values)
        return values

    def _postorder_traverse(self, node, values=[]):
        """
        """
        if node:
            data = node.data
            left = node.left
            right = node.right

            self._postorder_traverse(left, values)
            self._postorder_traverse(right, values)
            values.append(data)
        return values

    def traverse_tree(self, order="in"):
        order = order.lower()
        if order == "in":
            values = self._inorder_traverse(node=self.root, values=[])
        elif order == "pre":
            values = self._preorder_traverse(node=self.root, values=[])
        elif order == "post":
            values = self._postorder_traverse(node=self.root, values=[])
        else:
            raise ValueError(f"order : '{order}' is not defined.")

        return values


if __name__ == "__main__":

    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.left = Node(6)
    root.right.right = Node(7)

    print(root.traverse_tree(order="pre"))
