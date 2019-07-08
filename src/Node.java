import java.util.LinkedList;
import java.util.Stack;

/**
 * Created by Wayne.A.Z on 2019-07-08.
 */

//public class Node {
//    public int value;
//    public Node next;
//
//    public Node(int value) {
//        this.value = value;
//    }
//
//    public Node addLastNode(int value) {
//        Node n = new Node(value);
//        return n;
//    }
//
//}
//
//    public static LinkedList insert(LinkedList list, int data)
//    {
//        Node new_node = new Node(data);
//        new_node.next = null;
//
//        // If the Linked List is empty,
//        // then make the new node as head
//        if (list.head == null) {
//            list.head = new_node;
//        }
//        else {
//            // Else traverse till the last node
//            // and insert the new_node there
//            Node last = list.head;
//            while (last.next != null) {
//                last = last.next;
//            }
//
//            // Insert the new_node at last node
//            last.next = new_node;
//        }
//
//        // Return the list by head
//        return list;
//    }
//
//    // Method to print the LinkedList.
//    public static void printList(LinkedList list)
//    {
//        Node currNode = list.head;
//
//        System.out.print("LinkedList: ");
//
//        // Traverse through the LinkedList
//        while (currNode != null) {
//            // Print the data at current node
//            System.out.print(currNode.data + " ");
//
//            // Go to next node
//            currNode = currNode.next;
//        }
//    }
//
//
//    //p51
//public  Node JosephusKill(Node head, int m){
//    if(head == null || head.next == head || m < 1){
//        return head;
//    }
//    Node last = head;
//    while(last.next  != head){
//        last = last.next;
//    }
//    int count = 0;
//    while(head != last){//
//        if(++count == m){
//            head = last.next;
//        }else{
//            last = last.next;
//        }
//        head = last.next;
//    }
//    return head;
//    }
//
////p56
//    public Boolean isPalindrome(Node head){
//        Stack<Node> s = new Stack<>();
//        Node cur = head;
//        while(cur != null){
//            s.push(cur);
//            cur = cur.next;
//        }
//        while( head.next != head){
//            if(s.pop().value != head.value){
//                return false;
//            }
//        }
//        return true;
//    }
//
//}