import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

/**
 * Created by Wayne.A.Z on 2020-06-26.
 */


public class Solution_LinkedList {
    public static class ListNode { // 静态类 才能在下面的额静态方法中被访问
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    public static int[] stringToIntegerArray(String input) {
        input = input.trim();
        input = input.substring(1, input.length() - 1);
        if (input.length() == 0) {
            return new int[0];
        }

        String[] parts = input.split(",");
        int[] output = new int[parts.length];
        for(int index = 0; index < parts.length; index++) {
            String part = parts[index].trim();
            output[index] = Integer.parseInt(part);
        }
        return output;
    }
    public static ListNode stringToListNode(String input) {
        // Generate array from the input
        int[] nodeValues = stringToIntegerArray(input);

        // Now convert that list into linked list
        ListNode dummyRoot = new ListNode(0);
        ListNode ptr = dummyRoot;
        for(int item : nodeValues) {
            ptr.next = new ListNode(item);
            ptr = ptr.next;
        }
        return dummyRoot.next;
    }
    public static void prettyPrintLinkedList(ListNode node) {
        while (node != null && node.next != null) {
            System.out.print(node.val + "->");
            node = node.next;
        }

        if (node != null) {
            System.out.println(node.val);
        } else {
            System.out.println("Empty LinkedList");
        }
    }
    // 24
    public ListNode swapPairs(ListNode head) {
        if(head == null) return head;
        ListNode dummy = new ListNode(-1);
        ListNode prev = dummy, first = head, second, next;
        while(first != null && first.next != null){
            second = first.next;
            next = second.next;

            first.next = second.next;
            second.next = next;
            prev.next = second;

            prev = first;
            first = next;
            System.out.println(first.val);
        }
        return dummy.next;
    }
    // 100163
//    public ListNode removeDuplicateNodes(ListNode head) {
//        ListNode cur = head.next;
//        ListNode res = new ListNode(head.val);
//        ListNode resHead = res;
//
//        while(cur != null){
//            ListNode before = head;
//
//            while(before.next != cur){
//                before = before.next;
//                if(before.val == cur.val) {
//                    break;
//                }
//            }
//            System.out.println(cur.val);
//            res.next = cur;
//            res = res.next;
//            cur = cur.next;
//        }
//        return resHead;
//    }

    public ListNode removeDuplicateNodes(ListNode head) {
        Set<Integer> set = new HashSet<>(); // 去重
        ListNode cur = head;
        while(cur != null){
            set.add(cur.val);
            if(cur.next != null && set.contains(cur.next.val)){
                cur.next = cur.next.next;
            }else{
                cur = cur.next;
            }
        }
        return head;
    }

    // 148
//    public ListNode sortList(ListNode head) {
//        if(head == null || head.next == null) return head;
//        ListNode fast = head.next, slow = head;
//        while(fast != null && fast.next != null){
//            fast = fast.next.next;
//            slow = slow.next;
//        }
//        ListNode mid = slow.next;
//        slow.next = null;
//        ListNode left = sortList(head);
//        ListNode right = sortList(mid);
//        ListNode dummyHead = new ListNode(0);
//        ListNode cur = dummyHead;
//        while(left != null && right != null){
//            if(left.val < right.val){
//                cur.next = left;
//                left = left.next;
//            }else{
//                cur.next = right;
//                right = right.next;
//            }
//            cur = cur.next;
//        }
//        cur.next = left != null ? left : right;
//        return dummyHead.next;
//    }
    public ListNode sortList(ListNode head) {
        if(head == null) return head;
        ListNode fast = head, slow = head;
        while(fast.next != null && fast.next.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode mid = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(mid);

        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;

        while(left != null && right != null){
            if(left.val < right.val){
                cur = left;
                left = left.next;
            }
            else{
                cur = right;
                right = right.next;
            }
            cur = cur.next;
        }
        cur.next = left!= null ? left : right;
        System.out.println(cur);
        return dummy.next;
    }

    // 147
//    public ListNode insertionSortList(ListNode head) {
//        ListNode prev = head, cur = head, dummy = new ListNode(Integer.MIN_VALUE);
//        dummy.next = head;
//        while(cur != null){
//            if(prev.val <= cur.val) {
//                prev = cur;
//                cur = cur.next;
//            }
//            else{
//                ListNode p = dummy;
//                while(p != prev && p.next.val < cur.val){
//                    p = p.next;
//                }
//                // prev始终指向已排序链表末尾
//                prev.next = cur.next;
//                // cur 插入到p和p.next之间
//                cur.next = p.next;
//                p.next = cur;
//
//                cur = prev.next;
//            }
//        }
//        return dummy.next;
//    }
    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = head;
        while(head != null){
            head = dummy;
            while(cur.next != null && cur.next.val > head.val) cur = cur.next;
            head.next = cur.next;
            cur.next = head;
        }
        return dummy.next;
    }
    // 326
    public ListNode oddEvenList(ListNode head) {
        if(head == null) return null;
        ListNode odd = head, even = head.next;
        ListNode evenHead = even;
        while(even!= null && odd != null){
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }
    // 61
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || head.next == null || k == 0) return head;
        ListNode tail, newTail = head, newHead = head;
        int len = 1;
        ListNode cur = head;
        while(cur.next != null){
            cur = cur.next;
            len ++;
        }
        tail = cur;

        if(k % len == 0) return head;//

        int cnt = 0;

        while(cnt < len - k % len - 1){
            newTail = newTail.next;
            cnt ++;
        }
        newHead = newTail.next;
        tail.next = head;
        newTail.next = null;
        return newHead;
    } // k = 0
//    public ListNode rotateRight(ListNode head, int k) {
//        // base cases
//        if (head == null) return null;
//        if (head.next == null) return head;
//
//        // close the linked list into the ring
//        ListNode old_tail = head;
//        int n;
//        for(n = 1; old_tail.next != null; n++)
//            old_tail = old_tail.next;
//        old_tail.next = head;
//
//
//        ListNode new_tail = head;
//        for (int i = 0; i < n - k % n - 1; i++)
//            new_tail = new_tail.next;
//        ListNode new_head = new_tail.next;
//
//        // break the ring
//        new_tail.next = null;
//
//        return new_head;
//    }

    // 25
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy, end = dummy;
        while(end.next != null){
            for(int i = 0;i < k && end != null;i ++) end = end.next;
            if(end == null) break;
            ListNode start = prev.next;
            ListNode next = end.next;
            end.next = null;
            prev.next = reverse(start);
            start.next = next;
            prev = start;

            end = prev;

        }
        return dummy.next;
    }
    public ListNode reverse(ListNode head){
        if(head == null || head.next == null) return head; // !
        ListNode prev = null, cur = head, next;
        while(cur != null){
            next = cur.next;
            cur.next = prev; // 引用传递，会同时改变next的值
            prev = cur;
            cur = next;
        }
        prettyPrintLinkedList(prev);
        return prev;
    }
    // 86
    public ListNode partition(ListNode head, int x) {

        ListNode dummy1 = new ListNode(-1);
        ListNode dummy2 = new ListNode(-1);
        ListNode smaller = dummy1;
        ListNode larger = dummy2;
        while(head != null){
            if(head.val < x){
                smaller.next = head;
                smaller = smaller.next;
            }else{
                larger.next = head;
                larger = larger.next;
            }
            head = head.next;
        }
        larger.next = null;
        smaller.next = dummy2.next;
        return dummy1.next;
    }
    // 725
    public ListNode[] splitListToParts(ListNode root, int k) {
        ListNode p = root;
        int len = 0;
        while(p != null){
            len += 1;
            p = p.next;
        }

        ListNode[] res = new ListNode[k];
        int l = len / k + 1;
        p = root;
        int i = 0;
        while(p != null){
            ListNode newDummy = new ListNode(-1);
            newDummy.next = p;
            ListNode newRoot = newDummy, newTail = newDummy;
            int m = 0;
            while(m < l){
                p = p.next;
                newTail = p;
                m ++;
            }
            newTail.next = null;
            res[i] = newDummy.next;
            p = p.next;
        }
        return res;
    }


    public static void main(String[] args) {
        Solution_LinkedList s = new Solution_LinkedList();
//        ListNode head = s.stringToListNode("[1, 2, 3, 3,3,2, 1]");
//        s.prettyPrintLinkedList(s.removeDuplicateNodes(head));
        ListNode head = s.stringToListNode("[1, 2, 3, 4]");
        s.prettyPrintLinkedList(s.swapPairs(head));

//        ListNode head = s.stringToListNode("[4,2,1,3]");
//        s.prettyPrintLinkedList(s.sortList(head));
//        ListNode head = s.stringToListNode("[4,2,1,3]");
//        s.prettyPrintLinkedList(s.insertionSortList(head));
//        ListNode head = s.stringToListNode("[1,2,3,4,5,6]");
//        s.prettyPrintLinkedList(s.oddEvenList(head));
//        ListNode head = s.stringToListNode("[0, 1, 2]");
//        s.prettyPrintLinkedList(s.rotateRight(head, 4));
//        ListNode head = s.stringToListNode("[1, 2, 3, 4, 5]");
//        s.prettyPrintLinkedList(s.rotateRight(head, 2));
//        ListNode head = s.stringToListNode("[1, 2]");
//        s.prettyPrintLinkedList(s.rotateRight(head, 0));
//        ListNode head = s.stringToListNode("[1, 2, 3, 4, 5]");
//        s.prettyPrintLinkedList(s.reverseKGroup(head, 3));
//        ListNode head = s.stringToListNode("[1,4,3,2,5,2]");
//        s.prettyPrintLinkedList((s.partition(head, 3)));
//        ListNode head = s.stringToListNode("[1,2,3,4]");
//        ListNode[] reses = s.splitListToParts(head, 5);
//        for(ListNode res: reses) s.prettyPrintLinkedList(res);


    }
}
