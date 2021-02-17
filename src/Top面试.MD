1. 340 至少有K个重复字符的最长子串
```java
// 双指针模板 + hashmap
class Solution {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        int res = 0;
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        for(int i = 0;i < s.length(); i ++){
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
            while(map.size() > k){
                if(map.get(s.charAt(left)) > 1) map.put(s.charAt(left), map.get(s.charAt(left)) - 1);
                else map.remove(s.charAt(left));
                left ++;
            }
            res = Math.max(res, i - left + 1);
        }
        return res;
    }
}
```
2. 285 BST的中序遍历后继节点
```java
class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        // 法1： 面试指南：stack辅助迭代中序遍历
        // 用pre保存前一个值，判断等于p,当前则为下一个值
        Deque<TreeNode> s = new ArrayDeque<>();
        int pre = Integer.MIN_VALUE;// 保存上一个值
        while(root != null || !s.isEmpty()){
            if(root != null){
                s.addLast(root);
                root = root.left;
            }else{
                root = s.removeLast();
                if(pre == p.val) return root;
                pre = root.val;
                root = root.right;   
            }
            
        }
        return null; // 如果没有找到
    }
    // 法2：递归
    // 更新 TODO
}
```
3. 277 最多调3 * n次know(a, b) 函数，找到名人；没有名人返回-1
```java
// 贪心
public class Solution extends Relation {
    public int findCelebrity(int n) {
        int res = 0;
        // 两两比较，排除其中一个
        for(int i = 0;i < n;i ++){
            if(knows(res, i)) res = i;
        }
        // 校验是名人的两个条件
        for(int i = 0;i < n;i ++){
            if(i == res) continue;// 附加条件中非g[i][i] == 1隐含条件
            if(knows(res, i) || !knows(i, res)) return -1;
        }
        return res;
    }
}
```
4. tic-tac-toe游戏
```java
// 矩阵，空间换O(1) move
class TicTacToe {
    int[][] rows;// 空间换O(1) move
    int[][] cols;
    int[][] diags;
    int n;
    /** Initialize your data structure here. */
    public TicTacToe(int n) {
        rows = new int[3][n]; // 只用row[1]row[2],对应2个player
        cols = new int[3][n];
        diags = new int[3][2];//只能是主对角线和辅对角线
        this.n = n;
    }
    
    /** Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins. */
    public int move(int row, int col, int player) {
        if(++rows[player][row] == n) return player;
        else if(++cols[player][col] == n) return player;
        else if(row == col && ++diags[player][0] == n) return player;
        else if(row + col == n-1 && ++diags[player][1] == n) return player;
        else return 0;
    }
}
// TODO: 类似题？N 皇后问题的某一种解法？
```
