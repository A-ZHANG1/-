import java.util.*;

/**
 * Created by Wayne.A.Z on 2020-08-16.
 */
public class Weiruan {
    // zhousai 198 easy
    public int numWaterBottles(int numBottles, int numExchange) {
        int res = numBottles;
        int nEmpty = numBottles;
        while(nEmpty != 0 && nEmpty >=numExchange){
            int nCur = nEmpty / numExchange;
            res += nCur;
            nEmpty = nEmpty % numExchange + nCur;
        }
        return res;
    }

    public int[] countSubTrees(int n, int[][] edges, String labels) {
        // 用arraylist的数组做邻接链表
        List<Integer>[] points = new List[n];
        for(int i = 0;i < n;i ++) points[i] = new ArrayList<>();
        for (int[] point : edges) {
            points[point[0]].add(point[1]);
            points[point[1]].add(point[0]); // 无向图
        }

        //
        int[] lbs = new int[n];
        int idx = 0;
        for(char c: labels.toCharArray()){
            lbs[idx ++] = c - 'a';
        }

        res = new int[n];
        visited = new boolean[n];
        visited[0] = true;
        dfs(0, points, lbs);
        return res;
    }
    int[] res;
    boolean[] visited;
    public int[] dfs(int i, List<Integer>[] points,int[] lbs){
        int[] curLbs = new int[26];
        curLbs[lbs[i]] ++;
        for(int child: points[i]){
            // 对无向图，需要判断遍历过的跳过
            if(visited[child]) continue;
            visited[child]= true;
            int[] childLbs = dfs(child, points, lbs);
            for(int j = 0;j < 26;j ++){ // 当前节点在内的子树中每个孩子节点label的记录
                curLbs[j] += childLbs[j];
            }
        }
        res[i] = curLbs[lbs[i]];
        return curLbs;
    }

    public List<String> maxNumOfSubstrings(String s){
        List<String> res = new ArrayList<>();

        int N = s.length();
        char[] sArr = s.toCharArray();
        // 26个字母出现的第一、最后一个位置
        int[][] arr = new int[26][2]; // 或hashmap或类
        for (int i = 0; i < 26; i++) {
            arr[i][0] = Integer.MAX_VALUE; // 左
            arr[i][1] = Integer.MIN_VALUE; // 右
        }

        for (int i = 0; i < N; i++) {
            arr[sArr[i] - 'a'][0] = Math.min(arr[sArr[i] - 'a'][0], i);
            arr[sArr[i] - 'a'][1] = Math.max(arr[sArr[i] - 'a'][1], i);
        }

//        for (int i = 0; i < 26; i++) {
//            System.out.println(Arrays.toString(arr[i]));
//        }
        // 合并包含在首末位置间的字符的区间
        for (int i = 0; i < 26; i++) {
            if(arr[i][0] == Integer.MAX_VALUE) continue;
            for (int j = arr[i][0] + 1; j < arr[i][1]; j++) {
                int cur = sArr[j] - 'a';
                if((char)(i + 'a') != s.charAt(j) && (arr[cur][0] < arr[i][0] || arr[cur][1] > arr[i][1])) {
                    arr[i][0] = Math.min(arr[i][0], arr[cur][0]);
                    arr[i][1] = Math.max(arr[i][1], arr[cur][1]);
                    arr[cur] = arr[i];
                }
            }
        }
//        for (int i = 0; i < 26; i++) {
//            System.out.println(Arrays.toString(arr[i]));
//        }
        // 贪心，从短的区间开始选

        // 去重后排序
        List<List<Integer>> list = new ArrayList<>();
        Set<String> substrings = new HashSet<>();
        for (int i = 0; i < 26; i++) {
            if(arr[i][0] == Integer.MAX_VALUE) continue;
            String sub = s.substring(arr[i][0], arr[i][1] + 1);
            if(substrings.contains(sub)) continue;
            list.add(Arrays.asList(arr[i][0], arr[i][1])); //方便后续排序
            substrings.add(sub);
        }

        Collections.sort(list, ((o1, o2)->((o1.get(1) - o1.get(0)) - (o2.get(1) - o2.get(0)))));

        System.out.println(list);
        // 若子区间已被选择过，跳过
        List<List<Integer>> pick = new ArrayList<>();
        for(List<Integer> tmp : list){
            boolean flag = true;
            for(List<Integer> picked: pick){

                if(tmp.get(0) < picked.get(0) && tmp.get(1) > picked.get(1)){
                    flag = false;
                    break;
                }
            }
            if(flag){
                pick.add(tmp);
                res.add(s.substring(tmp.get(0), tmp.get(1) + 1));
            }
        }
    return res;

    }
    // 同 898 O(n log C),  C为数组最大范围
    public int closestToTarget(int[] arr, int target) {
        int ans = Integer.MAX_VALUE;
        Set<Integer> valid = new HashSet<>();
        for(int num: arr){
            Set<Integer> validPlusNum = new HashSet<>();
            validPlusNum.add(num);
            ans = Math.min(ans, Math.abs(num - target));
            for(int prev: valid){
                int cur = prev & num;
                validPlusNum.add(cur);
                ans = Math.min(ans, Math.abs(num - target));
            }
            valid = validPlusNum;
        }
        return ans;
    }




    public static void main(String[] args) {
        Weiruan s = new Weiruan();
//        System.out.println(s.numWaterBottles(5, 5));

//        int[][] edges = {{0,1}, {1,2}, {0,3}};
//        System.out.println(Arrays.toString(s.countSubTrees(4, edges,"bbbb")));
        System.out.println(s.maxNumOfSubstrings("adefaddaccc"));
        System.out.println(s.maxNumOfSubstrings("cabcccbaa"));
    }
}
