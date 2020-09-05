/**
 * Created by Wayne.A.Z on 2020-08-15.
 */



import java.util.*;
import java.io.*;

// 美团点评2020校招系统开发方向笔试题
public class Meituan {
//    public static void main(String args[]){
//        Scanner scanner = new Scanner(new BufferedInputStream(System.in));
//        PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));
//
//        // 获取输入值
////        int n = scanner.nextInt();
//        String a = scanner.nextLine();
//        String b = scanner.nextLine();
//
//        a = a.substring(1, a.length() - 1);
//        b = b.substring(1, b.length() - 1);
//
//        int intA = Integer.parseInt(a);
//        int intB = Integer.parseInt(b);
//        int sum = intA + intB;
//
//        // 输出
//        out.println(sum);
//
//        //刷新缓冲区
//        out.flush();
//    }


    public int numPalindrome(String s){
        if(s == null || s.length() == 0) return 0;
        if(s.length() == 1) return 1;

        int N = s.length();
        int ans = N;
        int[][] dp = new int[N][N];
        for (int i = N - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for(int j = i + 1;j < N;j ++){
                if(s.charAt(i) == s.charAt(j) && (i == j - 1 || dp[i + 1][j - 1] == 1)){
                    dp[i][j] = 1;
                    ans ++;
                }
            }
        }
        return ans;
    }
    //TODO:  1000 花花  区间dp模板
    public int mergeStones(int[] stones, int K){
        int N = stones.length;

        if((N - 1) % (K - 1) != 0) return -1;

        // 求前缀
        int[] prefixSum = new int[N + 1];
        for (int i = 1; i <= N; i++) {
            prefixSum[i] = prefixSum[i - 1] + stones[i - 1];
        }

        // 初始化
        int[][][] dp = new int[N][N][K + 1]; // start, end, 堆。表示将 [i, j] 区间的石头缩小成 k 堆的最小体力花费
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                Arrays.fill(dp[i][j], Integer.MAX_VALUE);
            }
            dp[i][i][1] = 0;
        }

        // 区间dp模板
        for(int len = 2;len <= N; len++){ // sub problem length
            for(int i = 0;i <= N-len;i ++){
                int j = i + len - 1;
                for(int k = 2;k <= K;k ++){
                    for(int m = i;m < j;m += K-1){ // m 跳步应该是K-1,不应该用k-1;
                        dp[i][j][k] = Math.min(dp[i][j][k],dp[i][m][1]+dp[m+1][j][k-1]);
                    }
                    dp[i][j][1] = dp[i][j][k] + prefixSum[j + 1] - prefixSum[i];
                }
            }
        }
        return dp[0][N - 1][1];
    }

//    public void split(String[] strs){
//        int N = strs.length;
//        int[] match_count = new int[N];
//        for(int i = 0;i < N;i ++){
//            for(int j = i + 1;j < N;j ++){
//                int cur_match_count = 0;
//                while(strs[i].charAt(cur_match_count) < strs[j].charAt(cur_match_count)) cur_match_count ++;
//                if(match_count[i] < cur_match_count) match_count[i] = cur_match_count;
//                if(match_count[j] < cur_match_count) match_count[j] = cur_match_count;
//            }
//        }
//        String[] res = new String[N];
//        for(int i = 0;i < N;i ++){
//            res[i] =
//        }
//    }


    public static void main(String args[]) {
        Scanner scanner = new Scanner(new BufferedInputStream(System.in));
        PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));

        // 获取输入值
//        String input = scanner.nextLine();

        Meituan s = new Meituan();

//        int res = s.numPalindrome(input);

        // 输出
//        out.println(res);
        int[] stones = {3, 2, 4, 1};
        out.println(s.mergeStones(stones, 2));

        String[] strs = {"meituanapp","meituanwaimai","dianpingren","dianpingjieguo"};
//        s.split(strs);

        //刷新缓冲区
        out.flush();
    }
}

