//import java.util.ArrayList;
//import java.util.List;
import java.util.*;
import java.io.*;
////程序员代码面试指南看书
//public class Main {
//
//    public List<Integer> diffWaysToCompute(String input) {
//        List<Integer> numList = new ArrayList<>();
//        List<Character> opList = new ArrayList<>();
//        char[] array = input.toCharArray();
//        int num = 0;
//        for (int i = 0; i < array.length; i++) {
//            if (isOperation(array[i])) {
//                numList.add(num);
//                num = 0;
//                opList.add(array[i]);
//                continue;
//            }
//            num = num * 10 + array[i] - '0';
//        }
//        numList.add(num);
//        int N = numList.size(); // 数字的个数
//
//        // 一个数字
//        ArrayList<Integer>[][] dp = (ArrayList<Integer>[][]) new ArrayList[N][N];
//        for (int i = 0; i < N; i++) {
//            ArrayList<Integer> result = new ArrayList<>();
//            result.add(numList.get(i));
//            dp[i][i] = result;
//        }
//        // 2 个数字到 N 个数字
//        for (int n = 2; n <= N; n++) {
//            // 开始下标
//            for (int i = 0; i < N; i++) {
//                // 结束下标
//                int j = i + n - 1;
//                if (j >= N) {
//                    break;
//                }
//                ArrayList<Integer> result = new ArrayList<>();
//                // 分成 i ~ s 和 s+1 ~ j 两部分
//                for (int s = i; s < j; s++) {
//                    ArrayList<Integer> result1 = dp[i][s];
//                    ArrayList<Integer> result2 = dp[s + 1][j];
//                    for (int x = 0; x < result1.size(); x++) {
//                        for (int y = 0; y < result2.size(); y++) {
//                            // 第 s 个数字下标对应是第 s 个运算符
//                            char op = opList.get(s);
//                            result.add(caculate(result1.get(x), op, result2.get(y)));
//                        }
//                    }
//                }
//                dp[i][j] = result;
//
//            }
//        }
//        return dp[0][N-1];
//    }
//
//    private int caculate(int num1, char c, int num2) {
//        switch (c) {
//            case '+':
//                return num1 + num2;
//            case '-':
//                return num1 - num2;
//            case '*':
//                return num1 * num2;
//        }
//        return -1;
//    }
//
//    private boolean isOperation(char c) {
//        return c == '+' || c == '-' || c == '*';
//    }
//
//    public static void main(String[] args) {
//        //perm test
////        int arr[] ={1,2,3,0,5,4};
////        Perm perm = new Perm();
////        perm.perm(arr,0,6);
////        System.out.println(perm.max);
//
//        //josephKill
//
//
//    }
//}
public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(new BufferedInputStream(System.in));
        PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));
        Main M = new Main();
        int N = sc.nextInt();
        int m = sc.nextInt();
        int c = sc.nextInt();
        int[][] nums = new int[N  + m ][c];
        for(int i = 0;i < N;i ++){
            int num_i = sc.nextInt();
//            int[] color = new int[c];
            for(int j = 0;j < num_i;j ++){
                int cur = sc.nextInt();
                nums[i][c - 1] = 1;
            }
//            nums[i] = color;
        }
        for(int i = N;i < N + m;i ++){
            nums[i] = nums[i - N];
        }
//        System.out.println(Arrays.toString(nums));
        int res = 0;
        for(int i = 0;i < N;i ++){
//            int check = nums[i];
            for(int j = 1;j <  m;j ++){
//                System.out.printf("i: %d,", i);
//                System.out.printf("j: %d,",j);
//                System.out.printf("numsij: %d\n",nums[i + j]);
//                check &= nums[i + j];
//                System.out.println(check);
                for (int col = 0; col < c; col++) {
                    if(nums[i][col] == 1 && nums[i + j][col] == 1) {
                        res += 1;
                        continue;
                    }
                }

            }
        }

        System.out.println(res);
    }
}
