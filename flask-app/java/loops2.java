//print a table of n numbers input by a user

import java.util.*;

public class loops2 {
    public static void main(String args[]) {
        try (Scanner sc = new Scanner(System.in)) {
            int n = sc.nextInt();

            for (int i = 1; i <= 10; i++) {
                System.out.println((n * i));
            }
        }
    }
}
