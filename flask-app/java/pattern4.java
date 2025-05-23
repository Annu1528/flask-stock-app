//inverted half pyramid
public class pattern4 {
    public static void main(String[] args) {
        int n = 5;

        for (int i = n; i >= 1; i--) {        // Start from n down to 1
            for (int j = 1; j <= i; j++) {    // Print j stars
                System.out.print("*");
            }
            System.out.println();             // New line after each row
        }
    }
}


