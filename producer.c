#include <stdio.h>


int main() {
    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < 10; ++j) {
            printf("%d ", j);
        }
        printf("\n");
    }

    return 0;
}