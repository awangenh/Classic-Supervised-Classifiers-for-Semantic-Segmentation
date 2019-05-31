#include <stdio.h>

#ifdef DEBUG
#define PRINT_DEBUG(format, args...) \
    do { \
    printf("%s:%d: " format "\n", __FUNCTION__, __LINE__, ## args); fflush(stdout); \
    } while(0);
#else
#define PRINT_DEBUG(format, args...)
#endif

