/* roundeven for libms without it (macOS). On the x86-64 baseline (no SSE4.1) compilers turn
 * __builtin_roundeven into a call; rint rounds to nearest even in the default rounding mode. */
#include <math.h>
double roundeven(double x) { return rint(x); }
