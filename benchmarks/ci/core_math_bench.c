/* Speed and agreement of CORE-MATH's correctly rounded functions against the system libm, for the
 * functions miter calls: exp (gradients, predict_proba), log and log1p (init score, metrics, the
 * Counter CTR), sin and cos (Box-Muller weights of oblique splits).
 *
 * Every function runs REPS times over N inputs; the table gives the median, mean and minimum time
 * per call, and how many of the N results differ from CORE-MATH's (which is correctly rounded, so
 * a difference is the system's rounding). "sigmoid" is miter's two-branch logistic over raw scores,
 * the loop the binary gradients and predict_proba run. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

double cr_exp(double), cr_log(double), cr_log1p(double), cr_sin(double), cr_cos(double);

static double now(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}

#define N 1000000
static double x[N], a[N], b[N];
typedef double (*fn)(double);

static double sys_exp(double v) { return exp(v); }
static double sys_log(double v) { return log(v); }
static double sys_log1p(double v) { return log1p(v); }
static double sys_sin(double v) { return sin(v); }
static double sys_cos(double v) { return cos(v); }

static fn volatile g_exp;  /* the sigmoid loop calls exp through this, so both sides pay one call */
static double sigmoid(double z) {
    if (z >= 0.0) {
        const double ez = g_exp(-z);
        return 1.0 / (1.0 + ez);
    }
    const double ez = g_exp(z);
    return ez / (1.0 + ez);
}

static int cmp_double(const void* p, const void* q) {
    const double u = *(const double*)p, v = *(const double*)q;
    return (u > v) - (u < v);
}

typedef struct { double med, mean, min; } stats;

static stats time_fn(fn volatile f, double* out, int reps, double* t) {
    for (int r = 0; r < reps; r++) {
        const double t0 = now();
        for (int i = 0; i < N; i++) out[i] = f(x[i]);
        t[r] = (now() - t0) * 1e9 / N;
    }
    qsort(t, (size_t)reps, sizeof(double), cmp_double);
    stats s = {t[reps / 2], 0.0, t[0]};
    for (int r = 0; r < reps; r++) s.mean += t[r] / reps;
    return s;
}

static void fill(double lo, double hi) {
    uint64_t s = 88172645463325252ULL;
    for (int i = 0; i < N; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        x[i] = lo + (hi - lo) * ((double)(s >> 11) * 0x1.0p-53);
    }
}

static int reps;
static double* tbuf;

static void row(const char* name, const char* range, fn sys, fn cr) {
    /* Interleave: system, CORE-MATH, system, CORE-MATH, so a noisy neighbour hits both. */
    stats ss = {0}, sc = {0};
    for (int half = 0; half < 2; half++) {
        stats u = time_fn(sys, a, reps / 2, tbuf);
        stats v = time_fn(cr, b, reps / 2, tbuf);
        if (half == 0) { ss = u; sc = v; continue; }
        ss.med = (ss.med + u.med) / 2; ss.mean = (ss.mean + u.mean) / 2; ss.min = ss.min < u.min ? ss.min : u.min;
        sc.med = (sc.med + v.med) / 2; sc.mean = (sc.mean + v.mean) / 2; sc.min = sc.min < v.min ? sc.min : v.min;
    }
    long d = 0;
    for (int i = 0; i < N; i++) d += memcmp(&a[i], &b[i], sizeof(double)) != 0;
    printf("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2fx | %.3f%% |\n", name, range, ss.med, ss.mean,
           ss.min, sc.med, sc.mean, sc.min, sc.med / ss.med, 100.0 * (double)d / N);
}

static double sig_sys(double z) { g_exp = sys_exp; return sigmoid(z); }
static double sig_cr(double z) { g_exp = cr_exp; return sigmoid(z); }

int main(int argc, char** argv) {
    reps = argc > 1 ? atoi(argv[1]) : 30;
    if (reps < 2) reps = 2;
    tbuf = malloc(sizeof(double) * (size_t)reps);
    printf("| function | inputs | system median ns | mean | min | CORE-MATH median ns | mean | min | ratio (median) | results differ |\n");
    printf("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    fill(-30, 30);              row("exp", "[-30, 30]", sys_exp, cr_exp);
    fill(-8, 8);                row("sigmoid", "[-8, 8]", sig_sys, sig_cr);
    fill(1e-12, 1);             row("log", "(0, 1]", sys_log, cr_log);
    fill(1, 1e6);               row("log", "[1, 1e6]", sys_log, cr_log);
    fill(0, 1);                 row("log1p", "[0, 1]", sys_log1p, cr_log1p);
    fill(0, 1e6);               row("log1p", "[0, 1e6]", sys_log1p, cr_log1p);
    fill(0, 6.283185307179586); row("sin", "[0, 2pi]", sys_sin, cr_sin);
    fill(0, 6.283185307179586); row("cos", "[0, 2pi]", sys_cos, cr_cos);
    double chk = 0;
    for (int i = 0; i < N; i += 997) chk += a[i] + b[i];
    fprintf(stderr, "checksum %.17g\n", chk);
    free(tbuf);
    return 0;
}
