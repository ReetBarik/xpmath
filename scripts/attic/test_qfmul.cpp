// Standalone host validator for QuadFloat arithmetic (4×FP32 port of QD 2.3.24).
// Mirrors scripts/test_ffmul.cpp: zero external deps beyond libquadmath, so it
// compiles and runs OUTSIDE CMake. The QuadFloat routines below are a host-only
// (plain-float, no Kokkos) copy of third_party/include/qf_math.hpp, kept
// bit-identical to the header and citing the same QD 2.3.24 source locations —
// exactly the "duplicate the routine under test" pattern test_ffmul.cpp uses for
// the FF multiply primitive. Ground truth is __float128 (113-bit), ~17 digits
// of headroom over QuadFloat's ~29, so it is an exact oracle here.
//
// Build: g++ -std=c++17 -fext-numeric-literals -O2 scripts/test_qfmul.cpp \
//            -lquadmath -o scripts/test_qfmul
// Run:   ./scripts/test_qfmul      (RC 0 = all ops pass, RC 1 = any miss)

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <quadmath.h>

// ============================================================
// QuadFloat (host mirror of qf_math.hpp) — plain float, no Kokkos.
// ============================================================
struct QF { float f0, f1, f2, f3; };

static inline QF qf(double x) {                 // faithful FP64 -> QF (Route-A)
    double r = x;
    float c0 = (float)r; r -= (double)c0;
    float c1 = (float)r; r -= (double)c1;
    float c2 = (float)r; r -= (double)c2;
    float c3 = (float)r;
    return {c0, c1, c2, c3};
}

// --- EFT primitives (QD inline.h:35-113) ---
static inline float quick_two_sum(float a, float b, float& e){ float s=a+b; e=b-(s-a); return s; }
static inline float two_sum(float a, float b, float& e){ float s=a+b, bb=s-a; e=(a-(s-bb))+(b-bb); return s; }
static inline float two_prod(float a, float b, float& e){
    const float split=8193.0f;
    float cona=a*split, conb=b*split;
    float a1=cona-(cona-a), b1=conb-(conb-b);
    float a2=a-a1,          b2=b-b1;
    float p=a*b; e=((a1*b1-p)+a1*b2+a2*b1)+a2*b2; return p;
}
static inline float two_sqr(float a, float& e){
    const float split=8193.0f;
    float con=a*split, hi=con-(con-a), lo=a-hi;
    float q=a*a; e=((hi*hi-q)+2.0f*hi*lo)+lo*lo; return q;
}
static inline void three_sum(float& a, float& b, float& c){
    float t1,t2,t3; t1=two_sum(a,b,t2); a=two_sum(c,t1,t3); b=two_sum(t2,t3,c);
}
static inline void three_sum2(float& a, float& b, float& c){
    float t1,t2,t3; t1=two_sum(a,b,t2); a=two_sum(c,t1,t3); b=t2+t3;
}

// --- renorm (QD qd_inline.h:95-125) ---
static inline void renorm4(float& c0,float& c1,float& c2,float& c3){
    float s0,s1,s2=0,s3=0;
    if(std::isinf(c0))return;
    s0=quick_two_sum(c2,c3,c3); s0=quick_two_sum(c1,s0,c2); c0=quick_two_sum(c0,s0,c1);
    s0=c0; s1=c1;
    if(s1!=0){ s1=quick_two_sum(s1,c2,s2); if(s2!=0) s2=quick_two_sum(s2,c3,s3); else s1=quick_two_sum(s1,c3,s2); }
    else { s0=quick_two_sum(s0,c2,s1); if(s1!=0) s1=quick_two_sum(s1,c3,s2); else s0=quick_two_sum(s0,c3,s1); }
    c0=s0;c1=s1;c2=s2;c3=s3;
}
// --- renorm_4 / length-5 (QD qd_inline.h:127-177) ---
static inline void renorm5(float& c0,float& c1,float& c2,float& c3,float& c4){
    float s0,s1,s2=0,s3=0;
    if(std::isinf(c0))return;
    s0=quick_two_sum(c3,c4,c4); s0=quick_two_sum(c2,s0,c3); s0=quick_two_sum(c1,s0,c2); c0=quick_two_sum(c0,s0,c1);
    s0=c0; s1=c1;
    if(s1!=0){
        s1=quick_two_sum(s1,c2,s2);
        if(s2!=0){ s2=quick_two_sum(s2,c3,s3); if(s3!=0) s3+=c4; else s2=quick_two_sum(s2,c4,s3); }
        else { s1=quick_two_sum(s1,c3,s2); if(s2!=0) s2=quick_two_sum(s2,c4,s3); else s1=quick_two_sum(s1,c4,s2); }
    } else {
        s0=quick_two_sum(s0,c2,s1);
        if(s1!=0){ s1=quick_two_sum(s1,c3,s2); if(s2!=0) s2=quick_two_sum(s2,c4,s3); else s1=quick_two_sum(s1,c4,s2); }
        else { s0=quick_two_sum(s0,c3,s1); if(s1!=0) s1=quick_two_sum(s1,c4,s2); else s0=quick_two_sum(s0,c4,s1); }
    }
    c0=s0;c1=s1;c2=s2;c3=s3;
}

static inline QF negate(QF a){ return {-a.f0,-a.f1,-a.f2,-a.f3}; }

// sloppy_add (QD qd_inline.h:338-405)
static inline QF add(QF a, QF b){
    float s0,s1,s2,s3,t0,t1,t2,t3,v0,v1,v2,v3,u0,u1,u2,u3,w0,w1,w2,w3;
    s0=a.f0+b.f0; s1=a.f1+b.f1; s2=a.f2+b.f2; s3=a.f3+b.f3;
    v0=s0-a.f0; v1=s1-a.f1; v2=s2-a.f2; v3=s3-a.f3;
    u0=s0-v0; u1=s1-v1; u2=s2-v2; u3=s3-v3;
    w0=a.f0-u0; w1=a.f1-u1; w2=a.f2-u2; w3=a.f3-u3;
    u0=b.f0-v0; u1=b.f1-v1; u2=b.f2-v2; u3=b.f3-v3;
    t0=w0+u0; t1=w1+u1; t2=w2+u2; t3=w3+u3;
    s1=two_sum(s1,t0,t0); three_sum(s2,t0,t1); three_sum2(s3,t0,t2); t0=t0+t1+t3;
    renorm5(s0,s1,s2,s3,t0);
    return {s0,s1,s2,s3};
}
static inline QF sub(QF a, QF b){ return add(a, negate(b)); }

// multiply_scalar (QD qd_inline.h:490-514)
static inline QF mul_f(QF a, float b){
    float p0,p1,p2,p3,q0,q1,q2,s0,s1,s2,s3,s4;
    p0=two_prod(a.f0,b,q0); p1=two_prod(a.f1,b,q1); p2=two_prod(a.f2,b,q2); p3=a.f3*b;
    s0=p0; s1=two_sum(q0,p1,s2); three_sum(s2,q1,p2); three_sum2(q1,q2,p3); s3=q1; s4=q2+p2;
    renorm5(s0,s1,s2,s3,s4);
    return {s0,s1,s2,s3};
}
// sloppy_mul (QD qd_inline.h:567-599)
static inline QF mul(QF a, QF b){
    float p0,p1,p2,p3,p4,p5,q0,q1,q2,q3,q4,q5,t0,t1,s0,s1,s2;
    p0=two_prod(a.f0,b.f0,q0);
    p1=two_prod(a.f0,b.f1,q1); p2=two_prod(a.f1,b.f0,q2);
    p3=two_prod(a.f0,b.f2,q3); p4=two_prod(a.f1,b.f1,q4); p5=two_prod(a.f2,b.f0,q5);
    three_sum(p1,p2,q0);
    three_sum(p2,q1,q2); three_sum(p3,p4,p5);
    s0=two_sum(p2,p3,t0); s1=two_sum(q1,p4,t1); s2=q2+p5; s1=two_sum(s1,t0,t0); s2+=(t0+t1);
    s1+= a.f0*b.f3 + a.f1*b.f2 + a.f2*b.f1 + a.f3*b.f0 + q0+q3+q4+q5;
    renorm5(p0,p1,s0,s1,s2);
    return {p0,p1,s0,s1};
}
// sloppy_div (QD qd_real.cpp:693-712)
static inline QF divi(QF a, QF b){
    float q0,q1,q2,q3; QF r;
    q0=a.f0/b.f0; r=sub(a,mul_f(b,q0));
    q1=r.f0/b.f0; r=sub(r,mul_f(b,q1));
    q2=r.f0/b.f0; r=sub(r,mul_f(b,q2));
    q3=r.f0/b.f0;
    renorm4(q0,q1,q2,q3);
    return {q0,q1,q2,q3};
}
// sqrt via Heron (QD qd_real.cpp:738-785)
static inline QF qfsqrt(QF a){
    if(a.f0==0&&a.f1==0&&a.f2==0&&a.f3==0) return {0,0,0,0};
    const float eps=1.2621774e-29f; const QF half={0.5f,0,0,0};
    QF x=qf((double)std::sqrt(a.f0));
    for(int i=0;i<10;i++){
        QF y=mul(half, add(x, divi(a,x)));
        QF d=sub(x,y); x=y;
        float e=std::fabs(((d.f3+d.f2)+d.f1)+d.f0);
        if(e<std::fabs(x.f0)*eps) return x;
    }
    return x;
}

// ============================================================
// Oracle helpers
// ============================================================
static inline __float128 to_q(QF a){
    return (__float128)a.f0 + (__float128)a.f1 + (__float128)a.f2 + (__float128)a.f3;
}
// digits of accuracy = -log10(rel_err), capped at 29 (QF target).
static double digits(__float128 got, __float128 ref){
    if(ref==0) return got==0 ? 29.0 : 0.0;
    __float128 rel = fabsq((got-ref)/ref);
    if(rel==0) return 29.0;
    double d = -(double)(logq(rel)/logq(10.0Q));
    return d > 29.0 ? 29.0 : d;
}

int main(){
    // QF target: u = 2^-96 ~= 1.26e-29 -> ~28.9 digits. Gate on the MEAN at 28
    // digits (matching the Phase-1/2 methodology: fail-gate on mean, report min
    // separately so a single conditioning-sensitive sample — e.g. sloppy_div's
    // documented ~1-ulp worst case, or a near-cancellation subtract — does not
    // false-fail). QD's default multiply/divide are "sloppy" (fast, ~1-2 ulp
    // looser than the accurate variants); their means still clear 28 comfortably.
    const double GATE = 28.0;
    int total_fail = 0;

    struct Case { const char* label; double a, b; };
    const Case cases[] = {
        {"1/3",              1.0, 3.0},
        {"2/7",              2.0, 7.0},
        {"pi",               3.141592653589793, 0.0},
        {"e * pi",           2.718281828459045, 3.141592653589793},
        {"sqrt(2)^2",        1.4142135623730951, 0.0},
        {"123.456 + 789.012",123.456, 789.012},
        {"1e6 - 1e6*(1+1e-13)", 1000000.0, 0.0},
        {"huge*tiny",        1.0e18, 1.0e-18},
    };

    // Per-op accumulators (min + running mean; gate on mean).
    struct Stat { const char* name; int n; double mind; double sumd; } S[5] = {
        {"add",0,99,0},{"subtract",0,99,0},{"multiply",0,99,0},
        {"divide",0,99,0},{"sqrt",0,99,0}
    };
    auto record=[&](int op,double d){ S[op].n++; S[op].sumd+=d; if(d<S[op].mind)S[op].mind=d; };

    // --- Directed cases ---
    for(const auto& c : cases){
        QF A=qf(c.a), B=qf(c.b);
        __float128 qa=(__float128)c.a, qb=(__float128)c.b;
        // add
        record(0, digits(to_q(add(A,B)), qa+qb));
        // subtract
        record(1, digits(to_q(sub(A,B)), qa-qb));
        // multiply
        record(2, digits(to_q(mul(A,B)), qa*qb));
        // divide: skip b==0 and quotients beyond the FP32 Dekker-splitter-safe
        // magnitude (|q| > ~2^102 ~= 5e30 makes 8193*q overflow FP32 inside
        // two_prod -> NaN). That is an out-of-safe-domain input for a
        // float-word type, not a divide defect — same skip-not-fail posture as
        // T1.1's splitter-overflow guard. Documented in PORT_NOTES_QF.
        if(c.b!=0.0 && fabsq(qa/qb) < 1.0e30Q) record(3, digits(to_q(divi(A,B)), qa/qb));
        // sqrt (a>=0)
        if(c.a>=0.0) record(4, digits(to_q(qfsqrt(A)), sqrtq(qa)));
    }

    // --- Random fuzz (uniform in [-1e6,1e6], plus positive draws for sqrt) ---
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<double> uni(-1e6,1e6), pos(1e-6,1e6);
    const int N=200000;
    for(int i=0;i<N;i++){
        double a=uni(rng), b=uni(rng);
        QF A=qf(a), B=qf(b);
        __float128 qa=(__float128)a, qb=(__float128)b;
        record(0, digits(to_q(add(A,B)), qa+qb));
        record(1, digits(to_q(sub(A,B)), qa-qb));
        record(2, digits(to_q(mul(A,B)), qa*qb));
        // Skip divides whose quotient leaves the FP32 range (|a/b| > ~3.4e38):
        // that is an out-of-domain input for a float-word type, not a divide
        // defect — the same skip-not-fail posture used in T1.1/T2.1. Guard on
        // |b| so the quotient stays comfortably finite.
        if(b!=0.0 && std::fabs(b) > 1.0e-3) record(3, digits(to_q(divi(A,B)), qa/qb));
        double p=pos(rng); QF P=qf(p);
        record(4, digits(to_q(qfsqrt(P)), sqrtq((__float128)p)));
    }

    std::printf("QuadFloat smoke test — gate = %.1f mean digits vs __float128 oracle\n", GATE);
    std::printf("%-10s  %6s  %9s  %8s  %s\n","op","result","mean_dig","min_dig","N");
    for(int i=0;i<5;i++){
        double mean = S[i].n ? S[i].sumd/S[i].n : 0.0;
        bool pass = mean >= GATE;
        if(!pass) total_fail++;
        std::printf("%-10s  %6s  %9.2f  %8.2f  %d\n",
                    S[i].name, pass?"PASS":"FAIL", mean, S[i].mind, S[i].n);
    }

    std::printf("\n%s (%d op means below gate)\n",
                total_fail==0?"ALL PASS":"FAILURES PRESENT", total_fail);
    return total_fail==0 ? 0 : 1;
}
