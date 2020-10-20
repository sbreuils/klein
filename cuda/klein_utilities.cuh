#ifndef __CUDA_UTILITIES_H
#define __CUDA_UTILITIES_H

#include <stdio.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
//#include "device_functions.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <math.h>
#include <algorithm>
#include <time.h>
#include <limits.h>
#include <vector>



// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check
//#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include "cutils_math.h"
#include "cutils_matrix.h"


// Preprocessor definitions for width and height of color and depth streams
#define THREAD_SIZE_X 32
#define THREAD_SIZE_Y 32
#define THREAD_SIZE_L_X 8
#define THREAD_SIZE_L_Y 8
#define THREAD_SIZE THREAD_SIZE_L_X*THREAD_SIZE_L_Y
#define STRIDE 512
#define MAXTOLERANCE 0.2
// #define EPSILON 0.1
#define ALPHA 0.8 

//#define PI 3.1415926535897932384626433832795

#define MAX_DEPTH 100.0

#define divUp(x,y) (x%y) ? ((x+y-1)/y) : (x/y)

using namespace std;



// p0 -> (e0, e1, e2, e3)
// p1 -> (1, e23, e31, e12)
// p2 -> (e0123, e01, e02, e03)
// p3 -> (e123, e032, e013, e021)

struct kln_plane
{
    float4 p0;
};

struct kln_line
{
    float4 p1;
    float4 p2;

    __device__ kln_line(float a, float b, float c, float d, float e, float f) {
        p1.x = 0.0f;
        p1.y = d;
        p1.z = e;
        p1.w = f;

        p2.x = 0.0f;
        p2.y = a;
        p2.z = b;
        p2.w = c;

    }
    __device__ kln_line(float4 _P1, float4 _P2) {
        p1 = _P1;
        p2 = _P2;
    }

};


// If integrating this library with other code, remember that the point layout
// here has the homogeneous component in p3[0] and not p3[3]. The swizzle to
// get the vec3 Cartesian representation is p3.yzw
struct kln_point
{
    float4 p3;
};

struct kln_rotor
{
    float4 p1;
};

struct kln_translator
{
    float4 p2;
};

struct kln_motor
{
    float4 p1;
    float4 p2;

    __device__ kln_motor(float a, float b, float c, float d, float e, float f, float g, float h) {
        p1.x = a;
        p1.y = b;
        p1.z = c;
        p1.w = d;

        // p2.x = e;
        // p2.y = f;
        // p2.z = g;
        // p2.w = h;
        p2.x = h;
        p2.y = e;
        p2.z = f;
        p2.w = g;

    }
    __device__ kln_motor(float4 _P1, float4 _P2) {
        p1 = _P1;
        p2 = _P2;
    }

    // add two motors, use it for blending operations
    __device__ kln_motor operator+ (kln_motor m2) {
        return kln_motor(p1 + m2.p1, p2 + m2.p2);
    }

    // multiply motors by weights, use it for blending operations
    __device__ kln_motor operator* (float s) {
        return kln_motor(s * p1, s * p2);
    }

    // compose two motors, again we can use this for blending operations
    __device__ kln_motor operator* (kln_motor b) {
        float4 cp1;
        float4 cp2;

        float4 a_zyzw = make_float4(p1.z,p1.y,p1.z,p1.w);
        float4 a_ywyz = make_float4(p1.y,p1.w,p1.y,p1.z);
        float4 a_wzwy = make_float4(p1.w,p1.z,p1.w,p1.y);


        float4 c_wwyz = make_float4(b.p1.w,b.p1.w,b.p1.y,b.p1.z);
        float4 c_yzwy = make_float4(b.p1.y,b.p1.z,b.p1.w,b.p1.y);//yzwy;
    
        cp1 = p1.x * b.p1;
        cp1 = p1.x * b.p1;
        float4 t = a_ywyz * c_yzwy;
        t += a_zyzw * make_float4(b.p1.z,b.p1.x,b.p1.x,b.p1.x);// zxxx;
        t.x = -t.x;
        cp1 += t;
        cp1 -= a_wzwy * c_wwyz;
    
        cp2 = p1.x * b.p2;
        cp2 = p1.x * b.p2;
        cp2 += p2 * b.p1.x;
        cp2 += a_ywyz * make_float4(b.p2.y,b.p2.z,b.p2.w,b.p2.y); // do smthing
        cp2 += make_float4(p2.y,p2.w,p2.y,p2.z) * c_yzwy; // do smthing
        t = a_zyzw * make_float4(b.p2.z,b.p2.x,b.p2.x,b.p2.x); // do smthing
        t += a_wzwy * make_float4(b.p2.w,b.p2.w,b.p2.y,b.p2.z); // do smthing
        t += make_float4(p2.z,p2.x,p2.x,p2.x) * make_float4(b.p1.z,b.p1.y,b.p1.z,b.p1.w); // do smthing
        t += make_float4(p2.w,p2.z,p2.w,p2.y) * c_wwyz; // do smthing
        t.x = -t.x;
        cp2 -= t;
        return kln_motor(cp1,cp2);
    }

};

__device__ kln_motor kln_normalize(kln_motor m){
    float4 mout1 = m.p1;
    float4 mout2 = m.p2;

    float4 b2 = kln_dp_bc(m.p1,m.p1); // dot product as float4
    float4 s  = kln_rsqrt_nr1(b2);
    
    float4 bc = kln_dp_bc(m.p1 * make_float4(-1.0f,1.0f,1.0f,1.0f),m.p2);
    float4 t = (bc*kln_rcp_nr1(b2))*s;


    float4 tmp = m.p2*s;
    mout2 = tmp - ((m.p1*t)*make_float4(-1.0f,1.0f,1.0f,1.0f));
    mout1 = m.p1*s;

    return kln_motor(mout1,mout2);
}


__device__ kln_motor kln_exp(kln_line l){

    float4 a = l.p1;
    float4 b = l.p2;



    // float4 b2 = kln_dp_bc(m.p1,m.p1); // dot product as float4
    // float4 s  = kln_rsqrt_nr1(b2);
    
    // float4 bc = kln_dp_bc(m.p1 * make_float4(-1.0f,1.0f,1.0f,1.0f),m.p2);
    // float4 t = (bc*kln_rcp_nr1(b2))*s;


    // float4 tmp = m.p2*s;
    // mout2 = tmp - ((m.p1*t)*make_float4(-1.0f,1.0f,1.0f,1.0f));
    // mout1 = m.p1*s;

    float4 a2 = kln_hi_dp_bc(a,a);
    float4 ab = kln_hi_dp_bc(a,b);

    float4 a2_sqrt_rcp = kln_rsqrt_nr1(a2);
    float4 u = a2*a2_sqrt_rcp;
    float4 minus_v = ab*a2_sqrt_rcp;

    float4 norm_real = a*a2_sqrt_rcp;
    float4 norm_ideal = b*a2_sqrt_rcp;

    norm_ideal -= a*(ab*(a2_sqrt_rcp*kln_rcp_nr1(a2))); 

    // store the lower part of u into
    float2 uv;
    uv.x = u.w; 
    uv.y = minus_v.w;
    
    float2 sincosu;
    sincosu.x = std::sin(uv.x);
    sincosu.y = std::cos(uv.x);

    float4 sinu = make_float4(sincosu.x);

    float4 p1_out = make_float4(sincosu.y,0.0f,0.0f,0.0f)+(sinu*norm_real);

    // The second partition has contributions from both the real and ideal
    // parts.
    float4 cosu = make_float4(0.0f, sincosu.y, sincosu.y, sincosu.y);


    float4 minus_vcosu = minus_v*cosu;

    float4 p2_out = sinu*norm_ideal;
    p2_out += (minus_vcosu*norm_real);

    p2_out += make_float4(uv.y * sincosu.x,0.0f,0.0f,0.0f);

    return kln_motor(p1_out,p2_out);
}


__device__ kln_line kln_log(kln_motor m){

    float4 p1 = m.p1;
    float4 p2 = m.p2;




    float4 bv_mask = make_float4(0.0f, 1.0f, 1.0f, 1.0f);
    float4 a = bv_mask*p1;
    float4 b = bv_mask*p2;

    // Next, we need to compute the norm as in the exponential.
    float4 a2 = kln_hi_dp_bc(a,a);

    float4 ab = kln_hi_dp_bc(a,b);
    float4 a2_sqrt_rcp = kln_rsqrt_nr1(a2);
    float4 s = a2*a2_sqrt_rcp;
    float4 minus_t = ab*a2_sqrt_rcp;
    // s + t e0123 is the norm of our bivector.

    // Store the scalar component
    float p = p1.x;
    // Store the pseudoscalar component
    float q = p2.x;
    float s_scalar = s.w;
    float t_scalar = minus_t.w;
    t_scalar *= -1.f;

    // p = cosu
    // q = -v sinu
    // s_scalar = sinu
    // t_scalar = v cosu
    
    bool p_zero = std::abs(p) < 1e-6;
    float u = p_zero ? std::atan2(-q, t_scalar) : std::atan2(s_scalar, p);
    float v = p_zero ? -q / s_scalar : t_scalar / p;

    // Now, (u + v e0123) * n when exponentiated will give us the motor, so
    // (u + v e0123) * n is the logarithm. To proceed, we need to compute
    // the normalized bivector.
    float4 norm_real  = (a*a2_sqrt_rcp);
    float4 norm_ideal = (b*a2_sqrt_rcp);

    norm_ideal -= (a*(ab*(a2_sqrt_rcp*kln_rcp_nr1(a2))));



    float4 uvec = make_float4(u);
    
    float4 p1_out = uvec*norm_real;
    float4 p2_out = (uvec*norm_ideal)-(make_float4(v)*norm_real);



    return kln_line(p1_out,p2_out);
}



// apply a motor to a point
// apply a motor to a plane
// apply a rotor to a point/vector
__device__ kln_point kln_apply(kln_motor m, kln_point p)
{
    float4 scale = make_float4(0.0f, 2.0f, 2.0f, 2.0f);
    float4 mp1_xwyz = make_float4(m.p1.x,m.p1.w,m.p1.y,m.p1.z);
    float4 mp1_xzwy = make_float4(m.p1.x,m.p1.z,m.p1.w,m.p1.y);


    float4 t1 = m.p1 * mp1_xwyz;
    t1 -= m.p1.x * mp1_xzwy;
    t1 *= scale; 

    float4 t2 = m.p1.x * mp1_xwyz;
    t2 += mp1_xzwy * m.p1; 
    t2 *= scale; 

    float4 t3 = m.p1 * m.p1; 
    t3 += make_float4(m.p1.y,m.p1.x,m.p1.x,m.p1.x) * make_float4(m.p1.y,m.p1.x,m.p1.x,m.p1.x); 
    float4 t4 = make_float4(m.p1.z,m.p1.w,m.p1.y,m.p1.z) * make_float4(m.p1.z,m.p1.w,m.p1.y,m.p1.z); 
    t4 += make_float4(m.p1.w,m.p1.z,m.p1.w,m.p1.y) * make_float4(m.p1.w,m.p1.z,m.p1.w,m.p1.y);
    t3 -= t4 * make_float4(-1.0f,1.0f,1.0f,1.0f);

    // up to above it is OKAY

    t4 = mp1_xzwy * make_float4(m.p2.x,m.p2.w,m.p2.y,m.p2.z); 
    t4 -= m.p1.x * m.p2; 
    t4 -= mp1_xwyz * make_float4(m.p2.x,m.p2.z,m.p2.w,m.p2.y);
    t4 -= m.p1 * m.p2.x; 
    t4 *= scale;

    // TODO: provide variadic motor-point application
    kln_point q;
    q.p3 = t1 * make_float4(p.p3.x,p.p3.w,p.p3.y,p.p3.z);
    q.p3 += t2 * make_float4(p.p3.x,p.p3.z,p.p3.w,p.p3.y);
    q.p3 += t3 * p.p3;
    q.p3 += t4 * p.p3.x;
    return  q;
}

// apply a motor to a plane
__device__ kln_plane kln_apply(kln_motor m, kln_plane p)
{
    float4 dc_scale = make_float4(1.0f, 2.0f, 2.0f, 2.0f);
    float4 neg_low  = make_float4(-1.0f, 1.0f, 1.0f, 1.0f);

    float4 mp1_xwyz = make_float4(m.p1.x,m.p1.w,m.p1.y,m.p1.z);
    float4 mp1_xzwy = make_float4(m.p1.x,m.p1.z,m.p1.w,m.p1.y);


    float4 t1 = make_float4(m.p1.z,m.p1.x,m.p1.x,m.p1.x) * make_float4(m.p1.z,m.p1.w,m.p1.y,m.p1.z) ;
    t1 += make_float4(m.p1.y,m.p1.z,m.p1.w,m.p1.y) * make_float4(m.p1.y,m.p1.y,m.p1.z,m.p1.w) ;
    t1 *= dc_scale;

    float4 t2 = m.p1 * mp1_xwyz;
    t2 -= (make_float4(m.p1.w,m.p1.x,m.p1.x,m.p1.x) * make_float4(m.p1.w,m.p1.z,m.p1.w,m.p1.y)) * neg_low;
    t2 *= dc_scale;

    float4 t3 = m.p1 * m.p1;
    t3 -= mp1_xwyz * mp1_xwyz;
    t3 += make_float4(m.p1.x*m.p1.x);
    t3 -= mp1_xzwy * mp1_xzwy;

    float4 t4 = m.p1.x * m.p2;
    t4 += mp1_xzwy * make_float4(m.p2.x,m.p2.w,m.p2.y,m.p2.z);
    t4 += m.p1 * m.p2.x;
    t4 -= mp1_xzwy * make_float4(m.p2.x,m.p2.z,m.p2.w,m.p2.y);
    t4 *= make_float4(0.0f, 2.0f, 2.0f, 2.0f);

    // TODO: provide variadic motor-plane application
    kln_plane q;
    q.p0 = t1 * make_float4(p.p0.x,p.p0.z,p.p0.w,p.p0.y); 
    q.p0 += t2 * make_float4(p.p0.x,p.p0.w,p.p0.y,p.p0.z);
    q.p0 += t3 * p.p0;
    q.p0 += make_float4(dot(t4, p.p0), 0.0, 0.0, 0.0);
    return q;
}

__device__ kln_point kln_apply(kln_rotor r, kln_point p)
{
    float4 scale = make_float4(0.0f, 2.0f, 2.0f, 2.0f);
    float4 rp1_xwyz = make_float4(r.p1.x,r.p1.w,r.p1.y,r.p1.z);
    float4 rp1_xzwy = make_float4(r.p1.x,r.p1.z,r.p1.w,r.p1.y);

    float4 t1 = r.p1 * rp1_xwyz;
    t1 -= r.p1.x * rp1_xzwy;
    t1 *= scale;

    float4 t2 = r.p1.x * rp1_xwyz;
    t2 += rp1_xzwy * r.p1;
    t2 *= scale;

    float4 t3 = r.p1 * r.p1;
    t3 += make_float4(r.p1.y,r.p1.x,r.p1.x,r.p1.x) * make_float4(r.p1.y,r.p1.x,r.p1.x,r.p1.x);
    float4 t4 = make_float4(r.p1.z,r.p1.w,r.p1.y,r.p1.z) * make_float4(r.p1.z,r.p1.w,r.p1.y,r.p1.z);
    t4 += make_float4(r.p1.w,r.p1.z,r.p1.w,r.p1.y) * make_float4(r.p1.w,r.p1.z,r.p1.w,r.p1.y);
    t3 -= t4 * make_float4(-1.0f, 1.0f, 1.0f, 1.0f);

    // TODO: provide variadic rotor-point application
    kln_point q;
    q.p3 = t1 * make_float4(p.p3.x,p.p3.w,p.p3.y,p.p3.z);
    q.p3 += t2 * make_float4(p.p3.x,p.p3.z,p.p3.w,p.p3.y);
    q.p3 += t3 * p.p3;
    return  q;
}







struct Quaternion_dev {
    float4 value;

    __device__ Quaternion_dev(float X, float Y, float Z, float W) {
        value.x = X;
        value.y = Y;
        value.z = Z;
        value.w = W;
    }

    __device__ Quaternion_dev(float3 t, float W) {
        value.x = t.x;
        value.y = t.y;
        value.z = t.z;
        value.w = W;
    }

    // Inverse
    __device__ Quaternion_dev Inv(Quaternion_dev a) {
        float norm = a.value.x * a.value.x + a.value.y * a.value.y + a.value.z * a.value.z + a.value.w * a.value.w;
        if (norm == 0.0f)
            return Quaternion_dev(0.0, 0.0, 0.0, 1.0);
        return Quaternion_dev(-a.value.x / norm, -a.value.y / norm, -a.value.z / norm, a.value.w / norm);
    }

    __device__ float3 Vector() {
        float3 r;
        r.x = value.x;
        r.y = value.y;
        r.z = value.z;
        return r;
    }

    __device__ float Scalar() {
        return value.w;
    }

    // Scalar Multiplication
    __device__ Quaternion_dev operator* (float s) {
        float3 v1;
        v1.x = s * value.x;
        v1.y = s * value.y;
        v1.z = s * value.z;
        return Quaternion_dev(v1, s * value.w);
    }

    // Multiplication
    __device__ Quaternion_dev operator* (Quaternion_dev b) {
        float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        float3 v2 = b.Vector();
        float w2 = b.Scalar();

        return Quaternion_dev(w1 * v2 + w2 * v1 + cross(v1, v2), w1 * w2 - dot(v1, v2));
    }

    // Addition
    __device__ Quaternion_dev operator+ (Quaternion_dev b) {
        float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        float3 v2 = b.Vector();
        float w2 = b.Scalar();

        return Quaternion_dev(v1 + v2, w1 + w2);
    }

    // Conjugate
    __device__ Quaternion_dev Conjugate() {
        return Quaternion_dev(-value.x, -value.y, -value.z, value.w);
    }

    // Magnitude
    __device__ float Magnitude() {
        return sqrt(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
    }

    __device__ float Dot(Quaternion_dev b) {
        float3 v1;
        v1.x = value.x;
        v1.y = value.y;
        v1.z = value.z;
        float w1 = value.w;
        float3 v2 = b.Vector();
        float w2 = b.Scalar();

        return w1 * w2 + dot(v1, v2);
    }

    //Normalize
    __device__ Quaternion_dev Normalize() {
        float norm = sqrt(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
        return Quaternion_dev(value.x / norm, value.y / norm, value.z / norm, value.w / norm);
    }

    //norm
    __device__ float Norm() {
        float norm = sqrt(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
        return norm;
    }
};







#endif
