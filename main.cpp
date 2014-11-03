#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <windows.h>
#include <cmath>
#include <cassert>
#include <string>

#define ANALYTIC 1

float randf(){ return rand()/float(RAND_MAX+1); }
#define RANDU randf()

struct param{
    param();

    float dist;
    float fov;
    float global_time;

    int iw, ih;

    /************************************************************************/
    /*                                                                      */
    /************************************************************************/
    float g_global_scale;//the scaling factor for units
    int spp;
    float albedo;
    int trace_depth;
    float g_light_scale;
    char *envmap;
    float env_rotate;

    int env_light_count;

    float camera_origin[3];
    float camera_direction[3];
    float HG_mean_cosine;
    int show_bg;

    char *load_fn;

    float density_max;

    float radius;
    float c[4];
    int maxIter;
};
extern param _p;

param::param()
{
    dist = 2;

    global_time = 0;

//     envmap = "sun2.bin"; g_light_scale = 20; 
    envmap = "null"; g_light_scale = 1;
    env_light_count = 40000;
    env_rotate = 0;
    
    show_bg = 0;

    camera_origin[0] = 0;
    camera_origin[1] = 0.2;
    camera_origin[2] = dist;
    camera_direction[0] = 0;
    camera_direction[1] = -0.2;
    camera_direction[2] = -dist ;

    //////////////////////////////////////////////////////////////////////////
    load_fn = "null";

    global_time = 0.5;
    camera_origin[0] = dist*sinf(global_time);
    camera_origin[1] = 0.2;
    camera_origin[2] = dist*cosf(global_time);
    camera_direction[0] = -camera_origin[0];
    camera_direction[1] = -camera_origin[1];
//     camera_direction[1] = -0.2-camera_origin[1];
    camera_direction[2] = -camera_origin[2];
    iw = 300;
    ih = 300;
    spp=300;
    env_rotate=  180;
    g_global_scale = 50; //100; //400;

#define SET_FLOAT4(x,a,b,c,d) x[0]=a;x[1]=b;x[2]=c;x[3]=d;
    radius = 3;
    //     SET_FLOAT4(c, -1, 0.2, 0, 0)
    //     SET_FLOAT4(c, -0.291,-0.399,0.339,0.437)
    //     SET_FLOAT4(c, -0.2,0.4,-0.4,-0.4)
    //     SET_FLOAT4(c, -0.213,-0.0410,-0.563,-0.560)
    //     SET_FLOAT4(c, -0.2,0.6,0.2,0.2)
    //     SET_FLOAT4(c, -0.162,0.163,0.560,-0.599)
    SET_FLOAT4(c, -0.2,0.8,0,0)
        //     SET_FLOAT4(c, -0.445,0.339,-0.0889,-0.562)
        //     SET_FLOAT4(c, 0.185,0.478,0.125,-0.392)
        //     SET_FLOAT4(c, -0.450,-0.447,0.181,0.306)
        //     SET_FLOAT4(c, -0.218,-0.113,-0.181,-0.496)
        //     SET_FLOAT4(c, -0.137,-0.630,-0.475,-0.046)
        //     SET_FLOAT4(c, -0.125,-0.256,0.847,0.0895)
        //     maxIter = 20;
    maxIter = 30;

    fov = 30;
    albedo = 0.9;
    HG_mean_cosine = 0.7;//higher improves performance
    trace_depth=100;

}

param _p;


#ifndef util_h__
#define util_h__

#define hvec std::vector
#define dvec std::vector
#define RAW(x) (&x[0])

#define M_PI 3.14159265358979323846264338328 
#define M_PI_2 1.57079632679489661923
#define M_1_PI 0.318309886183790671538
#define eps 1e-6f
#define inf 1e10

float f_min(float a, float b) { return a<b ? a : b; }
float f_max(float a, float b) { return a>b ? a : b; }
float my_saturate(float x){ return x<0 ? 0 : x>1 ? 1 : x; }

float signed_map(int x, int n){
        return 2*(x/(float)n)-1;
}
float sq(float x){ return x*x; }


struct vec{
    float x, y, z;
    vec():x(0),y(0),z(0){}
    vec(float a_):x(a_),y(a_),z(a_){}
    vec(float x_, float y_, float z_):
    x(x_),y(y_),z(z_){}
    float& operator[](int n){ return (&x)[n]; }
    const float& operator[](int n) const { return (&x)[n]; }
};
vec operator+ (const vec& a, const vec& b) { return vec(a.x+b.x, a.y+b.y, a.z+b.z); }
vec operator+ (const vec& a, float b) { return vec(a.x+b, a.y+b, a.z+b); }
vec operator- (const vec& a, const vec& b) { return vec(a.x-b.x, a.y-b.y, a.z-b.z); }
vec operator- (const vec& a) { return vec(-a.x, -a.y, -a.z); }
vec operator* (const vec& a, float b) { return vec(a.x*b, a.y*b, a.z*b); }
vec operator* (float b, const vec& a) { return vec(a.x*b, a.y*b, a.z*b); }
vec normalize(const vec& a) { float len = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z)+eps; return a*(1.0f/len); }
inline float dot(const vec& a, const vec& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
vec mult (const vec& a, const vec& b) { return vec(a.x*b.x, a.y*b.y, a.z*b.z); }
vec cross(const vec& a, const vec& b) { return vec(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
float dist_sq(const vec& a, const vec& b) { vec d(a-b); return dot(d,d); }
float dist(const vec& a, const vec& b) { return sqrtf(dist_sq(a,b)); }
float length(const vec& a) { return sqrtf(dot(a,a)); }
vec my_saturate(const vec& a) { return vec(my_saturate(a.x),my_saturate(a.y),my_saturate(a.z)); }

#endif // util_h__
#ifndef ppmLoader_h__
#define ppmLoader_h__

typedef vec rgb_t;

struct dir_light{
    vec direction;
    vec radiance;
    float pdf;
};

typedef enum{CLAMP,CYCLIC,NONE} kind_;
class ppmLoader
{
private:
    hvec<float> th_h_rawPixels;

public:
    char *filename;
    int width, height;
    float *p_d_rawPixels;
    float *p_h_rawPixels;

    float *pu_host, *pu_device;
    float *Pu_host, *Pu_device;
    float *pv_host, *pv_device;
    float *Pv_host, *Pv_device;

    ppmLoader():p_d_rawPixels(NULL),p_h_rawPixels(NULL){}
    ~ppmLoader(){ 
        if(p_d_rawPixels){
            std::cout<<"Freeing envmap"<<std::endl;
            memset((void*)p_d_rawPixels, 0, th_h_rawPixels.size()*sizeof(float));
            free(p_d_rawPixels);
        }

        delete [] pu_host;
        delete [] Pu_host;
        delete [] pv_host;
        delete [] Pv_host;
        free(pu_device);
        free(Pu_device);
        free(pv_device);
        free(Pv_device);
    }
    int openImageFile_hdr( const char *filename, float scale = 1.0f, float rotate_degree = 0.0f);
    rgb_t getRGB(float x, float y, kind_ k=NONE);
    rgb_t getRGBdevice(float x, float y, kind_ k=NONE);
    void toDevice();
    void init_pdf();

    void presample_envmap();
    hvec<dir_light> h_light;
    dvec<dir_light> d_light;
};

int ppmLoader::openImageFile_hdr(
    const char *filename,
    float scale,
    float rotate_degree)
{
    width = 600, height = 300;
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (ifs.good()) {
        ifs.read(reinterpret_cast<char*>(&width),sizeof(int));
        ifs.read(reinterpret_cast<char*>(&height),sizeof(int));
    }
    else {
        fprintf(stderr, "Can't open file %s\n", filename);
        ifs.close();
    }

    th_h_rawPixels.resize(width * height * 3);
    p_h_rawPixels = RAW(th_h_rawPixels);
    if(!p_h_rawPixels) return 1;

    if (ifs.good()) {
        ifs.read(reinterpret_cast<char*>(p_h_rawPixels),width*height*3*sizeof(float));
    }
    else {
        float *p = p_h_rawPixels;
        for (unsigned j = 0; j < height; ++j) {
            for (unsigned i = 0; i < width; ++i) {
                float hl = expf(-100*(sq((i-width*0.0)/width)+sq((j-height*0.5)/height)))*30+0.3;
                p[0] =  hl*.98;
                p[1] =  hl*.95;
                p[2] =  hl*.92;
                p += 3;
            }
        }
    }

    hvec<float> temp = th_h_rawPixels;
    int rshift(rotate_degree*width/360.0f);
    printf("shift by %d pixels\n",rshift);
    for(int j=0; j<height; j++){
        for(int i=0; i<width; i++){
            int offset = i+j*width;
            int offset2 = (i-rshift+width)%width+j*width;
            th_h_rawPixels[offset*3  ] = temp[offset2*3  ];
            th_h_rawPixels[offset*3+1] = temp[offset2*3+1];
            th_h_rawPixels[offset*3+2] = temp[offset2*3+2];
        }
    }

    ifs.close();	

    for(int n=0; n<th_h_rawPixels.size(); n++){
        th_h_rawPixels[n] *= scale;
    }

    init_pdf();
    presample_envmap();

    toDevice();
    return 0;
}

void ppmLoader::toDevice()
{
    size_t total = th_h_rawPixels.size()*sizeof(float);
    p_d_rawPixels = (float*)malloc(total);
    memcpy((void*)p_d_rawPixels, (void*)p_h_rawPixels, total);

    pu_device= (float*)malloc((height          )*sizeof(float) );
    Pu_device= (float*)malloc((height+1        )*sizeof(float) );
    pv_device= (float*)malloc((height*width    )*sizeof(float) );
    Pv_device= (float*)malloc((height*(width+1))*sizeof(float) );
    memcpy((void*)pu_device, (void*)pu_host, (height          )*sizeof(float));
    memcpy((void*)Pu_device, (void*)Pu_host, (height+1        )*sizeof(float));
    memcpy((void*)pv_device, (void*)pv_host, (height*width    )*sizeof(float));
    memcpy((void*)Pv_device, (void*)Pv_host, (height*(width+1))*sizeof(float));
}

template <typename T>
static inline T clamp(T x, T a, T b)
{
    return x<a ? a : x>b ? b : x;
}

rgb_t ppmLoader::getRGB( float x, float y, kind_ k )
{
    int x_ = int(x*(width -1)+.5);
    int y_ = int(y*(height-1)+.5);
    int offset;

    switch(k){
    case CLAMP:
        x_ = clamp(x_, 0, width -1);
        y_ = clamp(y_, 0, height-1);
        offset = (x_+y_*width)*3;
        break;
    case CYCLIC:
        //         x_ %= width;
        //         y_ %= height;
        x_ = (x_ + width *1000) % width ;
        y_ = (y_ + height*1000) % height;
        offset = (x_+y_*width)*3;
        break;
    case NONE:
    default:
        if(x_<0 || x_>width-1 || y_<0 || y_>height-1)
            return rgb_t(0,0,0);
        offset = (x_+y_*width)*3;
        break;
    }
    return rgb_t(p_h_rawPixels[offset],
        p_h_rawPixels[offset+1], p_h_rawPixels[offset+2]);
}

/************************************************************************/
/* Inversion method for importance sampling                             */
/************************************************************************/
static void precompute1D(float *f, float *pf, float *Pf, int nf){
    float I = 0.0f;
    for(int n=0; n<nf; n++){
        I += f[n];
    }
    for(int n=0; n<nf; n++){
        pf[n] = f[n] / I;
    }
    Pf[0] = 0.0f;
    for(int n=1; n<nf; n++){
        Pf[n] = Pf[n-1] + pf[n-1];
    }
    Pf[nf] = 1.0f;
}

static void precompute2D(
    float *f, float *pu, float *Pu, float *pv, float *Pv, 
    int nu, int nv)
{
    hvec<float> rowsum(nu);
    for(int u=0; u<nu; u++){
        //         precompute1D(f+u*nv, pv+u*nv, Pv+u*nv, nv);
        precompute1D(f+u*nv, pv+u*nv, Pv+u*(nv+1), nv);//bug fixed
        float temp = 0.0f;
        for(int n=0; n<nv; n++){
            temp += (f+u*nv)[n];
        }
        rowsum[u] = temp;
    }
    precompute1D(RAW(rowsum), pu, Pu, nu);
}

static void sample1D(float *pf, float *Pf, float unif, int nf,
    float &x, float &p)
{
    int i=0;
    for(; i<nf; i++){
        if(Pf[i]<=unif && unif<Pf[i+1] || i==nf-1)//avoid overflow by additional check
            break;
    }
    float t = Pf[i+1] > Pf[i] ? 
        (Pf[i+1] - unif) / (Pf[i+1] - Pf[i])
        : 0;

    x = (1-t) * i + t * (i+1);
    p = pf[i];
}

static void sample2D(float *pu, float *Pu, float *pv, float *Pv,
    float unif1, float unif2, int nu, int nv, 
    float &u, float &v, float &pdf)
{
    float pdfu, pdfv;
    sample1D(pu, Pu, unif1, nu, u, pdfu);
    sample1D(pv+int(u)*nv, Pv+int(u)*(nv+1), unif2, nv, v, pdfv);
    pdf = pdfu * pdfv;
}

void ppmLoader::init_pdf(){
    hvec<float> luminance(width*height);
    float luminance_sum = 0;
    for(int n=0; n<width*height; n++){
        int offset = n*3;
        luminance[n] =
              p_h_rawPixels[offset  ] * 0.2126
            + p_h_rawPixels[offset+1] * 0.7152
            + p_h_rawPixels[offset+2] * 0.0722;
        luminance_sum += luminance[n];
    }

    pu_host = new float[height];
    Pu_host = new float[height+1];
    pv_host = new float[height*width];
    Pv_host = new float[height*(width+1)];

    precompute2D(RAW(luminance), pu_host, Pu_host,
        pv_host, Pv_host, height, width);

    //test
    float z0=0;
    int test_count = 10000;
    float integral = 0;
    for(int n=0; n<test_count; n++){
        float test_u, test_v, test_pdf;
        sample2D(pu_host, Pu_host, pv_host, Pv_host, randf(), randf(), height, width,
            test_u, test_v, test_pdf);
        z0 += test_pdf;
        integral += (luminance[int(test_u)*width+int(test_v)] / luminance_sum) / test_pdf;

    }
    integral /= test_count;//integrand is luminance[i]/luminance_sum, should integrate to 1
    printf("z0=%f\n",z0);
    printf("z1=%f\n",float(test_count)/(width*height));
    printf("int=%f\n",integral);
}

void ppmLoader::presample_envmap()
{
    int N = int(sqrtf(_p.env_light_count));
    int n_light = N*N;
    hvec<float> e1, e2;
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            e1.push_back((i+randf())/float(N));
            e2.push_back((j+randf())/float(N));
        }
    }

    for(int n=0; n<n_light; n++)
    {
        float test_u, test_v, test_pdf;
        sample2D(pu_host, Pu_host, pv_host, Pv_host, e1[n], e2[n], height, width, 
            test_u, test_v, test_pdf);

        int pixel_idx = int(test_u)*width+int(test_v);
        vec sample_radiance (
            p_h_rawPixels[pixel_idx*3  ],
            p_h_rawPixels[pixel_idx*3+1],
            p_h_rawPixels[pixel_idx*3+2]
        );

        float theta = (int(test_u)+0.5)/float(height) * M_PI;
        float phi = (int(test_v)+0.5)/float(width) * (2.0f * M_PI);//azimuthal
        vec sample_dir = vec(sinf(theta)*sinf(phi),cosf(theta),sinf(theta)*-cosf(phi));

        float sample_pdf = test_pdf * ((float(width)*float(height))/(2.0f*M_PI*M_PI*sinf(theta)));//warpping
        dir_light dl;
        dl.direction = sample_dir;
        dl.radiance = sample_radiance;
        dl.pdf = sample_pdf;

        h_light.push_back(dl);
    }

    d_light = h_light;

    printf(".#light=..%d...\n",d_light.size());
}

ppmLoader ppm;
#endif // ppmLoader_h__

#undef min
#undef max
#define DEFAULT_DIM 256

static void save_hdr_image(const float *fb, int width, int height, const char *filename)
{
    std::ofstream ofstream_bin;
    ofstream_bin.open(filename, std::ios::out | std::ios::binary);
    if (!ofstream_bin.is_open()) {
        printf("%s not opened\n",filename);
        return;
    }

    ofstream_bin.write(reinterpret_cast<const char*>(&width), sizeof(int));
    ofstream_bin.write(reinterpret_cast<const char*>(&height), sizeof(int));
    ofstream_bin.write(reinterpret_cast<const char*>(fb),width*height*3*sizeof(float));
    ofstream_bin.close();

    printf("hdr::=======%f, %f=======\n",
        *std::min_element(fb,fb+width*height*3),
        *std::max_element(fb,fb+width*height*3)); 

    //ppm examination
    std::ifstream ifstream_bin;
    ifstream_bin.open(filename, std::ios::in | std::ios::binary);
    if (!ifstream_bin.is_open()) {
        return;
    }

    struct ppm_writer
    {
        ppm_writer(){}
        float clamp(float x){ return x<0 ? 0 : x>1 ? 1 : x; }
        void operator()(float *fb, int width, int height, const char *fn){
            std::ofstream ofs(fn, std::ios::out | std::ios::binary);
            ofs << "P6\n" << width << " " << height << "\n255\n";
            unsigned char *bufByte = new unsigned char[width * height * 3];
            for (unsigned n = 0; n < width * height *3; ++n) {
                bufByte[n] = (unsigned char)(clamp(fb[n])*255);
            }
            ofs.write(reinterpret_cast<char*>(bufByte), width * height *3);
            ofs.close();
            delete [] bufByte;
        }
    }save_ppm;

    ifstream_bin.read(reinterpret_cast<char*>(&width),sizeof(int));
    ifstream_bin.read(reinterpret_cast<char*>(&height),sizeof(int));

    std::vector<float> buf(width*height*3);

    ifstream_bin.read(reinterpret_cast<char*>(&buf[0]),width*height*3*sizeof(float));
    ifstream_bin.close();

    float mmax = *std::max_element(buf.begin(),buf.end());
    for(int n=0; n<width*height*3; n++){
        buf[n] /= mmax;
    }
    save_ppm(&buf[0],width,height,"_additional.ppm");
}

struct ray{
    vec o, d;
    vec invdir;
    int sign[3];
    ray(){}
    ray(const vec& o_, const vec& d_):o(o_),d(normalize(d_)){
        invdir.x = 1.0f / d.x;
        invdir.y = 1.0f / d.y;
        invdir.z = 1.0f / d.z;
        sign[0] = (invdir.x < 0);
        sign[1] = (invdir.y < 0);
        sign[2] = (invdir.z < 0);
    }
    vec advance(float t) const { return o + d * t; }
};
struct shade_rec{
    float t_near;
    vec normal;
    vec color;
};

int index_xyz(int x, int y, int z, int N, int NN) { return x+y*N+z*NN; }
int index_xzy(int x, int y, int z, int N, int NN) { return x+z*N+y*NN; }
int index_yxz(int x, int y, int z, int N, int NN) { return y+x*N+z*NN; }
int index_yzx(int x, int y, int z, int N, int NN) { return y+z*N+x*NN; }
int index_zxy(int x, int y, int z, int N, int NN) { return z+x*N+y*NN; }
int index_zyx(int x, int y, int z, int N, int NN) { return z+y*N+x*NN; }
#define index_convention index_xyz



void quatSq(float q[4])
{
    float r0;

    vec q_yzw(q[1], q[2], q[3]);

    r0 = q[0]*q[0] - dot(q_yzw, q_yzw);
    vec r_yzw( q_yzw*(q[0]*2) );

    q[0] = r0;
    q[1] = r_yzw.x;
    q[2] = r_yzw.y;
    q[3] = r_yzw.z;
}

void add(float a[4], float b[4]){
    a[0]+=b[0];
    a[1]+=b[1];
    a[2]+=b[2];
    a[3]+=b[3];
}
float length2(float a[4]){
    return a[0]*a[0] + 
           a[1]*a[1] + 
           a[2]*a[2] + 
           a[3]*a[3];
}

float eval_fractal(const vec& pos, float radius, float c[4], int maxIter){

    float q[4] = { pos.x*radius, pos.y*radius, pos.z*radius, 0 };

    int iter = 0;
    do 
    {
        quatSq(q);
        add(q,c);
    } while (length2(q)<10.0f && iter++ <maxIter);

    //     return iter * (iter>5);
    //     return iter / float(maxIter);
    //     return log((float)iter+1) / log((float)maxIter);
    return (iter>maxIter*0.9);
}

int clampi(int x, int a, int b)
{
    return x<a ? a : x>b ? b : x;
}

template<typename T>
struct _tex3d_proxy{//isocube
    int N, total, NN;
    vec min, max;
    float l;
    T *data;
    //////////////////////////////////////////////////////////////////////////
    T& operator[] (int n) { return data[n]; }
    T& operator() (int i, int j, int k) { return data[index(i,j,k)]; }
    T fetch_gpu(const vec& pos) const {
#if ANALYTIC
        return eval_fractal(pos, _p.radius, _p.c, _p.maxIter);
#else
        if(outside(pos))return 0;
        vec p = remap_to_one(pos);
        float x_ = p.x;
        float y_ = p.y;
        float z_ = p.z;
        //trilinear
        x_ = x_ * N - 0.5;
        y_ = y_ * N - 0.5;
        z_ = z_ * N - 0.5;
        int i_0 = (int)x_;
        int j_0 = (int)y_;
        int k_0 = (int)z_;
        int i_1 = i_0 + 1;
        int j_1 = j_0 + 1;
        int k_1 = k_0 + 1;
        //fixed
        i_0 = clampi(i_0, 0, N-1);
        j_0 = clampi(j_0, 0, N-1);
        k_0 = clampi(k_0, 0, N-1);
        i_1 = clampi(i_1, 0, N-1);
        j_1 = clampi(j_1, 0, N-1);
        k_1 = clampi(k_1, 0, N-1);
        float u1 = x_ - i_0;
        float v1 = y_ - j_0;
        float w1 = z_ - k_0;
        float u0 = 1 - u1;
        float v0 = 1 - v1;
        float w0 = 1 - w1;
        return u0 *(v0*(w0*(data[index(i_0,j_0,k_0)])
            +           w1*(data[index(i_0,j_0,k_1)]))
            +       v1*(w0*(data[index(i_0,j_1,k_0)])
            +           w1*(data[index(i_0,j_1,k_1)])))
            +  u1 *(v0*(w0*(data[index(i_1,j_0,k_0)])
            +           w1*(data[index(i_1,j_0,k_1)]))
            +       v1*(w0*(data[index(i_1,j_1,k_0)])
            +           w1*(data[index(i_1,j_1,k_1)])))
            ;
#endif
    }
    //////////////////////////////////////////////////////////////////////////
    bool outside(const vec& pos) const {
        return pos.x<min.x || pos.y<min.y || pos.z<min.z
            || pos.x>max.x || pos.y>max.y || pos.z>max.z;
    }
    vec remap_to_one(const vec& pos) const {//normalize
        return (pos-min)*(1.0f/l);
    }
    int index(int x, int y, int z) const {
        return index_convention(x, y, z, N, NN);
    }
};

template<typename T>
struct _tex3d{//isocube
    int N, total, NN;
    vec min, max;
    float l;
    hvec<T> data;
    dvec<T> data_gpu;
    struct abs_compare{
        abs_compare(){}
        bool operator()(float a, float b){
            return a<b;
        }
        bool operator()(vec a, vec b){
            return length(a)<length(b);
        }
    };
    struct norm_op{
        float _min, _max, inv;
        norm_op(float a, float b):_min(a),_max(b){
            inv = 1.0/f_max(eps, _max-_min);
        }
        float operator()(float x){
            return (x-_min)*inv;
        }
        vec operator()(vec x){
            return (x)*inv;
        }
    };
    struct threshold_op{
        float a;
        threshold_op(float a_):a(a_){}
        float operator()(const float x) const {
            return x < a ? x : a;
        }
    };
    struct binarize_op{
        float a;
        binarize_op(float a_):a(a_){}
        float operator()(const float x) const {
            return x < a ? 0 : 1;
        }
    };
    struct binarize_op_smooth{
        float a,b,inv;
        binarize_op_smooth(float a_, float b_):a(a_),b(b_),inv(1.0/(b-a)){}
        float operator()(const float x) const {
            return x < a ? 0 : x > b ? 1 : (x-a)*inv;
        }
    };
    vec get_voxel_position(int i, int j, int k){
        return vec((i+0.5f)/(float)N,(j+0.5f)/(float)N,(k+0.5f)/(float)N)*l + min;
    }
    T& operator[] (int n) { return data[n]; }
    T& operator() (int i, int j, int k) { return data[index(i,j,k)]; }
    void threshold(float th){
        std::transform(data.begin(), data.end(), data.begin(), threshold_op(th));
    }
    void load_binary(const char *fn) 
    {
        if(sizeof(T)!=sizeof(float))return;
        int Nx, Ny, Nz;
        FILE *fp = fopen(fn,"rb");
        fread(&Nx,1,sizeof(int),fp);
        fread(&Ny,1,sizeof(int),fp);
        fread(&Nz,1,sizeof(int),fp);
        N=Nx, NN=N*N, total=N*N*N;
        data.resize(total);
        fread(&data[0],sizeof(float),total,fp);
        fclose(fp);
        printf("loaded %s <%d,%d,%d>\n",fn,Nx,Ny,Nz);
    }
    void sub_volume(vec min, vec max){
        hvec<T> sub((max.x-min.x)*(max.y-min.y)*(max.z-min.z));
        int new_N = max.x-min.x;
        int new_NN = new_N*new_N;
        int new_total = new_N*new_NN;
        for(int i=min.x; i<max.x; i++){
            for(int j=min.y; j<max.y; j++){
                for(int k=min.z; k<max.z; k++){
                    sub[index_xyz(i-min.x,j-min.y,k-min.z,new_N,new_NN)]
                        = data[index_xyz(i,j,k,N,NN)];
                }
            }
        }
        thrust::swap(sub,data);
        N = new_N;
        NN = new_NN;
        total = new_total;
        printf("subvolume <%d,%d,%d>\n",N,N,N);
    }
    void binarize(float level){
        float _min = *std::min_element(data.begin(), data.end(),abs_compare());
        float _max = *std::max_element(data.begin(), data.end(),abs_compare());
        std::transform(data.begin(), data.end(), data.begin(), binarize_op(level*(_max-_min)+_min));
    }
    void binarize(float level1, float level2){
        float _min = *std::min_element(data.begin(), data.end(),abs_compare());
        float _max = *std::max_element(data.begin(), data.end(),abs_compare());
        std::transform(data.begin(), data.end(), data.begin(),
            binarize_op_smooth(level1*(_max-_min)+_min,level2*(_max-_min)+_min));
    }
    //meta(r) = (R^2-r^2)^2/R^4
    inline float meta(float r, float R0){
        float R2 = R0*R0;
        float R4 = R2*R2;
        float rr = r*r;
        if(rr>R2) return 0; 
        float t = R2-rr;
        return t*t/R4; 
    }
    void init_metaball(){
        vec center[200];
        for(int n=0; n<200; n++){
            center[n] = (vec(randf(),randf(),randf())*2.0f-1.0f)*0.3;
        }

        for(int k=0; k<N; k++){
            for(int j=0; j<N; j++){
                for(int i=0; i<N; i++){
                    vec pos = get_voxel_position(i,j,k);
                    for(int n=0; n<200; n++){
                        data[index(i,j,k)] = data[index(i,j,k)] + meta(length(pos-center[n]),0.04*randf());
                    }
                }
            }
        }
    }
    void load_binary_raw(const char *fn) 
    {
        if(sizeof(T)!=sizeof(float))return;
        FILE *fp = fopen(fn,"rb");
        N=DEFAULT_DIM, NN=N*N, total=N*N*N;
        data.resize(total);
        fread(&data[0],sizeof(float),total,fp);
        fclose(fp);
        printf("loaded %s <%d,%d,%d>\n",fn,N,N,N);
    }
    void load_field(const char *fn);
    void init_simple(){
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                for(int k=0; k<N; k++){
                    data[index(i,j,k)] = 
                        (int(ceil(5.0*(i+1)/N)+ceil(5.0*(j+1)/N)+ceil(5.0*(k+1)/N))&1)*1;
//                         (int(ceil(5.0*(i+1)/N)+ceil(5.0*(j+1)/N)+ceil(5.0*(k+1)/N))&1)*0.5+0.5;
//                         (int(ceil(5.0*(i+1)/N)+ceil(5.0*(j+1)/N)+ceil(5.0*(k+1)/N))&1)*(float(k)/N);
//                         (int(ceil(5.0*(i+1)/N)+ceil(5.0*(j+1)/N)+ceil(5.0*(k+1)/N))&1)*(1-float(k)/N);
//                         (int(ceil(5.0*(i+1)/N)+ceil(5.0*(j+1)/N)+ceil(5.0*(k+1)/N))&1)*(float(i)/N);
                }
            }
        }
    }
    void normalize(){
        float _min = *std::min_element(data.begin(), data.end(),abs_compare());
        float _max = *std::max_element(data.begin(), data.end(),abs_compare());
        printf("<%f,%f>\n",_min,_max);
        std::transform(data.begin(), data.end(), data.begin(), norm_op(_min, _max));
        _min = *std::min_element(data.begin(), data.end(),abs_compare());
        _max = *std::max_element(data.begin(), data.end(),abs_compare());
        printf("<%f,%f>\n",_min,_max);
    }
    T max_element() {
        return *std::max_element(data.begin(), data.end(),abs_compare());
    }
    _tex3d(int N_=DEFAULT_DIM, float l_=1, const vec& ref=vec(-0.5))
        :N(N_),NN(N_*N_),total(N_*N_*N_),l(l_),
        min(ref),max(ref+l_)
    {
        data.resize(total);
        data.assign(total, T(0.5));
        toDevice();
    }
    void toDevice(){ data_gpu = data; }
    T fetch(const vec& pos) const {
        if(outside(pos))return 0;
        vec p = remap_to_one(pos);
        float x_ = p.x;
        float y_ = p.y;
        float z_ = p.z;
        //trilinear
        x_ *= N-2;
        y_ *= N-2;
        z_ *= N-2;
        int i_0 = (int)x_;
        int j_0 = (int)y_;
        int k_0 = (int)z_;
        int i_1 = i_0 + 1;
        int j_1 = j_0 + 1;
        int k_1 = k_0 + 1;
        float u1 = x_ - i_0;
        float v1 = y_ - j_0;
        float w1 = z_ - k_0;
        float u0 = 1 - u1;
        float v0 = 1 - v1;
        float w0 = 1 - w1;
        return u0 *(v0*(w0*(data[index(i_0,j_0,k_0)])
                       +w1*(data[index(i_0,j_0,k_1)]))
                +   v1*(w0*(data[index(i_0,j_1,k_0)])
                       +w1*(data[index(i_0,j_1,k_1)])))
             + u1 *(v0*(w0*(data[index(i_1,j_0,k_0)])
                       +w1*(data[index(i_1,j_0,k_1)]))
                +   v1*(w0*(data[index(i_1,j_1,k_0)])
                       +w1*(data[index(i_1,j_1,k_1)])))
        ;
    }
    //////////////////////////////////////////////////////////////////////////
    bool outside(const vec& pos) const {
        return pos.x<min.x || pos.y<min.y || pos.z<min.z
            || pos.x>max.x || pos.y>max.y || pos.z>max.z;
    }
    vec remap_to_one(const vec& pos) const {//normalize
        return (pos-min)*(1.0f/l);
    }
    int index(int x, int y, int z) const {
        return index_convention(x, y, z, N, NN);
    }

    //////////////////////////////////////////////////////////////////////////
    _tex3d_proxy<T> get_proxy(){
        _tex3d_proxy<T> _proxy;
        _proxy.N        = N;
        _proxy.total    = total;
        _proxy.NN       = NN;
        _proxy.min      = min;
        _proxy.max      = max;
        _proxy.l        = l;
        _proxy.data     = RAW(data_gpu);

        return _proxy;
    }
};
typedef _tex3d<float> tex3d;
typedef _tex3d<vec> tex3d_vec;

typedef _tex3d_proxy<float> tex3d_proxy;
typedef _tex3d_proxy<vec> tex3d_vec_proxy;

//////////////////////////////////////////////////////////////////////////
// Template structure to pass to kernel
template < typename T >
struct KernelArray
{
    T*  _array;
    int _size;

    KernelArray(dvec<T>& _a){
        _array = RAW(_a);
        _size = (int)_a.size();
    }

    T& operator[](int n) { return _array[n]; }
};

bool intersect_vol(const ray& r, const vec& aabb_min, const vec& aabb_max, float& t_near, float& t_far){
    vec bounds[2] = {aabb_min,aabb_max};
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    tmin = (bounds[r.sign[0]].x - r.o.x) * r.invdir.x;
    tmax = (bounds[1-r.sign[0]].x - r.o.x) * r.invdir.x;
    tymin = (bounds[r.sign[1]].y - r.o.y) * r.invdir.y;
    tymax = (bounds[1-r.sign[1]].y - r.o.y) * r.invdir.y;
    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    tzmin = (bounds[r.sign[2]].z - r.o.z) * r.invdir.z;
    tzmax = (bounds[1-r.sign[2]].z - r.o.z) * r.invdir.z;
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    if (tmin > 0) t_near = tmin; else t_near = 0;
    if (tmax < inf) t_far = tmax;
    if(tmax>=inf){
        return false;
    }
    return true;
}


vec sampleHG(float g, float e1, float e2) {
    float s=2.0*e1-1.0, f = (1.0-g*g)/(1.0+g*s);
    float cost = 0.5*(1.0/g)*(1.0+g*g-f*f), sint = sqrtf(1.0-cost*cost);
    return vec(cosf(2.0f * M_PI * e2) * sint, sinf(2.0f * M_PI * e2) * sint, cost);
}

float HG_phase_function(float g, float cos_t){
    return (1-g*g)/(4.*M_PI*powf(1+g*g-2*g*cos_t,1.5));
}


vec envmap_sample_dir(vec r, const float *d_pixels, 
    int width, int height)
{
    float xi =(( r.x >= 0 ? atanf(r.z/r.x) : atanf(r.z/r.x) + M_PI) + M_PI_2) / (2 * M_PI), yi = acosf(r.y)/M_PI;

    int x_ = int(xi*(width -1)+.5);
    int y_ = int(yi*(height-1)+.5);
    if(x_<0 || x_>width-1 || y_<0 || y_>height-1)
        return vec(0,0,0);
    int offset = (x_+y_*width)*3;
    return vec(d_pixels[offset],d_pixels[offset+1], d_pixels[offset+2]);
}


float get_transmittance_woodcock(const vec& a, const vec& b, 
    const tex3d_proxy& pdensity_volume, const param& _p,  
        float inv_sigma_max, float inv_density_max)
{
    ray light_ray(a,normalize(b-a));

    float t_near, t_far;
    bool shade_vol = intersect_vol(light_ray, pdensity_volume.min, pdensity_volume.max, t_near, t_far);
    if(!shade_vol)
    {
        return 1;
    }

    float max_t = f_min(t_far, dist(a,b));

    int nsamp = 1; //2;
    float count = 0;

    for(int n=0; n<nsamp; n++)
    {
        /// woodcock tracking
        float dist = t_near;
        float density;

        while(1){
            dist += -logf(1-RANDU) * inv_sigma_max;
            if(dist >= max_t){
                count += 1;
                break;
            }
            vec pos = light_ray.advance(dist);
            density = pdensity_volume.fetch_gpu(pos);
            if(RANDU < density * inv_density_max)
            {
                break;
            }
        }
    }
    return count/nsamp;
}

void raytrace_volpath_multiscatter(
    KernelArray<ray> cam_ray_array,
    KernelArray<vec> pixels,
    tex3d_proxy pdensity_volume, 
    tex3d_proxy pemission_volume, 
    const float *d_pixels, int texWidth, int texHeight,
    int width, int height, const param _p,
    float *pu, float *Pu, float *pv, float *Pv,
    dir_light *d_light, int n_light, int idx, 
    float density_max)
{
    float albedo = _p.albedo; // sigma_s / sigma_t

    ray cr(cam_ray_array[idx]);
    vec radiance(0,0,0);
    vec throughput(1,1,1);

    /// for woodcock tracking
    float sigma_max = density_max * _p.g_global_scale;
    float inv_sigma_max = 1.0f / sigma_max;
    float inv_density_max = 1.0f / density_max;

    int max_depth(_p.trace_depth);
    for(int depth=0; depth<max_depth; depth++)
    {
        float t_near, t_far;
        bool shade_vol = intersect_vol(cr, pdensity_volume.min, pdensity_volume.max, t_near, t_far);

        if(!shade_vol && depth==0 && _p.show_bg){
            radiance = radiance + mult(envmap_sample_dir(cr.d, d_pixels, texWidth, texHeight),throughput);
            break;
        }

        vec front = cr.advance(t_near);
        vec back = cr.advance(t_far);

        if(shade_vol)
        {
            /// woodcock tracking
            vec pos = front;//current incident radiance evaluation point
            float dist = t_near;
            float density;

            int through = 0;
            while(1){
                dist += -logf(1-RANDU) * inv_sigma_max;
                if(dist >= t_far){
                    through = 1;//transmitted through the volume, probability is 1-exp(-optical_thickness)
                    break;
                }
                pos = cr.advance(dist);
                density = pdensity_volume.fetch_gpu(pos);
                if(RANDU < density * inv_density_max)//sigma_t = density * g_sigma_t(scale factor)
                {
                    break;
                }
            }

            if(0==through)
            {
                throughput = throughput * albedo; //all subsequent light evaluations are scattered
 
                // the sun light may be strongly directional, but is always scattered by atmosphere,
                // hence modeled by IBL (depends on Mie phase function for directionality)
                {
                    dir_light envmap_dl = d_light[int(RANDU*n_light)];
                    float transmittance = 
                        get_transmittance_woodcock(pos, pos+envmap_dl.direction*1000.0f, pdensity_volume, _p, 
                        inv_sigma_max, inv_density_max);
                    vec attenuated_radiance = 
                        envmap_dl.radiance
                        * (transmittance);
                    radiance = radiance +
                        mult(attenuated_radiance,throughput) * 
                        (HG_phase_function(_p.HG_mean_cosine, dot(cr.d, envmap_dl.direction))
                        / envmap_dl.pdf);
                }

                vec dir = sampleHG(_p.HG_mean_cosine,RANDU, RANDU);
                vec ref_dir(2,3,5);
                ref_dir = normalize(ref_dir);
                vec u = normalize(cross(cr.d, ref_dir));
                vec v = cross(cr.d, u);

                dir = u * dir.x + v * dir.y + cr.d * dir.z;//by construction of the sample coordinates, dir is guaranteed to be unit
                cr = ray(pos, dir);
            }
            else
            {
                if(depth==0 && _p.show_bg)
                {
                    radiance = radiance + mult(envmap_sample_dir(cr.d, d_pixels, texWidth, texHeight),throughput);
                }
                
                break;
            }

        }
    }

    for(int n=0; n<3; n++)
        if(radiance[n]!=radiance[n]) radiance[n]=0; //NAN? 

    pixels[idx] = pixels[idx] + radiance;
}

tex3d *g_vol;

struct vec_scale_op{
    float _s;
    vec_scale_op(float s):_s(s){}
    vec operator()(vec x){
            return x*_s;
    }
};

void render() 
{
    tex3d& vol = *g_vol;

    float dist = _p.dist;
    float fov = _p.fov;

    vec lookat(0,0,0);
    vec up(0,1,0);
    vec cam_o(_p.camera_origin[0],_p.camera_origin[1],_p.camera_origin[2]);
    vec cam_d(normalize(vec(_p.camera_direction[0],_p.camera_direction[1],_p.camera_direction[2])));
    vec cam_x(normalize(cross(cam_d,up)));
    vec cam_y(cross(cam_x,cam_d));

    _p.global_time += 0.1;

    dvec<vec> fb(_p.iw*_p.ih);
    hvec<ray> hcam_rays(_p.iw*_p.ih);
    for(int j=0; j<_p.ih; j++){
        for(int i=0; i<_p.iw; i++){
            ray r(cam_o, cam_d
                +cam_x*(signed_map(i,_p.iw)*tan(fov*0.5/180.0f*M_PI))
                +cam_y*(-signed_map(j,_p.ih)*tan(fov*0.5/180.0f*M_PI)));
            int offset = (i+j*_p.iw);
            hcam_rays[offset] = r;
        }
    }
    dvec<ray> cam_rays = hcam_rays;
    printf("DEBUG::launching kernel\n");

    std::fill(fb.begin(),fb.end(),vec(0));
    int spp = _p.spp;
    for(int k=0; k<spp; k++)
    {
        for(int idx=0; idx<_p.iw*_p.ih; idx++)
        {
            raytrace_volpath_multiscatter
                (cam_rays, fb, 
                vol.get_proxy(), vol.get_proxy(),
                ppm.p_d_rawPixels, ppm.width, ppm.height,
                _p.iw, _p.ih, _p, 
                ppm.pu_device, ppm.Pu_device, ppm.pv_device, ppm.Pv_device,
                RAW(ppm.d_light), ppm.d_light.size(), idx, _p.density_max);
        }
        printf("\rfinished %.2f%%",100.0f*float(k+1)/float(spp));
    }
    std::transform(fb.begin(), fb.end(), fb.begin(), vec_scale_op(1.0f/spp));

    printf("\rDEBUG::kernel finished\n");

    float *fb1 = reinterpret_cast<float*>(&fb[0]);
    printf("color range: [%f, %f] \n",
        *std::min_element(fb1, fb1+_p.iw*_p.ih*3),
        *std::max_element(fb1, fb1+_p.iw*_p.ih*3));

    //not tonemapped
    save_hdr_image(fb1, _p.iw, _p.ih, "_additional.bin");
}

struct scale_op{
    float a;
    scale_op(float a_):a(a_){}
    float operator()(float x){
        return x*a;
    }
};

int main(int argc, char *argv[])
{
    printf("DEBUG:: begin program\n");
    tex3d vol(DEFAULT_DIM,1,vec(-0.5));

    vol.init_simple();
//     vol.load_binary(_p.load_fn);
//     vol.normalize();
//     vol.binarize(0.02);

    _p.density_max = vol.max_element();
    printf("maximum density is %f\n",_p.density_max);

    g_vol = &vol;

    g_vol->toDevice();

    printf("init complete\n");

    ppm.openImageFile_hdr(_p.envmap,_p.g_light_scale,_p.env_rotate);


    printf("DEBUG::preparing to render\n");

    render();

    return 0;
}
