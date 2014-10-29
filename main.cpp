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


#ifndef texDisplay_h__
#define texDisplay_h__

#define __device__
#define __host__
#define __global__

float randf(){ return rand()/float(RAND_MAX+1); }
#define RANDU randf()


template <typename T>
class texBuffer
{
public:
    T *mBuf;
    T *mOffset;
    size_t mTotal;
    size_t mOffsetN;
    size_t *mSampCount;

    texBuffer():mBuf(0),mSampCount(0){}
    ~texBuffer()
    {
        if(mBuf) delete [] mBuf;
        if(mSampCount) delete [] mSampCount;
    }
    void init(size_t _mTotal)
    {
        mTotal = _mTotal;
        mBuf = new T[mTotal];
        mSampCount = new size_t[mTotal];
        mOffset = mBuf;
        mOffsetN = 0;
        reset();
    }
    void rewind() // return pointer to head
    {
        mOffset = mBuf;
        mOffsetN = 0;
    }
    void reset()
    {
        rewind();
        memset(mBuf,0,sizeof(T)*mTotal);
        memset(mSampCount,0,sizeof(size_t)*mTotal);
//         for(size_t n =0; n<mTotal; n++){
//             mBuf[n] = 0;
//             mSampCount[n] = 0;
//         }
    }
    texBuffer& operator<< (T _Val)
    {
        if(!(mOffsetN < mTotal)) rewind();
//         if(mOffsetN < mTotal)
//         {
            *(mOffset++) = _Val;
            mSampCount[mOffsetN]++;
            mOffsetN++;
//         }
        return *this;
    }
    inline T fetch()
    {
        return *mOffset;
    }
    inline T fetch(size_t n)
    {
        return mBuf[n];
    }
//     inline void set(size_t x, size_t y, _ColorChannnel _color)
//     {
//         mBuf[]
//     }
    void acc(size_t spp, T _Val)
    {
        if(mOffsetN < mTotal)
        {
            size_t accSamps = mSampCount[mOffsetN];
            *(mOffset++) =    
                (_Val * spp + mBuf[mOffsetN] * accSamps) / (accSamps + spp);
            mSampCount[mOffsetN] += spp;
            mOffsetN++;
        }
    }
    void acc(size_t spp, T _Val, size_t _OffsetN)
    {
        if(_OffsetN < mTotal)
        {
            size_t accSamps = mSampCount[_OffsetN];
            mBuf[_OffsetN] =    
                (_Val * spp + mBuf[_OffsetN] * accSamps) / (accSamps + spp);
            mSampCount[_OffsetN] += spp;
        }
    }
    inline T operator[] (size_t n){
        return mBuf[n];
    }
};

enum _ColorMode{_RGB,_GRAY};

class texDisplay
{
public:
    size_t mWidth, mHeight;
    size_t mTotal;
    enum _ColorMode mColor;
    bool _update;
    texBuffer<float> texBuf;
    texBuffer<double> texBuf2;

    float yVecLMB;
    float xVecLMB;
    float yVecMMB;
    float xVecMMB;
    float yVecRMB;
    float xVecRMB;

    texDisplay():bGLUT(false),externKey(0),externRender(0){}
    ~texDisplay();
    void initializeGL(void);
    void draw_func();
    void init( size_t _width, size_t _height, _ColorMode _kind, int argc, char **argv );
    void init( size_t _width, size_t _height, _ColorMode _kind );
    template <typename T> void fillTexture(T *buf);
    bool redisplay();
    void update();
    void update(size_t _width, size_t _height, float *d);
    void update(size_t _width, size_t _height, double *d);
    void loop();
    void reset();
    void rewind();
    void toSingle();
    void toDouble();
    void toSingle_with_correction(double gamma);
    void toDouble_with_correction(float gamma);
//     void printScreen();
    void printScreen(const char *filename);
//     bool bKeyboard;
    bool bGLUT;
    typedef void (*tKeyboardCB)(unsigned char key);
    typedef void (*tRenderCB)(void);
    void regKeyboard(tKeyboardCB externKey_);
    void regRender(tRenderCB externRender_);
    void resize(int w, int h);
    tKeyboardCB externKey;
    tRenderCB externRender;
};

extern texDisplay tD;

struct CColor{
    double h, wl, wr, c;
    CColor(double h_, double wl_, double wr_, double c_)
        :h(h_), wl(wl_), wr(wr_), c(c_){}
    inline double clamp(double x) { return x<0 ? 0 : x>1 ? 1 : x; }
    double operator() (double x){
        double w = x<c ? wl : wr;
        return clamp((w-abs(clamp(x)-c))/w*h);
    }
};

extern CColor bwrR;//bwr
extern CColor bwrG;//bwr
extern CColor bwrB;//bwr
extern CColor jetR;//jet
extern CColor jetG;//jet
extern CColor jetB;//jet
// CColor cmap(1,1,1,1,1);//grayscale
// CColor cmap(1,1,0,0,0);//grayscale

#endif // texDisplay_h__

texDisplay tD;
texDisplay::~texDisplay()
{
}

static int mouse_down[3];
static int x_anchor, y_anchor;


void motion(int x, int y)
{
    if(mouse_down[0])
    {
        tD.xVecLMB = (x - x_anchor);
        tD.yVecLMB = (y - y_anchor);
    }
    if(mouse_down[1])
    {
        tD.xVecMMB = (x - x_anchor);
        tD.yVecMMB = (y - y_anchor);
    }
    if(mouse_down[2])
    {
        tD.xVecRMB = (x - x_anchor);
        tD.yVecRMB = (y - y_anchor);
    }

    x_anchor = x;
    y_anchor = y;
}

void texDisplay::regKeyboard(tKeyboardCB externKey_)
{
    externKey = externKey_;
}

void texDisplay::regRender(tRenderCB externRender_)
{
    externRender = externRender_;
}

void texDisplay::init( size_t _width, size_t _height, _ColorMode _kind )
{
    bGLUT = false;
    mColor = _kind;
    mWidth = _width;
    mHeight = _height;
    mTotal = mWidth * mHeight * 3;
    _update = true;
    externKey = 0;

    texBuf.init(mTotal);
    texBuf2.init(mTotal);
    //     fillTexture(texBuf.mBuf);
    reset();
}

bool texDisplay::redisplay()
{
    if(_update){
        //         fillTexture(texBuf.mBuf);
        //         fillTexture(texBuf2.mBuf);
        _update = false;
        //         glutPostRedisplay();
        return true;
    }
    return false;
}
void texDisplay::update()
{
    if(!bGLUT) return;
    _update = true;
}



template <typename T>
void texDisplay::fillTexture(T *buf)
{
    for(size_t n=0; n<mTotal; n++)
    {
        buf[n] = rand()/(T)RAND_MAX;
    }
}

void texDisplay::reset()
{
    texBuf.reset();
    texBuf2.reset();
    //     fillTexture(texBuf.mBuf);
    //     fillTexture(texBuf2.mBuf);
}

void texDisplay::rewind()
{
    texBuf.rewind();
    texBuf2.rewind();
}

void texDisplay::toSingle()
{
    for(size_t n=0; n<mTotal; n++)
    {
        texBuf.mBuf[n] = (float)texBuf2.mBuf[n];
    }
}

void texDisplay::toDouble()
{
    for(size_t n=0; n<mTotal; n++)
    {
        texBuf2.mBuf[n] = (double)texBuf.mBuf[n];
    }
}

template <typename T>
static inline T clamp(T x)
{
    return x<0 ? 0 : x>1 ? 1 : x;
}

void texDisplay::toSingle_with_correction(double gamma)
{
    for(size_t n=0; n<mTotal; n++)
    {
        texBuf.mBuf[n] = (float)pow(clamp(texBuf2.mBuf[n]),1.0/gamma);
    }
}

void texDisplay::toDouble_with_correction(float gamma)
{
    for(size_t n=0; n<mTotal; n++)
    {
        texBuf2.mBuf[n] = (double)powf(clamp(texBuf.mBuf[n]),1.0f/gamma);
    }
}

// void texDisplay::printScreen()
// {
//     // Save result to a PPM image (keep these flags if you compile under Windows)       NOTE::especially std:ios::binary which is equivalent to "wb" in fprintf()
//     std::ofstream ofs("snapshot.ppm", std::ios::out | std::ios::binary);
//     ofs << "P6\n" << mWidth << " " << mHeight << "\n255\n";
//     for (unsigned i = 0; i < mWidth * mHeight * 3; ++i) {
//         ofs <<  (unsigned char)(min(1.0f, texBuf.mBuf[i]) * 255);
//     }
//     ofs.close();
// }

void texDisplay::printScreen(const char *filename)
{
    // Save result to a PPM image (keep these flags if you compile under Windows)       NOTE::especially std:ios::binary which is equivalent to "wb" in fprintf()
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs << "P6\n" << mWidth << " " << mHeight << "\n255\n";
    for (unsigned i = 0; i < mWidth * mHeight * 3; ++i) {
        ofs <<  (unsigned char)(min(1.0f, texBuf.mBuf[i]) * 255);
    }
    ofs.close();
}



struct param{
    param();

    float g_brightness     ;
    float g_density        ;
    float g_transfer_offset;
    float g_transfer_scale ;
    float g_sigma          ;


    //for shading
    float kd, ks, ka;
    float shininess;
    float STEP;

    float dist;
    float fov;
    float global_time;

    int iw, ih;

    float bg[3];

    bool pregen_grad;
    bool use_sobol;

    float h;
    float gradient_normalization_factor;

    float threshold;

    /************************************************************************/
    /*                                                                      */
    /************************************************************************/
    float g_sigma_t;//the scaling factor
    float g_density_scale;
    int spp;
    float albedo;
    int trace_depth;
    float g_light_scale;
    char *envmap;
    float env_rotate;
    int max_render;
    float sigma_t_channel[3];
    float albedo_channel[3];

    float sigma_s_channel[3];
    int env_light_count;

    float camera_origin[3];
    float camera_direction[3];
    float c_alpha;
    float c_beta;
    float HG_mean_cosine;
    int show_bg;

    char *load_fn;

    float density_max;
};
extern param _p;

#if 0
param::param()
{
    STEP = 0.0005;
    g_brightness          =7; //10.0f; //10.0f; //1.00f;
    g_density             = 2 * STEP;//premultiplied by raymarch integration stepsize
    g_transfer_offset     = 0.05;//0.06;//0.07;//0.06f; //0.05f;//0.06f;
    g_transfer_scale      =1.1;//3.00f;
    g_sigma = 0;

    kd = 5;
    ks = 15;
    ka = 0.2;
    shininess = 50;

    dist = 1;
    global_time = 0.5;1.4; 

    iw = 1000;
    ih = 1000;
}
#else
param::param()
{
    STEP = 0.001;//   0.0005;  //
    g_density             = 50 * STEP;//premultiplied by raymarch integration stepsize
    g_transfer_offset     = 0; //0.05;//0.06;//0.07;//0.06f; //0.05f;//0.06f;
    g_transfer_scale      =1.3;//2;//1; //3.00f;
    g_sigma = 0;
//     g_brightness          =   g_density / STEP *0.7; //0.6; //0.3; //only control medium emission
    g_brightness          =   50 *4; //0.6; //0.3; //only control medium emission

    kd = 5;
    ks = 15;
    ka = 0.4;

    dist = 2; // 1;// 
    fov = 45;
    global_time = 0.5; 0.2;  1.4; 

    iw = 600;
    ih = 600;
//     iw = 1600;
//     ih = 1600;

    bg[0] = 0;
    bg[1] = 0;
    bg[2] = 0;

    shininess = 50;

    pregen_grad = 0;
    use_sobol = 0;

    //////////////////////////////////////////////////////////////////////////
    h=1.0/200; // 1.0/512; //    //0.1;
    gradient_normalization_factor= 1; //5; //0.5; //to normalize the gradient globally

    threshold = 2000;

    /************************************************************************/
    /*                                                                      */
    /************************************************************************/
//     g_sigma_t = 200;          //++++++++++++++++++++
//     g_density_scale = 5;      //++++++++++++++++++++
//     g_sigma_t = 40;
//     g_density_scale = 25;
    g_sigma_t = 100;
    g_density_scale = 4;

//     spp = 2000;
//     spp = 20;
    spp = 1000;
//     spp = 100;
//     spp = 3000;
    albedo = 0.8; //0.97; //0.9;
    trace_depth = 11; //2; // ##for buggy version, the overflowed path (which is NOT through) is excessively illuminated
                        //to correct it, must evaluate final exiting transmittance for that path(either raymarching or woodcock statistics)

    global_time = 0;

    g_light_scale = 2; //1;

//     envmap = "envmap2.bin";//++++++++++++++++
//     envmap = "envmap.bin";
//     envmap = "sun.bin"; g_light_scale = 20; //front 10; //3;//where importance sampled direct lighting matters+++++++++++++++++
//     envmap = "sun2.bin"; g_light_scale = 20; //back 3;//where importance sampled direct lighting matters
//     envmap = "skydome_00040.bin"; g_light_scale = 2; 
    envmap = "null"; g_light_scale = 10;//cloud
    env_light_count = 40000; //400; //100; //2000; //fireflies seem to be caused by bad sampled light(s), so increasing the candidate count
                //kind of solves the problem. But it also could be the problem of sampling some lights excessively
    env_rotate = 0;

    max_render = 1; //60;
    
    //////////////////////////////////////////////////////////////////////////
    //multi-channel tests
    //considering the implementation, sigma_t is actually (sigma_s+sigma_a)*scale_factor, but conceptually sigma_a can be 1 for convenience, leaving scale_factor as desired
    //                     clr                       clr                   iso        fwrd       fwrd       ref                 cool
    sigma_t_channel[0] =   0.70f                ;  //2.00;    //2.00;    //4.00;    //4.47;    //.7 ;    //.7 ;    //.9 ;    //.9 ;    //.9;    //1 ;      //1  ;
    sigma_t_channel[1] =   1.22f                ;  //2.00;    //2.00;    //4.00;    //4.47;    //.7 ;    //.7 ;    //.8 ;    //.8 ;    //.8;    //1 ;      //1  ;
    sigma_t_channel[2] =   1.90f                ;  //2.00;    //2.00;    //4.00;    //4.47;    //.7 ;    //.7 ;    //.7 ;    //.7 ;    //.7;    //1 ;      //1  ;
    albedo_channel[0]  =   0.70f/(0.70f+0.0014f);  //.92 ;    //.999;    //.999;    //.992;    //.95;    //.95;    //.95;    //.95;    //.8;    //.7;      //0.8;
    albedo_channel[1]  =   1.22f/(1.22f+0.0025f);  //.99 ;    //.999;    //.999;    //.992;    //.95;    //.95;    //.97;    //.95;    //.8;    //.8;      //0.8;
    albedo_channel[2]  =   1.90f/(1.90f+0.0142f);  //.999;    //.999;    //.999;    //.992;    //.95;    //.95;    //.98;    //.95;    //.8;    //.9;      //0.8;

    trace_depth =  80; //20; //30;  // {80/30 not much differennt for albedo=.95}
    spp = 30; //1000; //3000;

    /////
//     g_density_scale = 0.1;//debug for sphere
    trace_depth = 20*4; //debug for directlight
    spp = 200;//30;// 10; // 1000; //1000; // 1;//         //debug for directlight
//     iw = 200;     //debug for directlight
//     ih = 200;     //debug for directlight


    sigma_s_channel[0] = sigma_t_channel[0] * albedo_channel[0];
    sigma_s_channel[1] = sigma_t_channel[1] * albedo_channel[1];
    sigma_s_channel[2] = sigma_t_channel[2] * albedo_channel[2];
    printf("param::sigma_s<%f,%f,%f>\n",sigma_s_channel[0],sigma_s_channel[1],sigma_s_channel[2]);

    show_bg = 0;

    camera_origin[0] = 0;
    camera_origin[1] = 0.2;
    camera_origin[2] = dist;
    camera_direction[0] = 0;
    camera_direction[1] = -0.2;
    camera_direction[2] = -dist ;

    //////////////////////////////////////////////////////////////////////////
//     camera_origin[0] = 0      -0.5 * 0.18; //0.14 ;
//     camera_origin[1] = -0.2   -0.4 * 0.18; //0.14 ;
//     camera_origin[2] = 0.5    -2   * 0.18; //0.14 ;
//     camera_origin[0] = 0      -0.5 * 0.18; //0.14 ;
//     camera_origin[1] = -0.25   -0.4 * 0.18; //0.14 ;
//     camera_origin[2] = 0.5    -2   * 0.18; //0.14 ;
//     camera_direction[0] = -0.5;
//     camera_direction[1] = -0.4;
//     camera_direction[2] = -2  ;
    sigma_t_channel[0] =   1.00f; 
    sigma_t_channel[1] =   1.00f; 
    sigma_t_channel[2] =   1.00f; 
    albedo_channel[0]  =  0.99; // 0.95; //0.9; //0.99;
    albedo_channel[1]  =  0.99; // 0.95; //0.9; //0.99;
    albedo_channel[2]  =  0.99; // 0.95; //0.9; //0.99;
//     c_alpha = -0.2; //-0.7; //-0.4; //-.9; //1;
//     c_beta = 1; //0;        //_p.albedo_channel[channel] * (_p.c_alpha*density+_p.c_beta)
    c_alpha =0;//-5;// -0.2;
    c_beta = 1;//5.95; //1.1;
    HG_mean_cosine = -0.7; //-0.4; //0.7; //0.96; //this backscattering HG setting worked fine 
                            //when only used for scattering events while the actual phase function 
                                        //for direct lighting was spherical
//     g_sigma_t = 400;
//     g_density_scale = 1;
//     g_sigma_t = 100;
//     g_density_scale = 4;
//     g_sigma_t = 200;
//     g_density_scale = 2;
    g_sigma_t = 200;       //++
    g_density_scale = 4;   //++
//     g_sigma_t = 100;         //++++++++++++++++
//     g_density_scale = 1;     //++++++++++++++++

//     load_fn = "300c.bin";
//     load_fn = "F:/GraphicsWork/render/SSS/Ifu/CLOUDS/300c.bin";//+++++++++++
//     load_fn = "F:/GraphicsWork/render/SSS/Ifu/CLOUDS/300d.bin";
//     load_fn = "F:/GraphicsWork/render/SSS/Ifu/CLOUDS/100c.bin";
//     load_fn = "vol_1.bin";
//     load_fn = "pyro_test_256.bin";
//     load_fn = "F:/GraphicsWork/render/SSS/Ifu/CLOUDS/100e.bin";
//     load_fn = "F:/GraphicsWork/render/SSS/Ifu/CLOUDS/300e.bin";
//     load_fn = "C:/Users/asus/Desktop/__TEMP__/ModernOPENGL/raycast/build/pyro400.bin";
    load_fn = "sample_cloud_beta2.bin";

    //nonspectral
    albedo = 0.99;
    spp = 600;//30; // three times(200*3) as for spectral version to achieve similar convergence
    HG_mean_cosine = 0.7; //0.4; //0.9; //0.95; //-0.7; // 0; //
//     albedo = 0.9999; trace_depth = 300;
//     g_sigma_t = 400;
//     g_density_scale = 8; //16;


    //
//     global_time = 4;
//     dist=1;
//     global_time = 4.5;   //+++++++++++++++++++++
//     fov=20;//24;//zoom in     //+++++++++++++++++++++
    camera_origin[0] = dist*sinf(global_time);
    camera_origin[1] = 0.2;
    camera_origin[2] = dist*cosf(global_time);
    camera_direction[0] = -camera_origin[0];
    camera_direction[1] = -camera_origin[1];
//     camera_direction[1] = 0.2-camera_origin[1];//++++++++++++++++++++
//     camera_direction[1] = -0.2-camera_origin[1];
    camera_direction[2] = -camera_origin[2];
    iw = 300;//*2;
    ih = 300;//*2;
    trace_depth=100; //300;
    spp=1000; //200;//30/3;
//     g_density_scale = 200;//equivalent of binarization at isolevel=0.02 with the previous setting
    env_rotate=  270; //180; //90;//1330; //300;//240;//210;//150;//120; //60;//30; //80; //
//     show_bg=1;
//     g_density_scale= 50;
    g_sigma_t = 50;       //++
    g_density_scale = 1;   //++

}
#endif

param _p;


#ifndef util_h__
#define util_h__

#define hvec std::vector
#define dvec std::vector
#define RAW(x) (&x[0])

#define FI //__forceinline__
#define Dev
#define HaD

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

    void sample_envmap();
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
                float hl = expf(-100*(sq((i-width*0.5)/width)+sq((j-height*0.5)/height)))*30+0.3;
                p[0] =  hl;
                p[1] =  hl;
                p[2] =  hl;
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
    sample_envmap();

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
/* Inversion method for importance sampling                               */
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

void ppmLoader::sample_envmap()
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
        vec sample_dir = vec(sinf(theta)*sinf(phi),cosf(theta),sinf(theta)*-cosf(phi));//in world coordinates, care with calibration with envmap sampling routine, trick is to offset and/or invert phi

        float sample_pdf = test_pdf * ((float(width)*float(height))/(2.0f*M_PI*M_PI*sinf(theta)));//for normalization over sphere
                                                                          //closer to the pole -> larger pdf
        dir_light dl;
        dl.direction = sample_dir;
        dl.radiance = sample_radiance;
        dl.pdf = sample_pdf;

        h_light.push_back(dl);
    }

    d_light = h_light;

    printf(".#light=..%d...\n",d_light.size());
}

void sample1D_CUDA(float *pf, float *Pf, float unif, int nf,
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

void sample2D_CUDA(float *pu, float *Pu, float *pv, float *Pv,
    float unif1, float unif2, int nu, int nv, 
    float &u, float &v, float &pdf)
{
    float pdfu, pdfv;
    sample1D_CUDA(pu, Pu, unif1, nu, u, pdfu);
    sample1D_CUDA(pv+int(u)*nv, Pv+int(u)*(nv+1), unif2, nv, v, pdfv);
    pdf = pdfu * pdfv;
}

int sample_envmap_inverse_method(vec& sample_dir, float& sample_pdf, 
    float *pu, float *Pu, float *pv, float *Pv, float e1, float e2,
    int width, int height)
{
    float test_u, test_v, test_pdf;
    sample2D_CUDA(pu, Pu, pv, Pv, e1, e2, height, width, 
        test_u, test_v, test_pdf);

    float theta = int(test_u)/float(height) * M_PI;
    float phi = int(test_v)/float(width) * (2.0f * M_PI);//azimuthal

    sample_pdf = test_pdf * ((float(width)*float(height))/(2.0f*M_PI*M_PI*sinf(theta)));
    sample_dir = vec(sinf(theta)*sinf(phi),cosf(theta),sinf(theta)*-cosf(phi));

    return int(test_u)*width+int(test_v);
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
            +           w1*(data[index(i_0,j_0,k_1)]))
            +       v1*(w0*(data[index(i_0,j_1,k_0)])
            +           w1*(data[index(i_0,j_1,k_1)])))
            +  u1 *(v0*(w0*(data[index(i_1,j_0,k_0)])
            +           w1*(data[index(i_1,j_0,k_1)]))
            +       v1*(w0*(data[index(i_1,j_1,k_0)])
            +           w1*(data[index(i_1,j_1,k_1)])))
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
        FI HaD norm_op(float a, float b):_min(a),_max(b){
            inv = 1.0/f_max(eps, _max-_min);
        }
        FI HaD float operator()(float x){
            return (x-_min)*inv;
        }
        FI HaD vec operator()(vec x){
            return (x)*inv;
        }
    };
    struct threshold_op{
        float a;
        threshold_op(float a_):a(a_){}
        FI HaD float operator()(const float x) const {
            return x < a ? x : a;
        }
    };
    struct binarize_op{
        float a;
        binarize_op(float a_):a(a_){}
        FI HaD float operator()(const float x) const {
            return x < a ? 0 : 1;
        }
    };
    struct binarize_op_smooth{
        float a,b,inv;
        binarize_op_smooth(float a_, float b_):a(a_),b(b_),inv(1.0/(b-a)){}
        FI HaD float operator()(const float x) const {
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
        thrust::transform(data.begin(), data.end(), data.begin(), binarize_op(level*(_max-_min)+_min));
    }
    void binarize(float level1, float level2){
        float _min = *std::min_element(data.begin(), data.end(),abs_compare());
        float _max = *std::max_element(data.begin(), data.end(),abs_compare());
        thrust::transform(data.begin(), data.end(), data.begin(),
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
    FI HaD bool outside(const vec& pos) const {
        return pos.x<min.x || pos.y<min.y || pos.z<min.z
            || pos.x>max.x || pos.y>max.y || pos.z>max.z;
    }
    FI HaD vec remap_to_one(const vec& pos) const {//normalize
        return (pos-min)*(1.0f/l);
    }
    FI HaD int index(int x, int y, int z) const {
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

    FI HaD T& operator[](int n) { return _array[n]; }
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
    const tex3d_proxy& pdensity_volume, const param& _p, int channel, 
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
            float e2 = RANDU;
            float fraction = density * inv_density_max;//assuming that sigma_t = density * g_sigma_t(scale factor) * sigma_t_channel[channel], and that the latter two are constants
            if(e2 < fraction)
            {
                break;
            }
        }
    }
    return count/nsamp;
}

void raytrace_volpath_multiscatter_channel_directlight__importance_nonspectral(
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
    //////////////////////////////////////////////////////////////////////////

    ray cr(cam_ray_array[idx]);
    vec radiance(0,0,0);
    vec throughput(1,1,1);

    /// for woodcock tracking
    float sigma_max = density_max * _p.g_sigma_t;
    float inv_sigma_max = 1.0f / sigma_max;
    float inv_density_max = 1.0f / density_max;

    int max_depth(_p.trace_depth);
    for(int depth=0; depth<max_depth; depth++)
    {
        float t_near, t_far;
        bool shade_vol = intersect_vol(cr, pdensity_volume.min, pdensity_volume.max, t_near, t_far);

        if(!shade_vol && depth==0 && _p.show_bg){
            radiance = radiance +
                mult(envmap_sample_dir(cr.d, d_pixels, texWidth, texHeight),throughput)
                * _p.g_light_scale;
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
                float e2 = RANDU;
                float fraction = density * inv_density_max;//assuming that sigma_t = density * g_sigma_t(scale factor) * sigma_t_channel[channel], and that the latter two are constants
                if(e2 < fraction)
                {
                    break;
                }
            }

            if(0==through)
            {
                throughput = throughput * albedo; //all subsequent light evaluations are scattered
 
                dir_light envmap_dl = d_light[int(RANDU*n_light)];
                float transmittance = 
                    get_transmittance_woodcock(pos, pos+envmap_dl.direction*1000.0f, pdensity_volume, _p, 0,
                     inv_sigma_max, inv_density_max);
                vec attenuated_radiance = 
                    envmap_dl.radiance
                    * (_p.g_light_scale * transmittance);
                radiance = radiance +
                    mult(attenuated_radiance,throughput) * (
                    ((HG_phase_function(_p.HG_mean_cosine, dot(cr.d, envmap_dl.direction))) / envmap_dl.pdf)//fixed to use HG always
                    * (my_saturate(_p.c_alpha*density+_p.c_beta)));

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
                    radiance = radiance +
                        mult(envmap_sample_dir(cr.d, d_pixels, texWidth, texHeight),throughput)
                        * _p.g_light_scale;
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

void render(char *filename) 
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
            raytrace_volpath_multiscatter_channel_directlight__importance_nonspectral
                (cam_rays, fb, 
                vol.get_proxy(), vol.get_proxy(), //for simplicity use density for illumination strength, anyway, this routine is only for debugging woodcock tracking
                ppm.p_d_rawPixels, ppm.width, ppm.height,
                _p.iw, _p.ih, _p, 
                ppm.pu_device, ppm.Pu_device, ppm.pv_device, ppm.Pv_device,
                RAW(ppm.d_light), ppm.d_light.size(), idx, _p.density_max);
        }
        printf("\rfinished %.2f%%",100.0f*float(k+1)/float(spp));
    }
    std::transform(fb.begin(), fb.end(), fb.begin(), vec_scale_op(1.0f/spp));

    printf("\rDEBUG::kernel finished\n");


    hvec<vec> hfb(fb);
    for(int j=0; j<_p.ih; j++){
#pragma omp parallel for
        for(int i=0; i<_p.iw; i++){
            int offset = (i+j*_p.iw);
            vec c = hfb[offset];
            tD.texBuf.mBuf[offset*3  ] = c.x;
            tD.texBuf.mBuf[offset*3+1] = c.y;
            tD.texBuf.mBuf[offset*3+2] = c.z;
        }
    }

    printf("color range: [%f, %f] \n",
        *std::min_element(tD.texBuf.mBuf, tD.texBuf.mBuf+_p.iw*_p.ih),
        *std::max_element(tD.texBuf.mBuf, tD.texBuf.mBuf+_p.iw*_p.ih));

    tD.update();

    tD.printScreen(filename);
    printf("Rendered to %s\n",filename);

    //not tonemapped
    save_hdr_image(tD.texBuf.mBuf, _p.iw, _p.ih, "_additional.bin");
}

int main(int argc, char *argv[]){
    system("mkdir ppm");

    printf("DEBUG:: begin program\n");
    tex3d vol(DEFAULT_DIM,1,vec(-0.5));

    vol.init_simple();
//     vol.load_binary(_p.load_fn);
    vol.threshold(_p.threshold);
    vol.normalize();

    _p.density_max = vol.max_element();
    printf("maximum density is %f\n",_p.density_max);

    g_vol = &vol;

    g_vol->toDevice();

    printf("init complete\n");

    tD.init(_p.iw,_p.ih,_RGB);

    ppm.openImageFile_hdr(_p.envmap,1,_p.env_rotate);


    printf("DEBUG::preparing to render\n");

    for(int n=0; n<_p.max_render; n++)
    {
        char fn[256];
        sprintf(fn,"ppm/test_%05d.ppm",n);
        render(fn);
    }

    return 0;
}
