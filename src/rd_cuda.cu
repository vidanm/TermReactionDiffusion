#include "rd_cuda.h"

__constant__ double FEED_RATE = 0.025;
__constant__ double KILL_RATE = 0.055;
__constant__ double DIFFUSION_RATE_U = 0.16;
__constant__ double DIFFUSION_RATE_V = 0.08;
__constant__ double DT = 0.5;

__global__ void laplacien(float *t1_gpu, float *t2_gpu,int m, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;
  float tim1;
  float tip1;
  float tjm1;
  float tjp1;
  int idx_tim1 = (i == 0 ? m-1: i-1)+j*m ;
  int idx_tip1 = (i == m-1 ? 0 : i+1)+j*m;
  int idx_tjm1 = i+(j == 0 ? n-1: j-1)*m;
  int idx_tjp1 = i+(j == n-1 ? 0 : j+1)*m; 

  tim1 = t1_gpu[idx_tim1];
  tip1 = t1_gpu[idx_tip1];
  tjm1 = t1_gpu[idx_tjm1];
  tjp1 = t1_gpu[idx_tjp1];
  t2_gpu[i+j*m] = tim1+tip1+tjm1+tjp1-4.0*t1_gpu[i+j*m];
}

__global__ void u(float *lapl_u,float* u, float* v,float* out_u,int m,int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;
 
  float previous_u = u[i+j*m];
  float previous_v = v[i+j*m];
  float lap_u = lapl_u[i+j*m];

  float reaction = -previous_u*(previous_v*previous_v)+FEED_RATE*(1-previous_u)+(DIFFUSION_RATE_U*lap_u);
  out_u[i+j*m] = fminf(fmaxf(previous_u+reaction*DT, 0.f), 1.f);
}

__global__ void v(float* lapl_v,float* u, float* v,float* out_v,int m, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;
 
  float previous_u = u[i+j*m];
  float previous_v = v[i+j*m];
  float lap_v = lapl_v[i+j*m];

//auto reaction = previous_u*(previous_v*previous_v)-(FEED_RATE+KILL_RATE)*previous_v+(DIFFUSION_RATE_V*laplacien_v);
  float reaction = previous_u*(previous_v*previous_v)-(FEED_RATE+KILL_RATE)*previous_v+(DIFFUSION_RATE_V*lap_v);
  out_v[i+j*m] = fminf(fmaxf(previous_v+reaction*DT, 0.f), 1.f);
}

struct RDHostContext {
  float* d_laplacien_u_in;
  float* d_laplacien_u_out;
  float* d_laplacien_v_in;
  float* d_laplacien_v_out;
  float* d_u_in;
  float* d_u_out;
  float* d_v_in;
  float* d_v_out;
  int W,H;
};

void rd_init(RDHostContext** hctx, int W, int H) {
    *hctx = new RDHostContext{};
    (*hctx)->W = W;
    (*hctx)->H = H;

    size_t n = W * H * sizeof(float);

    // allocations GPU
    cudaMalloc(&(*hctx)->d_laplacien_u_in,  n);
    cudaMalloc(&(*hctx)->d_laplacien_u_out, n);
    cudaMalloc(&(*hctx)->d_laplacien_v_in,  n);
    cudaMalloc(&(*hctx)->d_laplacien_v_out, n);
    cudaMalloc(&(*hctx)->d_u_in,n);
    cudaMalloc(&(*hctx)->d_u_out,n);
    cudaMalloc(&(*hctx)->d_v_in,n);
    cudaMalloc(&(*hctx)->d_v_out,n);

}


void rd_upload(RDHostContext* ctx, 
    const float* u_data,
    const float* v_data) {
    size_t n = ctx->W * ctx->H * sizeof(float);

    cudaMemcpy(ctx->d_laplacien_u_in, u_data, n, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_laplacien_v_in, v_data, n, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_u_in, u_data, n, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_v_in, v_data, n, cudaMemcpyHostToDevice);
}

void rd_download(RDHostContext* ctx,
    float *u_data,
    float *v_data) {
    size_t n = ctx->W * ctx->H * sizeof(float);

    cudaMemcpy(u_data, ctx->d_u_out, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_data, ctx->d_v_out, n, cudaMemcpyDeviceToHost);
}

void rd_free(RDHostContext* ctx){
  cudaFree(ctx->d_laplacien_u_in);
  cudaFree(ctx->d_laplacien_u_out);
  cudaFree(ctx->d_laplacien_v_in);
  cudaFree(ctx->d_laplacien_v_out);
  cudaFree(ctx->d_u_in);
  cudaFree(ctx->d_u_out);
  cudaFree(ctx->d_v_in);
  cudaFree(ctx->d_v_out);
  cudaFree(ctx);
}

void launch_rd(RDHostContext* context) {

    dim3 threads(16,16);

    dim3 blocks(
        (context->W + threads.x - 1) / threads.x,
        (context->H + threads.y - 1) / threads.y
    );

    laplacien<<<blocks, threads>>>(context->d_laplacien_u_in, context->d_laplacien_u_out, context->W, context->H);
    laplacien<<<blocks, threads>>>(context->d_laplacien_v_in, context->d_laplacien_v_out, context->W, context->H);
    u<<<blocks, threads>>>(context->d_laplacien_u_out, context->d_u_in,context->d_v_in,context->d_u_out, context->W, context->H);
    v<<<blocks, threads>>>(context->d_laplacien_v_out,context->d_u_in,context->d_v_in,context->d_v_out, context->W, context->H);
}
