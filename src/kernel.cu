

__global__ void laplace(float *t1_gpu, float *t2_gpu,int m, int n){
  float delta;
  int i = threadIdx.x;
  int j = threadIdx.y;
  float tim1;
  float tip1;
  float tjm1;
  float tjp1;
  int idx_tim1 = (i == 0 ? m-1: i-1)+j*m ;
  int idx_tip1 = (i == m-1 ? 0 : i+1)+j*m;
  int idx_tjm1 = i+(j == 0 ? n-1: j-1)*m;
  int idx_tjp1 = i+(j == n-1 ? 0 : j+1)*m; 
  float x = 0.0 + 2.0 * float(j) / float(n-1);
  float y = 0.0 + 1.0 * float(m-1-i) / float(m-1);

  tim1 = t1_gpu[idx_tim1];
  tip1 = t1_gpu[idx_tip1];
  tjm1 = t1_gpu[idx_tjm1];
  tjp1 = t1_gpu[idx_tjp1];
  delta = 1.0 / (float(m-1));
      // laplacien[idx(x,y,max_col)] = (a+b+c+d-4.0*w)/1.;
  t2_gpu[i+j*m] = tim1+tip1+tjm1+tjp1-4.0*t1_gpu[i+j*m];
}

// void launch_kernel() {
//   hello_kernel<<<1, 1>>>();
//   cudaDeviceSynchronize();
// }
