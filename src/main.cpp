#include "rd_cuda.h"
#include <algorithm>
#include <vector>
#include <cstddef>
#include <span>
#include <cmath>
#include <notcurses/nckeys.h>
#include <notcurses/notcurses.h>

const char* CHARSTYLE = "0";
const float FEED_RATE = 0.025;
const float KILL_RATE = 0.055;
const float DIFFUSION_RATE_U = 0.16;
const float DIFFUSION_RATE_V = 0.08;
const float DT = 0.5;

inline size_t idx(size_t x, size_t y,size_t W) { return y * W + x;}

void laplacien_discret(std::vector<float>& laplacien,std::span<float> vec,size_t max_row,size_t max_col) 
{
  for (int x = 0;x<max_col;x++){
    for (int y = 0;y<max_row;y++) {
      size_t xp = (x+1) % max_col;
      size_t xm = (x + max_col -1) %max_col;
      size_t yp = (y+1) % max_row;
      size_t ym = (y + max_row -1) % max_row;

      float a = vec[idx(xp,y,max_col)];
      float b = vec[idx(xm,y,max_col)];
      float c = vec[idx(x,yp,max_col)];
      float d = vec[idx(x,ym,max_col)];
      float w = vec[idx(x,y,max_col)];
      laplacien[idx(x,y,max_col)] = (a+b+c+d-4.0*w)/1.;
    }
  }
}

void calculate_u(
    std::vector<float>& u,
    std::span<float> v,
    std::span<float> vec_laplacien_u,
    std::span<float> vec_laplacien_v, 
    size_t max_row,
    size_t max_col)
{
  for (int x = 0;x < max_col;x++){
    for (int y = 0;y < max_row;y++) {
      auto previous_u = u[idx(x,y,max_col)];
      auto previous_v = v[idx(x,y,max_col)];
      auto laplacien_u = vec_laplacien_u[idx(x,y,max_col)];
      auto laplacien_v = vec_laplacien_v[idx(x,y,max_col)];
      auto reaction = -previous_u*(previous_v*previous_v)+FEED_RATE*(1-previous_u)+(DIFFUSION_RATE_U*laplacien_u); 
      u[idx(x,y,max_col)] = std::clamp(previous_u + reaction * DT,0.f,1.f);
    }
  }
}

void calculate_v(
    std::span<float> u,
    std::vector<float>& v,
    std::span<float> vec_laplacien_u,
    std::span<float> vec_laplacien_v, 
    size_t max_row,
    size_t max_col)
{
  for (int x = 0;x < max_col;x++){
    for (int y = 0;y < max_row;y++) {
      auto previous_u = u[idx(x,y,max_col)];
      auto previous_v = v[idx(x,y,max_col)];
      auto laplacien_u = vec_laplacien_u[idx(x,y,max_col)];
      auto laplacien_v = vec_laplacien_v[idx(x,y,max_col)];
      auto reaction = previous_u*(previous_v*previous_v)-(FEED_RATE+KILL_RATE)*previous_v+(DIFFUSION_RATE_V*laplacien_v);
      v[idx(x,y,max_col)] = std::clamp(previous_v + reaction * DT,0.f,1.f);
    }
  }
}

int main() {

  //notcurses
  struct notcurses_options opts = {};
  struct notcurses* nc = notcurses_init(&opts, nullptr);
  if (!nc) return 1;
  notcurses_mice_enable(nc,NCMICE_MOVE_EVENT);

  struct ncplane* stdp = notcurses_stdplane(nc);
  struct ncpalette* pal = ncpalette_new(nc);
  if (!pal) return -1;

  int i = 0;
  bool running = true;
  uint32_t palent;
  unsigned int rows,cols;
  notcurses_stddim_yx(nc, &rows, &cols);

  // Allocation CPU
  std::vector<float> u(rows*cols,1.0f);
  std::vector<float> v(rows*cols,0.0f);
  std::vector<float> vec_laplacien_u(rows*cols,0.0f);
  std::vector<float> vec_laplacien_v(rows*cols,0.0f);

  RDHostContext* d;
  rd_init(&d,cols,rows); //cudaMalloc

  while (running) {
    ncplane_erase(stdp);
    i = i +1;
    for (int x = 0; x < cols; x++){
      for (int y = 0; y < rows; y++) {
        auto value = u[idx(x,y,cols)];
        // int index = (int)round((((float)i/2000.)+value) *10.0);
        int index = (int)round(value*10.0);
        ncplane_set_fg_palindex(stdp, index);
        if (value < 0.95)
          ncplane_printf_yx(stdp,y,x,CHARSTYLE);
      }
    }

    notcurses_render(nc);

    struct ncinput ni;
    struct timespec ts = {0, 100 * 1000 * 1000}; // 100 ms
    int id = notcurses_get(nc,&ts,&ni);
    if (id == 'q')
      running = false;
    if (id == 'r')
    {
      std::fill(u.begin(),u.end(),1.0);
      std::fill(v.begin(),v.end(),0.0);
    }
    if (id == NCKEY_MOTION){
      for (int dy = -1; dy <= 1;dy++){
        for (int dx = -1; dx <=1;dx++) {
          auto x = (dx + ni.x) % cols;
          auto y = (dy + ni.y) % rows;

          u[idx(x,y,cols)] = 0.5;
          v[idx(x,y,cols)] = 0.25;
        }
      }
    }  
  
    rd_upload(d,u.data(),v.data());//Supprimer les laplaciens
    launch_laplacien(d);
    rd_download(d,u.data(),v.data());
    
    // laplacien_discret(vec_laplacien_u,u,rows, cols);
    // laplacien_discret(vec_laplacien_v,v,rows, cols);
    // calculate_u(u,v,vec_laplacien_u,vec_laplacien_v,rows,cols);
    // calculate_v(u,v,vec_laplacien_u,vec_laplacien_v,rows,cols);

  }

  rd_free(d);
  ncpalette_free(pal);
  notcurses_stop(nc);
  return 0;
}
