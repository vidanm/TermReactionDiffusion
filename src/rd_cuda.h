#pragma once

struct RDCudaContext; 
struct RDHostContext;

void rd_init(RDHostContext** ctx, int W, int H);
void rd_upload(RDHostContext* ctx,const float *u,const float*v);
void rd_download(RDHostContext* ctx,float*u,float*v);
void rd_free(RDHostContext* ctx);
void launch_laplacien(RDHostContext* context);
