#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

// declare CUDA fields
T *dT , *dr , *dp , *dq , *ds;
T *dcc, *dff, *dss, *dww;  // matrix A center, south, west stencil

T *kw , *kc;  // (TNS1) matrix M-1 with 6 point stencil
T       *ks, *kse;
T       *kf, *kfe;

T *dpp;              // matrix P = sqrt(D)
T *drh, *dsg;        // partial dot products
T *dV;               // cell volume
T *dqB;              // Neumann BC bottom



// initialize CUDA fields
template <class T>
void cudaInit( T *hT,
		T *hV,
		T *hcc,
		T *hff,
		T *hss,
		T *hww,
		T *hqB,
		const int blocks,
		const int Nx,
		const int Ny,
		const int Nz)
{
	cudaMalloc((void**)&dT ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dr ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dV ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dp ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dq ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&ds ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dpp,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dcc,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dff,sizeof(T)*(Nx*Ny*Nz+Nx*Ny  ));
	cudaMalloc((void**)&dss,sizeof(T)*(Nx*Ny*Nz+Nx     ));
	cudaMalloc((void**)&dww,sizeof(T)*(Nx*Ny*Nz+1      ));
	cudaMalloc((void**)&drh,sizeof(T)*(blocks          ));
	cudaMalloc((void**)&dsg,sizeof(T)*(blocks          ));
	cudaMalloc((void**)&dqB,sizeof(T)*(Nx*Ny           ));


	cudaMalloc((void**)&kc ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny)); // 6 diagonals for M^-1
	cudaMalloc((void**)&kf ,sizeof(T)*(Nx*Ny*Nz+Nx*Ny  ));
	cudaMalloc((void**)&kfe,sizeof(T)*(Nx*Ny*Nz+Nx*Ny-1));
	cudaMalloc((void**)&ks ,sizeof(T)*(Nx*Ny*Nz+Nx     ));
	cudaMalloc((void**)&kse,sizeof(T)*(Nx*Ny*Nz+Nx-1   ));
	cudaMalloc((void**)&kw ,sizeof(T)*(Nx*Ny+1         ));


	cudaMemcpy(dT ,hT ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny),cudaMemcpyHostToDevice);
	cudaMemcpy(dcc,hcc,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny),cudaMemcpyHostToDevice);
	cudaMemcpy(dV, hV ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny),cudaMemcpyHostToDevice);
	cudaMemcpy(dff,hff,sizeof(T)*(Nx*Ny*Nz+Nx*Ny  ),cudaMemcpyHostToDevice);
	cudaMemcpy(dss,hss,sizeof(T)*(Nx*Ny*Nz+Nx     ),cudaMemcpyHostToDevice);
	cudaMemcpy(dww,hww,sizeof(T)*(Nx*Ny*Nz+1      ),cudaMemcpyHostToDevice);
	cudaMemcpy(dqB,hqB,sizeof(T)*(Nx*Ny           ),cudaMemcpyHostToDevice);

	cudaMemset(dr ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(dp ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(dq ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(ds ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(dpp,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(drh,0,sizeof(T)*(blocks          ));
	cudaMemset(dsg,0,sizeof(T)*(blocks          ));

	cudaMemset(kc , 0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny)); // 6 diagonals for M^-1
	cudaMemset(kf , 0,sizeof(T)*(Nx*Ny*Nz+Nx*Ny  ));
	cudaMemset(kfe, 0,sizeof(T)*(Nx*Ny*Nz+Nx*Ny-1));
	cudaMemset(ks , 0,sizeof(T)*(Nx*Ny*Nz+Nx     ));
	cudaMemset(kse, 0,sizeof(T)*(Nx*Ny*Nz+Nx-1   ));
	cudaMemset(kw , 0,sizeof(T)*(Nx*Ny+1         ));
}



// destroy CUDA fields
void cudaFinalize()
{
	cudaFree(dT);
	cudaFree(dr);
	cudaFree(dp);
	cudaFree(dq);
	cudaFree(ds);
	cudaFree(dV);
	cudaFree(dpp);
	cudaFree(dcc);
	cudaFree(dff);
	cudaFree(dss);
	cudaFree(dww);
	cudaFree(drh);
	cudaFree(dsg);
	cudaFree(dqB);

	cudaFree(kc); // TNS1
	cudaFree(kf);
	cudaFree(kfe);
	cudaFree(ks);
	cudaFree(kse);
	cudaFree(kw);

	cudaDeviceReset();
}

// AXPY (y := alpha*x + beta*y)
template <class T>
__global__ void AXPY(T *y,
		const T *x,
		const T alpha,
		const T beta,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx*Ny;
	y[tid] = alpha * x[tid] + beta * y[tid];
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv1(T *y,
		const T *stC,
		const T *stF,
		const T *stS,
		const T *stW,
		const T *x,
		const int Nx,
		const int Ny)
{
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int tids = tid + Nx*Ny;  // tid shifted

	y[tids] = stC[tid+Nx*Ny] * x[tids]        // center
	    	+ stS[tid+Nx]    * x[tids+Nx]     // north               N
	        + stW[tid+1]     * x[tids+1]      // east              W C E
	        + stS[tid]       * x[tids-Nx]     // south               S
	        + stW[tid]       * x[tids-1]      // west
	        + stF[tid+Nx*Ny] * x[tids+Nx*Ny]  // back                B
	        + stF[tid]       * x[tids-Nx*Ny]; // front               F
}


// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv2(T *y,
		const T *stC,
		const T *stF,
		const T *stFE,
		const T *stS,
		const T *stSE,
		const T *stW,
		const T *x,
		const int Nx,
		const int Ny)
{
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int tids = tid + Nx*Ny;  // tid shifted                               BW - B
                                                 //                           /
	y[tids] = stC[tid+Nx*Ny]   * x[tids]         // center           NW - N  /
		    + stS[tid+Nx]      * x[tids+Nx]      // north                 | /
		    + stW[tid+1]       * x[tids+1]       // east              W - C - E
		    + stS[tid]         * x[tids-Nx]      // south                /|
		    + stW[tid]         * x[tids-1]       // west                / S - SE
		    + stF[tid+Nx*Ny]   * x[tids+Nx*Ny]   // back               /
		    + stF[tid]         * x[tids-Nx*Ny]   // front             F - FE
		    + stSE[tid]        * x[tids-Nx+1]    // south-east
		    + stSE[tid+Nx-1]   * x[tids+Nx-1]    // north-west
		    + stFE[tid]        * x[tids-Nx*Ny+1] // front-east
		    + stFE[tid+Nx*Ny-1]* x[tids+Nx*Ny-1];// back-west

}


// DOT PRODUCT
template <class T, unsigned int blockSize>
__global__ void DOTGPU(T *c,
		const T *a,
		const T *b,
		const int Nx,
		const int Ny,
		const int Nz)
{
	extern __shared__ T cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * (blockSize * 2);
	unsigned int gridSize = (blockSize*2)*gridDim.x;


	cache[tid] = 0;

	while(i<Nx*Ny*Nz) {
		cache[tid] += a[i+Nx*Ny] * b[i+Nx*Ny] + a[i+Nx*Ny+blockSize] * b[i+Nx*Ny+blockSize];
		i += gridSize;
	}

	__syncthreads();

	if(blockSize >= 512) {	if(tid < 256) { cache[tid] += cache[tid + 256]; } __syncthreads(); }
	if(blockSize >= 256) {	if(tid < 128) { cache[tid] += cache[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128) {	if(tid < 64 ) { cache[tid] += cache[tid + 64 ]; } __syncthreads(); }

	if(tid < 32) {
		cache[tid] += cache[tid + 32];
		cache[tid] += cache[tid + 16];
		cache[tid] += cache[tid + 8];
		cache[tid] += cache[tid + 4];
		cache[tid] += cache[tid + 2];
		cache[tid] += cache[tid + 1];
	}

	if (tid == 0) c[blockIdx.x] = cache[0];
}


//
template <class T>
__global__ void elementWiseMul(T *x,
		const T *p,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx*Ny;
	x[tid] *= p[tid];
}


// Truncated Neumann series 1 in 3D
template <class T>
__global__ void makeTNS1(T *smC,
		T *smF,
		T *smFE,
		T *smS,
		T *smSE,
		T *smW,
		const T *stC,
		const T *stF,
		const T *stS,
		const T *stW,
		const int Nx,
		const int Ny,
		const int Nz,
		const int NxNyNz,
		const int NxNy)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tids = tid + NxNy;  // tid shifted

	T tstC1 = 1. / stC[tids];
	T tstC2 = 0.;
	T tstC3 = 0.;
	T tstC5 = 0.;

	if (tid < NxNyNz-NxNy)   tstC5 = 1. / stC[tids+NxNy];    // smF
	if (tid < NxNyNz-Nx)     tstC2 = 1. / stC[tids+Nx];      // smS
	if (tid < NxNyNz-1)      tstC3 = 1. / stC[tids+1];       // smW

    smC[tid+NxNy] = tstC1 * (1. + stW[tid+1]    * stW[tid+1]    * tstC1 * tstC3
    		                    + stS[tid+Nx]   * stS[tid+Nx]   * tstC1 * tstC2
    		                    + stF[tid+NxNy] * stF[tid+NxNy] * tstC1 * tstC5);

    smW[tid+1]    = -stW[tid+1]  * tstC1 * tstC3;

    smS[tid+Nx]   = -stS[tid+Nx] * tstC1 * tstC2;

    smF[tid+NxNy] = -stF[tid+NxNy] * tstC1 * tstC5;

    // complete form
    /*T tstC4 = 0.;
    T tstC6 = 0.;

    if (tid < NxNyNz-NxNy+1) tstC6 = 1. / stC[tids+NxNy-1];  // smFE
    if (tid < NxNyNz-Nx+1)   tstC4 = 1. / stC[tids+Nx-1];    // smSE

    smSE[tid+Nx-1] = (-stS[tid+Nx] * tstC1 * tstC2) * (-stW[tid+Nx] * tstC4);
    smFE[tid+NxNy-1] = (-stF[tid+NxNy] * tstC1 * tstC5) * (-stW[tid+NxNy] * tstC6);*/
}


// for thermal boundary condition
template <class T>
__global__ void addNeumannBC(T *x,
		const T *Q,
		const T HeatFlux,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	x[tid+Nx*Ny] += HeatFlux * Q[tid];
}

#endif /* CUDAFUNCTIONS_H_ */
