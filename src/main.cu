#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <stdlib.h>
#include <math.h>

#define min(a,b) a<b ? a : b
#define max(a,b) a>b ? a : b
#define DEFL

using namespace std;

typedef float T;

typedef struct {
	unsigned int blocks;
	unsigned int blockSize;
	unsigned int blocksZ;
	unsigned int Nx;
	unsigned int Ny;
	unsigned int Nz;
	unsigned int size;
	unsigned int steps;
	unsigned int maxiter;
	T dt;
	T maxres;
	unsigned int nRowsZ;
	unsigned int nDV;
    unsigned int NxZ;
} Parameters;


typedef struct {
	T rho;
	T cp;
	T k;
} MaterialProperties;

Parameters params;

MaterialProperties steel;
MaterialProperties Ag;
MaterialProperties MgO;
MaterialProperties inconel;
MaterialProperties NiCr;

#include "cpuFunctions.h"
//#include "cpuFunctionsDeflation.h"
#include "cudaFunctions.h"
//#include "cudaFunctionsDeflation.h"

int main(void)
{
    // Parameters
	params.blocks = 256;  
	params.blockSize = 128;  
	params.Nx  = 32;
	params.Ny  = 32;
	params.Nz  = 128;
	params.size = params.Nx*params.Ny*params.Nz + 2*params.Nx*params.Ny;
	params.steps  = 300;
	params.maxiter  = 1000000;
	params.dt = 1./300.;    // 0.0033333333333333;
	params.maxres  = 1e-4;
	params.nRowsZ = 4;    // number of rows/columns for one deflation vector
	params.nDV = (params.Nx*params.Ny) / (params.nRowsZ*params.nRowsZ);
	params.NxZ = params.Nx/params.nRowsZ; // number of course cells in a row
		
	// steel
	steel.rho = 7610.0; // 7700
    steel.cp  = 545.0; // 560
    steel.k   = 21.0;

    // Ag
    Ag.rho = 8957.0;
    Ag.cp  = 362.0; // 368
    Ag.k   = 120; // 111.5

    // MgO
    MgO.rho = 3150.0;
    MgO.cp  = 1110.0; // 1140
    MgO.k   = 11.5;   // 10
    
    // inconel
    inconel.rho = 8470.0;
    inconel.cp  = 500.0; // 520
    inconel.k   = 20.5; 
    
    // NiCr
    NiCr.rho = 8200.0;
    NiCr.cp  = 528.0;
    NiCr.k   = 24.5;
    
    T t = 0; // time
    T totalIter = 0;
    
    
    dim3 dimGrid(params.blocks);
    dim3 dimBlock(params.blockSize);
    
    cout << "------------------" << endl;
	cout << "--- 3D CG TNS1 ---" << endl;
	cout << "------------------" << endl;
	
	
	cpuInit(params.blocks, params.Nx, params.Ny, params.Nz);
	//cpuInitDeflation(params.Nx, params.Ny, params.NxZ, params.nDV);
	
	readGeometry(hm, params.Nx, params.Ny);	// materials
	
	readCoords(xc, dx, "xCoords.txt");
	readCoords(yc, dy, "yCoords.txt");
	readCoords(zc, dz, "zCoords.txt");
	
	readBC(tHF, params.steps);
	
	initX(hT, params.Nx, params.Ny, params.Nz);
    initA(params.dt, params.Nx, params.Ny, params.Nz);
	
	//initAZ(params.Nx, params.Ny, params.nRowsZ);
	//initE(params.Nx, params.Ny, params.nRowsZ, params.NxZ);
	//spChol(params.NxZ, params.nDV);
	
	//check_varZ(ecc, ess, params.nDV, params.NxZ);
	
    //saveData<int>(hm, "materials3D32x32x128", params.Nx, params.Ny, params.Nz);
	//saveDataInTime(hT, t, "temperature32x32x128", params.Nx, params.Ny, params.Nz);
		
	//check_var(dy, params.Nx, params.Ny);

	// CUDA
	cudaInit(hT, hV, hcc, hff, hss, hww, hqB, params.blocks, params.Nx, params.Ny, params.Nz);
	//cudaInitDeflation(azc, azw, azs, ecc, eww, ess, params.NxZ, params.nDV, params.Nx, params.Ny);
	
	makeTNS1<<<128,1024>>>(kc, kf, kfe, ks, kse, kw, dcc, dff, dss, dww, 
			               params.Nx, params.Ny, params.Nz, params.Nx*params.Ny*params.Nz, params.Nx*params.Ny);

	
	cudaEvent_t startT, stopT;
	float elapsedTime;
	cudaEventCreate(&startT);
	cudaEventCreate(&stopT);
	cudaEventRecord(startT,0);

	for (int miter=0; miter<params.steps; miter++) {

		cudaMemcpy(dr, dT, sizeof(T)*params.size, cudaMemcpyDeviceToDevice);	 // r = rhs
		//               128x32x32 ~ grid dimensions
		elementWiseMul<<<128,1024>>>(dr, dV, params.Nx, params.Ny);	             // r = V*r
		//               32x32 ~ grid dimensions 
		addNeumannBC<<<1,1024>>>(dr, dqB, tHF[miter], params.Nx, params.Ny);     // time dependent Neumann boundary here ... r = r + NeumannBC (dqB)

		SpMVv1<<<128,1024>>>(dq, dcc, dff, dss, dww, dT, params.Nx, params.Ny);  // q = Ax (version 1)
		AXPY<<<128,1024>>>(dr, dq, (T)-1., (T)1., params.Nx, params.Ny);         // r = r - q
		
/*#ifdef DEFL
		
		// ---  (DEFLATION) r = Pr  ---
		///cudaMemset(dyZ,0,sizeof(T)*(params.nDV+2*params.NxZ)); // reset;
		ZTransXYDeflation<<<4,64>>>(drZ, dr, params.nRowsZ, params.NxZ, params.Nx);  // y1 = Z'*y
		// E*y2 = y1 (begin)
		cudaMemcpy(hrZ, drZ, (params.nDV+2*params.NxZ)*sizeof(double), cudaMemcpyDeviceToHost);   // copy drZ to hrZ
		solve(params.NxZ,params.nDV);
		cudaMemcpy(dyZ, hyZ, (params.nDV+2*params.NxZ)*sizeof(double), cudaMemcpyHostToDevice);   //copy hyZ to dyZ
		///printX(hyZ, params.NxZ,params.nDV);
		// E*y2 = y1 (end)
		YMinusAzXYDeflation<<<32,128>>>(dr,dyZ,dazc,dazw,dazs,params.nRowsZ,params.NxZ,params.Nx);  // r = P*r
#endif	*/    

		SpMVv2<<<128,1024>>>(ds, kc, kf, kfe, ks, kse, kw, dr, params.Nx, params.Ny);   // z = M^(-1)r (version 2)
		DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, ds, params.Nx, params.Ny, params.Nz);
		cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
		rhNew = dot(hrh,params.blocks);
		
		//stop = rhNew * params.maxres * params.maxres;
		
		// --- stop criterion here ---
		DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, ds, ds, params.Nx, params.Ny, params.Nz);
		cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
		stop = dot(hrh,params.blocks) * params.maxres * params.maxres;
		
		iter = 0;
		
		//cout << "stop:" << stop << ", residual: " << rhNew << endl;
		
		while (abs(rhNew) > stop && iter < params.maxiter) {
		//while (iter < 10) {  // only testing

			iter++;
			totalIter++;
			//cout << "iteration:" << iter << ", residual: " << abs(rhNew) << endl;
			
			if (iter==1) {
				cudaMemcpy(dp, ds, sizeof(T)*params.size,cudaMemcpyDeviceToDevice);
			}
			else {
				bt = rhNew/rhOld;	
				AXPY<<<128,1024>>>(dp, ds, (T)1., bt, params.Nx, params.Ny);   // p = z + beta*p	
			}

			SpMVv1<<<128,1024>>>(dq, dcc, dff, dss, dww, dp, params.Nx, params.Ny);  // q = Ap (version 1)
			
/*#ifdef DEFL
			
			// ---  (DEFLATION) q = Pq  ---
			ZTransXYDeflation<<<4,64>>>(drZ, dq, params.nRowsZ, params.NxZ, params.Nx);  // y1 = Z'*y
			// E*y2 = y1 (begin)
			cudaMemcpy(hrZ, drZ, (params.nDV+2*params.NxZ)*sizeof(T), cudaMemcpyDeviceToHost);   // copy drZ to hrZ
			solve(params.NxZ,params.nDV);
			cudaMemcpy(dyZ, hyZ, (params.nDV+2*params.NxZ)*sizeof(T), cudaMemcpyHostToDevice);   //copy hyZ to dyZ
			// E*y2 = y1 (end)
			YMinusAzXYDeflation<<<32,128>>>(dq,dyZ,dazc,dazw,dazs,params.nRowsZ,params.NxZ,params.Nx);  // q = Pq
#endif*/



			DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(dsg, dp, dq, params.Nx, params.Ny, params.Nz);   // sigma = <p,q>
			cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			sg = dot(hsg,params.blocks);
			ap = rhNew/sg;	// alpha = rhoNew / sigma
			AXPY<<<128,1024>>>(dr, dq, -ap, (T)1., params.Nx, params.Ny);   // r = r - alpha*q
			AXPY<<<128,1024>>>(dT, dp,  ap, (T)1., params.Nx, params.Ny);   // x = x + alpha*p

			SpMVv2<<<128,1024>>>(ds, kc, kf, kfe, ks, kse, kw, dr, params.Nx, params.Ny);   // z = M^(-1)r (version 2)

			rhOld = rhNew;

			DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, ds, params.Nx, params.Ny, params.Nz);   // rhoNew = <r,z>		
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
		    rhNew = dot(hrh,params.blocks);
		}
		
		// x = x +  (deflation)
		
		t += params.dt;
		//cout << "t= " << setprecision(3) << fixed << t << " s ,tmstp:" << miter << " ,it:" << iter << endl;
		
		//if ((miter+1)%4000==0)
		//{
		//cudaMemcpy(hT, dT, sizeof(T)*params.size, cudaMemcpyDeviceToHost);
		//saveDataInTime(hT, t, "temperature32x32x128", params.Nx, params.Ny, params.Nz);
		//}

	}
	
	cudaEventRecord(stopT,0);
	cudaEventSynchronize(stopT);
	cudaEventElapsedTime(&elapsedTime, startT, stopT);
	cout<< "ellapsed time (cuda): " << elapsedTime << " miliseconds" << endl;
	
	cout << "Simulation finished." << endl;
	cout << "total number of iterations: " << totalIter << endl;
	
	cudaEventDestroy(startT);
	cudaEventDestroy(stopT);
	
	//cudaFinalizeDeflation();
	cudaFinalize();
	
	//cpuFinalizeDeflation();
	cpuFinalize();
	
	return 0;
}

