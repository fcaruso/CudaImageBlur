#include <iostream>

// CUDA utilities and system includes
#include <cuda_runtime.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_image.h>

#include <cuda_profiler_api.h>

char *image_filename = "./data/lena.pgm";
unsigned int width, height;
unsigned char *h_img  = NULL;
unsigned char *d_img  = NULL;

#define BLOCK_WIDTH		64
#define BLOCK_HEIGHT	16

#ifndef __CUDA_ARCH__
#ifdef WIN32
#pragma message ("__CUDA_ARCH__ undefined")
#else
#warning "__CUDA_ARCH__ undefined"
#endif
#endif 


//////////////////////////////////////
/// Radial blur using global memory
//////////////////////////////////////
template<unsigned short RADIUS >
__global__ void kRadialBlur( unsigned char* img, unsigned width, unsigned height, size_t pitch)
{
	__shared__ unsigned char sh[BLOCK_HEIGHT + 2*RADIUS][BLOCK_WIDTH + 2*RADIUS];

	int g_x = blockDim.x*blockIdx.x + threadIdx.x;
	int g_y = blockDim.y*blockIdx.y + threadIdx.y;

	int pid_x = threadIdx.x + RADIUS;
	int pid_y = threadIdx.y + RADIUS;
	
	///////////////////////
	// gather into shared memory
	///////////////////////
	sh[pid_y][pid_x] = img[ g_y*pitch + g_x];

	// halo 
	if ( ( threadIdx.x < RADIUS ) && ( g_x  >= RADIUS ) )
	{
		sh[pid_y][pid_x - RADIUS] = img[ g_y*pitch + g_x - RADIUS];

		if ( ( threadIdx.y < RADIUS ) && ( g_y >= RADIUS ) )
		{
			sh[pid_y - RADIUS][pid_x - RADIUS] = img[ (g_y - RADIUS)*pitch + g_x - RADIUS];
		}
		if ( ( threadIdx.y > (BLOCK_HEIGHT -1 - RADIUS) ) )
		{
			sh[pid_y + RADIUS][pid_x - RADIUS] = img[ (g_y + RADIUS)*pitch + g_x - RADIUS];
		}
	}
	if ( ( threadIdx.x > ( BLOCK_WIDTH -1 - RADIUS ) ) && ( g_x < ( width - RADIUS ) ) )
	{
		sh[pid_y][pid_x + RADIUS ] = img[ g_y*pitch + g_x + RADIUS];

		if ( ( threadIdx.y < RADIUS ) && ( g_y > RADIUS ) )
		{
			sh[pid_y - RADIUS][pid_x + RADIUS] = img[ (g_y - RADIUS)*pitch + g_x + RADIUS];
		}
		if ( (threadIdx.y > (BLOCK_HEIGHT -1 - RADIUS ) ) && ( g_y < ( height - RADIUS ) ) )
		{
			sh[pid_y + RADIUS][pid_x + RADIUS] = img[ (g_y + RADIUS)*pitch + g_x + RADIUS];
		}
	}

	if ( ( threadIdx.y < RADIUS ) && ( g_y >= RADIUS ) )
	{
		sh[pid_y - RADIUS][pid_x] = img[ (g_y - RADIUS)*pitch + g_x];
	}
	if ( ( threadIdx.y > ( BLOCK_HEIGHT -1 - RADIUS ) ) && ( g_y < ( height - RADIUS ) ) )
	{
		sh[pid_y + RADIUS][pid_x] = img[ ( g_y + RADIUS)*pitch + g_x ];
	}

	__syncthreads();

	//////////////////////
	// compute the blurred value
	//////////////////////

	unsigned val = 0;
	unsigned k = 0;
	for (int i=-RADIUS; i<= RADIUS; i++ )
		for ( int j=-RADIUS; j<=RADIUS ; j++ )
		{
			if ( ( ( g_x + j ) < 0 ) || ( ( g_x + j ) > ( width - 1) ) )
				continue;
			if ( ( ( g_y + i ) < 0 ) || ( ( g_y + i ) > ( height - 1) ) )
				continue;
			val += sh[pid_y + i][pid_x + j];
			k++;
		}

	val /= k;

	////////////////////
	// write into global memory
	///////////////

	img[ g_y*pitch + g_x ] = (unsigned char) val;
			
}

//////////////////////////////////////
/// Radial blur using texture memory
//////////////////////////////////////

template<unsigned short RADIUS>
__global__ void kRadialBlur( unsigned char* img, cudaTextureObject_t tex,
							unsigned width, unsigned height, size_t pitch)
{
	__shared__unsigned char sh[BLOCK_HEIGHT + 2*RADIUS][BLOCK_WIDTH + 2*RADIUS];

	int g_x = blockDim.x*blockIdx.x + threadIdx.x;
	int g_y = blockDim.y*blockIdx.y + threadIdx.y;

	int pid_x = threadIdx.x + RADIUS;
	int pid_y = threadIdx.y + RADIUS;
	
	///////////////////////
	// gather into shared memory
	///////////////////////
	sh[pid_y][pid_x] = tex2D<unsigned char>(tex, g_x, g_y);
	
	// halo 
	if ( ( threadIdx.x < RADIUS ) && ( g_x  >= RADIUS ) )
	{
		sh[pid_y][pid_x - RADIUS] = tex2D<unsigned char>(tex, g_x - RADIUS , g_y);

		if ( ( threadIdx.y < RADIUS ) && ( g_y >= RADIUS ) )
		{
			sh[pid_y - RADIUS][pid_x - RADIUS] = tex2D<unsigned char>(tex, g_x , g_y - RADIUS);
		}
		if ( ( threadIdx.y > (BLOCK_HEIGHT -1 - RADIUS) ) )
		{
			sh[pid_y + RADIUS][pid_x - RADIUS] = tex2D<unsigned char>(tex, g_x - RADIUS, g_y - RADIUS);
		}
	}
	if ( ( threadIdx.x > ( BLOCK_WIDTH -1 - RADIUS ) ) && ( g_x < ( width - RADIUS ) ) )
	{
		sh[pid_y][pid_x + RADIUS ] = tex2D<T>(tex, g_x + RADIUS, g_y );

		if ( ( threadIdx.y < RADIUS ) && ( g_y > RADIUS ) )
		{
			sh[pid_y - RADIUS][pid_x + RADIUS] = tex2D<unsigned char>(tex, g_x + RADIUS, g_y - RADIUS);
		}
		if ( (threadIdx.y > (BLOCK_HEIGHT -1 - RADIUS ) ) && ( g_y < ( height - RADIUS ) ) )
		{
			sh[pid_y + RADIUS][pid_x + RADIUS] = tex2D<unsigned char>(tex, g_x + RADIUS, g_y + RADIUS);
		}
	}

	if ( ( threadIdx.y < RADIUS ) && ( g_y >= RADIUS ) )
	{
		sh[pid_y - RADIUS][pid_x] = tex2D<unsigned char>(tex, g_x , g_y - RADIUS);
	}
	if ( ( threadIdx.y > ( BLOCK_HEIGHT -1 - RADIUS ) ) && ( g_y < ( height - RADIUS ) ) )
	{
		sh[pid_y + RADIUS][pid_x] = tex2D<unsigned char>(tex, g_x , g_y + RADIUS);
	}

	__syncthreads();

	//////////////////////
	// compute the blurred value
	//////////////////////

	unsigned val = 0;
	unsigned k = 0;
	for (int i=-RADIUS; i<= RADIUS; i++ )
		for ( int j=-RADIUS; j<=RADIUS ; j++ )
		{
			if ( ( ( g_x + j ) < 0 ) || ( ( g_x + j ) > ( width - 1) ) )
				continue;
			if ( ( ( g_y + i ) < 0 ) || ( ( g_y + i ) > ( height - 1) ) )
				continue;
			val += sh[pid_y + i][pid_x + j];
			k++;
		}

	val /= k;

	////////////////////
	// write into global memory
	///////////////

	img[ g_y*pitch + g_x ] = (unsigned char) val;
			
}

int main(int argc, char* argv[])
{
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	cudaProfilerStart();
	cudaError_t err;
    // load image (needed so we can get the width and height before we create the window
	sdkLoadPGM(image_filename, (unsigned char **) &h_img, &width, &height);
	printf("width: %d \t height: %d \n", width, height);

	// fill GPU  memory
	unsigned char* d_img = NULL;
	size_t pitch;
	cudaMallocPitch( (void**) &d_img, &pitch, width*sizeof(unsigned char), height );
	cudaMemcpy2D( d_img, pitch*sizeof(unsigned char), 
			h_img, width*sizeof(unsigned char), width*sizeof(unsigned char), height, 
			cudaMemcpyHostToDevice );

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);
	cudaMemcpyToArray(cuArray, 0, 0, h_img, 
		width*height*sizeof(unsigned char),
		cudaMemcpyHostToDevice );

	cudaResourceDesc resDesc;
	memset( &resDesc, 0, sizeof(resDesc) );
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	memset( &texDesc, 0, sizeof( texDesc ) );
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject( &texObj, &resDesc, &texDesc, NULL );

	// create vars for timing
	cudaEvent_t startEvent, stopEvent;
	err = cudaEventCreate(&startEvent, 0);
	assert( err == cudaSuccess );
	err = cudaEventCreate(&stopEvent, 0);
	assert( err == cudaSuccess );
	float elapsedTime;

	// process image
	dim3 dGrid(width / BLOCK_WIDTH, height / BLOCK_HEIGHT);
	dim3 dBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	
	// execution of the version using global memory
	cudaEventRecord(startEvent);
	kRadialBlur<4> <<< dGrid, dBlock >>> (d_img, width, height, pitch );
	cudaThreadSynchronize();
	cudaEventRecord(stopEvent);
	cudaEventSynchronize( stopEvent );
	cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent);

	printf("elapsed time of version using global memory: %f\n", elapsedTime );

	// execution of the version using texture memory
	if ( deviceProp.major >= 3 ) // Texture objects are supported from arch 3.X
	{
		cudaEventRecord(startEvent);
		kRadialBlur<4> <<< dGrid, dBlock >>> (d_img, texObj, width, height, pitch );
		cudaThreadSynchronize();
		cudaEventRecord(stopEvent);
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent);

		printf("elapsed time of version using texture memory: %f\n", elapsedTime );

	}
	else
	{
        printf("CUDA Texture Object requires a GPU with compute capability "
               "3.0 or later\n");
	}
	// save image
	cudaMemcpy2D( h_img, width*sizeof(unsigned char), 
		d_img, pitch*sizeof(unsigned char), width*sizeof(unsigned char), height,
		cudaMemcpyDeviceToHost );
	sdkSavePGM("./data/blurred_tex.ppm", h_img, width, height );

	// free memory
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(cuArray);
	cudaFree(d_img);
	cudaProfilerStop();
	cudaDeviceReset();
	free(h_img);

	return 0;
}