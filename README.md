CudaImageBlur
=============

CUDA kernel function for image blur

To compile the source you have just to: 

1.  Create a brand new CUDA project 
2.  Add the file imageBlur.cu 
3.  Add the CUDA Sample/common/inc directory to the include directory search path 

Optionally, you can uncomment the #define USE_TEXTURE_OBJECT to compile also the kernel using Texture Objects.
It requires a graphics card with CUDA Compute Capability >= 3.0
