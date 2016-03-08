#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define blockWidth 16

//@@ INSERT CODE HERE
__global__ void ImageConvolution(float *image, int imgWidth, int imgHeight, int iChannels,
								 const float * __restrict__ mask, int tWidth, int tHeight, 
								 float *output) {
	// Calculate all indices for later use
	int oIdx_x = blockIdx.x*tWidth + threadIdx.x;
	int oIdx_y = blockIdx.y*tHeight + threadIdx.y;
	int iIdx_x = oIdx_x-2;
	int iIdx_y = oIdx_y-2;
	
	__shared__ float img[blockWidth][blockWidth];
	
	for (int channel=0; channel<iChannels; ++channel) {
		
		// Load image into shared memory	
		if (iIdx_x>=0 && iIdx_x<imgWidth && iIdx_y>=0 && iIdx_y<imgHeight)
			img[threadIdx.x][threadIdx.y] = image[(iIdx_y*imgWidth+iIdx_x)*iChannels+channel];
		else
			img[threadIdx.x][threadIdx.y] = 0.0;
		__syncthreads();
	
		// Calculate convolution
		float tmp = 0;
		if (threadIdx.x<tWidth && threadIdx.y<tHeight) {		
			for (int offset_x=0; offset_x<Mask_width; ++offset_x)
				for (int offset_y=0; offset_y<Mask_width; ++offset_y)
					tmp+=img[threadIdx.x+offset_x][threadIdx.y+offset_y]*mask[offset_y*Mask_width+offset_x];
			// Output
			if (oIdx_x<imgWidth && oIdx_y<imgHeight)
				output[(oIdx_y*imgWidth+oIdx_x)*iChannels+channel] = min(max(tmp, 0.0), 1.0);
		}
		__syncthreads();
	}
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */
	
    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);
	/*
	for (int i=0; i<25; ++i)
		hostMaskData[i] = 0.0;
	hostMaskData[12] = 1.0;
	*/
    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    
	//@@ INSERT CODE HERE
	int tileWidth = blockWidth-Mask_width+1;
	int tileHeight = blockWidth-Mask_width+1;
	dim3 block(blockWidth, blockWidth, 1);
	dim3 grid((imageWidth-1)/tileWidth+1, (imageHeight-1)/tileHeight+1, 1);
	ImageConvolution<<<grid, block>>>(deviceInputImageData, imageWidth, imageHeight, imageChannels,
									  deviceMaskData, tileWidth, tileHeight,
									  deviceOutputImageData);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
	/*
	int y=0;
	for (int c=0; c<imageChannels; ++c)
		for (int x=0; x<imageWidth; ++x)
			wbLog(TRACE, x, ") Input=", hostInputImageData[(y*imageWidth+x)*imageChannels+c],
				  ", Output=", hostOutputImageData[(y*imageWidth+x)*imageChannels+c]);
	*/
    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}