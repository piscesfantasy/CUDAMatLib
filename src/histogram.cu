// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_WIDTH 16

//@@ insert code here

__global__ void InitHistogram(int *hist){
	hist[threadIdx.x] = 0;
}

__global__ void RGB2GrayHistogram (float *image, unsigned char *ucharImage, int *hist, int width, int height, int channel) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x<width && y<height) {
		int index = (y*width+x)*channel;
		ucharImage[index] = (unsigned char)(255 * image[index]); // r
        ucharImage[index+1] = (unsigned char)(255 * image[index+1]); // g
        ucharImage[index+2] = (unsigned char)(255 * image[index+2]); // b
        unsigned char grayScale = (unsigned char)(0.21*ucharImage[index] + 0.71*ucharImage[index+1] + 0.07*ucharImage[index+2]); // (unsigned char)
		atomicAdd( &(hist[(int)grayScale]), 1);
	}
}

__global__ void	Distribution2CDF (int *hist, float *cdf, int width, int height) {
	__shared__ float tmp[HISTOGRAM_LENGTH];
	
	// Read memory
	tmp[threadIdx.x] = (float)hist[threadIdx.x];
	
	// Reduction phase
	for (int stride=1; stride<=HISTOGRAM_LENGTH/2; stride*=2) {
		__syncthreads();
		int reductionIdx = (threadIdx.x+1)*2*stride-1;
		if (reductionIdx < HISTOGRAM_LENGTH)
			tmp[reductionIdx]+=tmp[reductionIdx-stride];
	}
	
	// Post reduction reverse phase
	for (int stride=HISTOGRAM_LENGTH/4; stride>0; stride/=2) {
		__syncthreads();
		int reductionIdx = (threadIdx.x+1)*2*stride-1;
		if (reductionIdx+stride < HISTOGRAM_LENGTH)
			tmp[reductionIdx+stride]+=tmp[reductionIdx];
	}
	
	// Write memory
	__syncthreads();
	cdf[threadIdx.x] = tmp[threadIdx.x]/(width*height);
}

__global__ void ApplyHEQ (float *image, unsigned char *ucharImage, float *cdf, int width, int height, int channel) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x<width && y<height) {
		int index = (y*width+x)*channel;
		image[index] = (float)(min(max(255.0*(cdf[ucharImage[index]]-cdf[0])/(1-cdf[0]), 0.0), (float)HISTOGRAM_LENGTH))/255.0;
		image[index+1] = (float)(min(max(255.0*(cdf[ucharImage[index+1]]-cdf[0])/(1-cdf[0]), 0.0), (float)HISTOGRAM_LENGTH))/255.0;
		image[index+2] = (float)(min(max(255.0*(cdf[ucharImage[index+2]]-cdf[0])/(1-cdf[0]), 0.0), (float)HISTOGRAM_LENGTH))/255.0;
	}
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    const char * inputImageFile;
	
	float * hostInputImageData;
    float * hostOutputImageData;
	
	float * deviceInputImageData;
	unsigned char * deviceUCharImageData;
	int * deviceHistogram;
	float * deviceCDF;
	float * deviceOutputImageData;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); // parse the input arguments

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here	
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceUCharImageData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(int));
	cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");
	
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
	
	wbTime_start(Compute, "Doing the computation on the GPU");

	dim3 block2D(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 grid2D((imageWidth-1)/BLOCK_WIDTH+1, (imageHeight-1)/BLOCK_WIDTH+1, 1);
	dim3 block1D(HISTOGRAM_LENGTH, 1, 1);
	dim3 grid1D(1, 1, 1);

	InitHistogram<<<grid1D, block1D>>>(deviceHistogram);
	RGB2GrayHistogram<<<grid2D, block2D>>>(deviceInputImageData, deviceUCharImageData, deviceHistogram, imageWidth, imageHeight, imageChannels);
	Distribution2CDF<<<grid1D, block1D>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);
	ApplyHEQ<<<grid2D, block2D>>> (deviceOutputImageData, deviceUCharImageData, deviceCDF, imageWidth, imageHeight, imageChannels);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from GPU");
    cudaMemcpy(hostOutputImageData,
			   deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from GPU");
	
    wbSolution(args, outputImage);

    //@@ insert code here
	cudaFree(deviceInputImageData);
    cudaFree(deviceUCharImageData);
	cudaFree(deviceHistogram);
	cudaFree(deviceCDF);
	cudaFree(deviceOutputImageData);
	free(hostInputImageData);
	free(hostOutputImageData);

    return 0;
}