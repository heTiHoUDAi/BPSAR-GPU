
#pragma comment(lib, "ws2_32.lib")

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <string>
#include <WinSock2.h>
#include <Windows.h>
#include "cufft.h"
#include "cufftw.h"
#include <math_constants.h>

#define pi 3.141592654

using namespace std;
typedef struct Es_Data
{
	cufftDoubleComplex *data;
	float *phi;
	float *thi;
	double *RadarPos;
	double *XGeo;
	double *YGeo;
	int nSampling_f, nSampling_phi;
	int GeoNumberX, GeoNumberY;
	double B; 
	double f0; //for ES simulation results, this will be the first frequecny appealled.
	double R0; 
	int fft_times;
}EsDataStr;
EsDataStr Es;

typedef struct BFAInputPars
{
	int nSampling_f;
	int geoPulse;
	double maxRange;
	double R0;
	double f0;
	int times;
	int GeoNumberX, GeoNumberY;
}BFAInputParsStr;

#pragma region Pre Defination
void ReadEsFile(string filepath, int polarInc);
__global__ void BFAProcess(double2 *, double *, double *, double *, cufftDoubleComplex *, BFAInputParsStr, int, int);
int ReadInputFile(void);
void OutputData(char *filepath, double2 *data, int len);
#pragma endregion


#pragma region /* GPU Global Variables */
__device__ cufftDoubleComplex *gpuRangeSignal;
__device__ cufftDoubleComplex *gpuEs;
__device__ double *gpuGeoX;
__device__ double *gpuGeoY;
__device__ double *gpuRadarPos;
__device__ double2 *gpuGeo;
#pragma endregion


int main(int argc, char *argv[])
{
	/* Read the input file */
	printf("BFA start!\r\n");
	ReadInputFile();

	/* Ask memory from GPU */
	cudaMalloc((void **)&gpuRangeSignal, sizeof(cufftDoubleComplex)*Es.nSampling_f * Es.nSampling_phi * Es.fft_times);
	cudaMalloc((void **)&gpuEs, sizeof(cufftDoubleComplex)*Es.nSampling_f * Es.nSampling_phi * Es.fft_times);
	cudaMalloc((void **)&gpuGeoX, sizeof(double)* Es.GeoNumberX);
	cudaMalloc((void **)&gpuGeoY, sizeof(double)* Es.GeoNumberY);
	cudaMalloc((void **)&gpuRadarPos, sizeof(double)*Es.nSampling_phi * 3);
	/* Copy the Es data, X Geo and Y Geo data to the GPU RAM */
	cudaMemcpy(gpuEs, Es.data, sizeof(cufftDoubleComplex)*Es.nSampling_f * Es.nSampling_phi *  Es.fft_times, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuGeoX, Es.XGeo, sizeof(double)* Es.GeoNumberX, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuGeoY, Es.YGeo, sizeof(double)* Es.GeoNumberY, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuRadarPos, Es.RadarPos, sizeof(double)*Es.nSampling_phi * 3, cudaMemcpyHostToDevice); 

	/* calulate the range signal */
	printf("BFA compress signal in every pulse!");
	for (int ii = 0; ii < Es.nSampling_phi; ii++)
	{
		cufftHandle plan;
		cufftResult res = cufftPlan1d(&plan, Es.nSampling_f * Es.fft_times, CUFFT_Z2Z, 1);
		res = cufftExecZ2Z(plan, &gpuEs[ii * Es.nSampling_f * Es.fft_times], &gpuRangeSignal[ii * Es.nSampling_f * Es.fft_times], CUFFT_FORWARD);
		cudaDeviceSynchronize();
		cufftDestroy(plan);
	}
	/* fftshift and conj*/
	cudaMemcpy(Es.data, gpuRangeSignal, sizeof(cufftDoubleComplex)*Es.nSampling_f * Es.nSampling_phi *  Es.fft_times, cudaMemcpyDeviceToHost);
	for (int jj = 0; jj < Es.nSampling_phi; jj++)
	{
		long phiInc = jj * Es.nSampling_f * Es.fft_times;
		for (int ii = 0; ii < Es.nSampling_f * Es.fft_times / 2; ii++)
		{
			cufftDoubleComplex tmp; tmp.x = Es.data[ii + phiInc].x; tmp.y = Es.data[ii + phiInc].y;
			Es.data[ii + phiInc].x = Es.data[ii + phiInc + Es.nSampling_f * Es.fft_times / 2].x;
			Es.data[ii + phiInc].y = Es.data[ii + phiInc + Es.nSampling_f * Es.fft_times / 2].y * (-1);
			Es.data[ii + phiInc + Es.nSampling_f * Es.fft_times / 2].x = tmp.x ;
			Es.data[ii + phiInc + Es.nSampling_f * Es.fft_times / 2].y = tmp.y * (-1);
		}
	}
	/*Copy to GPU RAM*/
	cudaMemcpy(gpuRangeSignal, Es.data, sizeof(cufftDoubleComplex)*Es.nSampling_f * Es.nSampling_phi *  Es.fft_times, cudaMemcpyHostToDevice);
	printf("  done! \r\n");

	/*start BFA*/
	/*Setup GPU RAM for saving the imaging result*/
	printf("BFA start bp algorithm!");
	 cudaMalloc((void **)&gpuGeo, sizeof(double2)*Es.GeoNumberX*Es.GeoNumberY);
	 /*Write a Struct to transmit the setup parameters to the CUDA function*/
	 BFAInputParsStr parStr; // set the input parameter for CUDA.
	 parStr.f0 = Es.f0; parStr.geoPulse = Es.nSampling_phi; parStr.maxRange = 0.3 / Es.B*Es.nSampling_f / 4;
	 parStr.nSampling_f = Es.nSampling_f; parStr.R0 = Es.R0; parStr.times = Es.fft_times;
	 parStr.GeoNumberX = Es.GeoNumberX; parStr.GeoNumberY = Es.GeoNumberY;

	 /* If the pixel needed is too large, we need to cut the batch
	 The max number of X and Y is 500 and 500. 	 */
	 /*Determinate how many batches are needed*/
	int XBatch = (int)((double)Es.GeoNumberX / 500.0);
	int YBatch = (int)((double)Es.GeoNumberY / 500.0);
	/*Less than mini possible range, do not need batch*/
	if (XBatch == 0 && YBatch == 0)
	{
		/*Run BFA Cuda Funtion*/
		dim3 threadPerBlock(Es.GeoNumberY, 1, 1);
		dim3 numBlock(Es.GeoNumberX, 1, 1);
		BFAProcess << <numBlock, threadPerBlock>> >(gpuGeo, gpuRadarPos,
			gpuGeoX, gpuGeoY, gpuRangeSignal, parStr, 0, 0);
		cudaError_t err1 = cudaPeekAtLastError();
		cudaError_t err2 = cudaDeviceSynchronize();
	}
	else
	{
		/*larger than max range*/
		for (int jj = 0; jj < (YBatch); jj++)
		{
			for (int ii = 0; ii < (XBatch); ii++)
			{
				dim3 threadPerBlock(500, 1, 1);
				dim3 numBlock(500, 1, 1);
				BFAProcess << <numBlock, threadPerBlock >> >(gpuGeo, gpuRadarPos,
							gpuGeoX, gpuGeoY, gpuRangeSignal, parStr, 500*ii, 500*jj);
				cudaError_t err1 = cudaPeekAtLastError();
				cudaError_t err2 = cudaDeviceSynchronize();
			}
		}
		/*Process margin batch*/
		/* X range */
		for (int ii = 0; ii < (XBatch); ii++)
		{
			dim3 threadPerBlock(Es.GeoNumberY - 500 * (YBatch), 1, 1);
			dim3 numBlock(500, 1, 1);
			BFAProcess << <numBlock, threadPerBlock >> >(gpuGeo, gpuRadarPos,
				gpuGeoX, gpuGeoY, gpuRangeSignal, parStr, 500 * ii, 500 * (YBatch));
			cudaError_t err1 = cudaPeekAtLastError();
			cudaError_t err2 = cudaDeviceSynchronize();
		}
		/* Y range */
		for (int ii = 0; ii < (YBatch); ii++)
		{
			dim3 threadPerBlock(500, 1, 1);
			dim3 numBlock(Es.GeoNumberX - 500 * (XBatch), 1, 1);
			BFAProcess << <numBlock, threadPerBlock >> >(gpuGeo, gpuRadarPos,
				gpuGeoX, gpuGeoY, gpuRangeSignal, parStr, 500 * (XBatch), 500 * ii);
			cudaError_t err1 = cudaPeekAtLastError();
			cudaError_t err2 = cudaDeviceSynchronize();
		}
		/*In the corner*/
		dim3 threadPerBlock(Es.GeoNumberY - 500 * (YBatch), 1, 1);
		dim3 numBlock(Es.GeoNumberX - 500 * (XBatch), 1, 1);
		BFAProcess << <numBlock, threadPerBlock >> >(gpuGeo, gpuRadarPos,
			gpuGeoX, gpuGeoY, gpuRangeSignal, parStr, 500 * (XBatch), 500 * (YBatch));
		cudaError_t err1 = cudaPeekAtLastError();
		cudaError_t err2 = cudaDeviceSynchronize();
	}
	/*display result*/
	/*Setup CPU RAM to save the imaging result*/
	double2 *resultBFA2 = (double2 *)malloc(sizeof(double2)*Es.GeoNumberX*Es.GeoNumberY);
	cudaError_t err3 = 
	cudaMemcpy(resultBFA2, gpuGeo, sizeof(double2)*Es.GeoNumberX*Es.GeoNumberY, cudaMemcpyDeviceToHost);
	printf("  done! \r\n");
	printf("BFA output data!");
	OutputData("output.txt", resultBFA2, Es.GeoNumberX*Es.GeoNumberY);
	printf("  done! \r\n");
	/*Not all the RAM are free, but, since this function will end, there is no need to free it.
	* Please note that, all GPU RAM are free.
	*/
	cudaFree(gpuRangeSignal);
	cudaFree(gpuEs);
	cudaFree(gpuGeoX);
	cudaFree(gpuGeoY);
	cudaFree(gpuRadarPos);
	cudaFree(gpuGeo);

	return 0;
}

__global__ void BFAProcess(double2 *geo, double *pos,
										double *xgeo, double *ygeo,
										cufftDoubleComplex *range, BFAInputParsStr parStr, int xBatch, int yBatch)
{
	int xi = blockIdx.x;
	int patchi = threadIdx.x;

	double RadarPostionParameterX, RadarPostionParameterY, RadarPostionParameterZ;
	double distance;
	long position;

	geo[(xi + xBatch) + (patchi + yBatch)*parStr.GeoNumberX].x = 0;
	geo[(xi + xBatch) + (patchi + yBatch)*parStr.GeoNumberX].y = 0;

	/*Calculate every pulse*/
	for (int pulseNum = 0; pulseNum < parStr.geoPulse; pulseNum++)
	{
		/*Calculate the distance between the current radar pos and 
		current pixel of SAR image*/
		distance = 0;
		RadarPostionParameterX = pos[pulseNum * 3 + 0];
		RadarPostionParameterY = pos[pulseNum * 3 + 1];
		RadarPostionParameterZ = pos[pulseNum * 3 + 2];
		distance = (RadarPostionParameterX - xgeo[xi + xBatch])*(RadarPostionParameterX - xgeo[xi + xBatch]);
		distance = (RadarPostionParameterY - ygeo[patchi + yBatch]) * (RadarPostionParameterY - ygeo[patchi + yBatch]) + distance;
		distance = RadarPostionParameterZ * RadarPostionParameterZ + distance;
		distance = sqrt(distance);
		/*Deteminate where is the range*/
		position = (long)floor(((distance - parStr.R0) / (parStr.maxRange * 2) + 1.0 / 2.0)*parStr.nSampling_f * parStr.times);
		/*Coherent Accumulation*/
		geo[(xi + xBatch) + (patchi + yBatch)*parStr.GeoNumberX].x = range[position + pulseNum*parStr.nSampling_f * parStr.times].x * cos(4 * pi * parStr.f0 / 0.3*(distance))
			- range[position + pulseNum*parStr.nSampling_f * parStr.times].y*sin(4 * pi * parStr.f0 / 0.3*(distance)) + geo[(xi + xBatch) + (patchi + yBatch)*parStr.GeoNumberX].x;
		geo[(xi + xBatch) + (patchi + yBatch)*parStr.GeoNumberX].y = range[position + pulseNum*parStr.nSampling_f * parStr.times].x * sin(4 * pi * parStr.f0 / 0.3*(distance))
			+ range[position + pulseNum*parStr.nSampling_f *parStr.times].y*cos(4 * pi * parStr.f0 / 0.3*(distance)) + geo[(xi + xBatch) + (patchi + yBatch)*parStr.GeoNumberX].y;
	}
}

void OutputData(char *filepath, double2 *data, int len)
{
	ofstream OutFile;
	OutFile.open(filepath);

	char *outputBuf = (char *)malloc(20 * len * 2);

	long pos = 0; int num = 0;
	num = sprintf(&outputBuf[pos], " %d %d\r\n", Es.GeoNumberX, Es.GeoNumberY);
	pos = pos + num;
	for (int ii = 0; ii < len; ii++)
	{
		num = sprintf(&outputBuf[pos], " %g %g\r\n", data[ii].x, data[ii].y);
		pos = pos + num;
		//OutFile << data[ii].x << ' ' << data[ii].y << endl;
	}
	OutFile.write(outputBuf, pos);
	OutFile.close();
}

void ReadEsFile(string filepath, int polarInc)
{
	ifstream EsFile;
	char output[200];
	char number[15][30];

	Es.data = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*Es.nSampling_f*Es.nSampling_phi*Es.fft_times);
	Es.phi = (float *)malloc(sizeof(float)*Es.nSampling_f*Es.nSampling_phi);
	Es.thi = (float *)malloc(sizeof(float)*Es.nSampling_f*Es.nSampling_phi);

	//zeros all memory
	for (int ii = 0; ii < Es.nSampling_f * Es.nSampling_phi * Es.fft_times; ii++)
	{
		Es.data[ii].x = 0; 	Es.data[ii].y = 0;
	}

	EsFile.open(filepath);
	EsFile.getline(output, 200);
	for (int i = 0; i < Es.nSampling_phi; i++)
	{
		for (int j = 0; j < Es.nSampling_f; j++)
		{
			EsFile >> number[0] >> number[1] >> number[2] >> number[3] >> number[4] >> number[5]
				>> number[6] >> number[7] >> number[8] >> number[9] >> number[10] >> number[11]
				>> number[12] >> number[13] >> number[14];
			Es.data[j + i*Es.nSampling_f * Es.fft_times].x = atof(number[polarInc - 1]);
			Es.data[j + i*Es.nSampling_f * Es.fft_times].y = atof(number[polarInc]);
			Es.thi[j + i*Es.nSampling_f] = atof(number[1]) / 180 * pi;
			Es.phi[j + i*Es.nSampling_f] = atof(number[2]) / 180 * pi;
			// get the center frequency.
			if (i == 0 && j == 0)	Es.f0 = atof(number[0]);
			//get the bandwidth.
			if (i == 0 && j == (Es.nSampling_f - 1)) Es.B = atof(number[0]) - Es.f0;
		}
	}
	EsFile.close();
}

int ReadInputFile(void)
{
	ifstream InputFile; 	
	char output[200]; //characters buffer
	char number[6][30]; //number buffer

	InputFile.open("bfa_input.txt");
	printf("BFA read input file!\r\n");

	if (InputFile.fail())
	{
		printf("Cannot open necessary setup file: bfa_input.txt. \r\n");
		return 0;
	}
	// ignore first 9 lines.
	for (int ii = 0; ii < 9; ii ++)
		InputFile.getline(output, 200);

	//##1 get file path
	string esFilePath;
	InputFile >> esFilePath;

	InputFile.getline(output, 200); InputFile.getline(output, 200); InputFile.getline(output, 200); //ignore 3 lines.
	//##2 read X range
	InputFile >> number[0] >> number[1] >> number[2];
	Es.GeoNumberX = floor((atof(number[1]) - atof(number[0]))/atof(number[2]));
	Es.XGeo = (double *)malloc(sizeof(double)*Es.GeoNumberX);
	for (int ii = 0; ii < Es.GeoNumberX; ii++)
		Es.XGeo[ii] = atof(number[0]) + atof(number[2])*ii;

	InputFile.getline(output, 200); InputFile.getline(output, 200); InputFile.getline(output, 200); //ignore 3 lines.
	//##3 read Y range
	InputFile >> number[0] >> number[1] >> number[2];
	Es.GeoNumberY = floor((atof(number[1]) - atof(number[0])) / atof(number[2]));
	Es.YGeo = (double *)malloc(sizeof(double)*Es.GeoNumberY);
	for (int ii = 0; ii < Es.GeoNumberY; ii++)
		Es.YGeo[ii] = atof(number[0]) + atof(number[2])*ii;

	InputFile.getline(output, 200); InputFile.getline(output, 200); InputFile.getline(output, 200); //ignore 3 lines.
	//##4 read size of Es file.
	InputFile >> number[0] >> number[1];
	Es.nSampling_f = (int)(atof(number[0]));
	Es.nSampling_phi = (int)(atof(number[1]));
	
	InputFile.getline(output, 200); InputFile.getline(output, 200); InputFile.getline(output, 200); //ignore 3 lines.
	//##5 read polar
	InputFile.getline(output, 200);
	int polarInc = 0;
	if (output[0] == 'V' && output[1] == 'V') polarInc = 5;
	if (output[0] == 'H' && output[1] == 'H') polarInc = 8;
	if (output[0] == 'V' && output[1] == 'H') polarInc = 11;
	if (output[0] == 'H' && output[1] == 'V') polarInc = 14;

	InputFile.getline(output, 200); InputFile.getline(output, 200); //ignore 3 lines.
	//##6 Set the distance between the APC and scene center
	InputFile >> number[0];
	Es.R0 = atof(number[0]);

	InputFile.getline(output, 200); InputFile.getline(output, 200); InputFile.getline(output, 200); //ignore 3 lines.
	//##7 Set the points of fouries for range compression. 
	InputFile >> number[0];
	Es.fft_times = (int)(atof(number[0]));

	//Read Es File
	printf("BFA reading es file!");
	ReadEsFile(esFilePath, polarInc);
	printf("  done!\r\n");

	//Calu the Radar Pos
	printf("BFA calu radar position for every pulse!");
	Es.RadarPos = (double *)malloc(sizeof(double)*Es.nSampling_phi * 3);
	for (int ii = 0; ii < Es.nSampling_phi; ii++)
	{	
		Es.RadarPos[ii * 3 + 0] = Es.R0 * sin(Es.phi[Es.nSampling_f * ii]) * sin(Es.thi[Es.nSampling_f * ii]);
		Es.RadarPos[ii * 3 + 1] = Es.R0 * cos(Es.phi[Es.nSampling_f * ii]) * sin(Es.thi[Es.nSampling_f * ii]);
		Es.RadarPos[ii * 3 + 2] = Es.R0 * cos(Es.thi[Es.nSampling_f * ii]);
	}
	printf("  done!\r\n");

	return 1;
}