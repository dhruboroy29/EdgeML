#include <stdio.h>
#include <stdlib.h>

#define min(a,b) (b>a)?a:b
#define max(a,b) (a>b)?a:b

typedef long long ll;

const long long qW1_transp_l[][8] = {{-1,2,-6,-4,5,1,4,6},{0,-5,-6,1,4,7,0,-6},{-2,-6,6,-1,-6,3,2,4},{-5,-1,6,7,6,1,-3,0},{-7,-6,2,-1,2,-3,-1,1}};
const long long qFC_Bias_l[2] = {21,35};
const long long qW2_transp_l[][5] = {{-1,-2,-1,-4,3},{-2,-10,0,2,1},{9,-3,7,-15,-5},{-5,2,-3,-8,9},{-1,0,-7,-6,-6},{-4,-2,-2,-1,-3},{2,2,2,1,3},{10,-4,-5,-1,4},{-6,2,8,-2,-7},{-4,9,-4,10,5},{-6,-17,-3,-4,-3},{-2,2,-4,-1,-4},{-4,0,-3,-7,10},{4,-1,2,2,-3},{2,1,0,0,2},{4,13,2,1,0},{2,0,1,3,-2},{9,-2,8,9,-15},{-12,-2,-7,4,0},{5,-1,7,-7,-2},{6,-2,0,6,-8},{-5,1,-6,8,1},{-4,-9,-4,-7,-5},{-11,5,4,7,-5},{2,1,2,-1,0},{0,-1,1,1,0},{-1,0,0,-2,5},{10,-2,3,13,-16},{7,2,12,6,5},{-6,1,-3,-8,10},{1,0,-1,2,-4},{0,0,1,3,-2}};
const long long qU2_transp_l[][6] = {{-5,21,-12,16,15,3},{5,1,-1,1,-1,-1},{-4,3,-1,-3,9,-4},{4,-7,-6,-10,3,-5},{5,-2,-4,1,-5,3},{-9,-3,3,-1,11,-9},{-4,6,5,-2,-17,7},{-3,-6,-2,2,-2,-2},{-4,3,-1,-1,3,-2},{-5,-4,3,0,4,-3},{-2,2,6,-3,4,-1},{-4,-8,-3,1,8,-1},{4,2,-22,-13,13,5},{-16,5,3,-8,-3,14},{5,-5,-7,3,-11,6},{2,-5,2,0,6,-3},{6,-8,4,3,-7,1},{1,2,4,-3,2,-8},{17,6,-9,-16,0,-3},{1,-5,6,2,1,1},{-6,-11,-15,-18,12,9},{8,3,0,0,3,0},{2,0,1,1,-17,7},{5,4,-8,-3,-2,1},{10,12,-8,-3,1,2},{1,-1,-1,-6,7,-9},{8,-10,15,3,-7,8},{-13,-5,-13,0,-1,15},{-8,-3,-5,-2,8,-7},{2,-5,6,0,-2,-7},{-3,7,-12,-6,11,-8},{0,-15,2,0,-3,1}};
const long long qFC_Weight_l[][2] = {{40,-31},{-22,79},{35,-47},{83,-21},{-52,54},{-1,-23},{33,-77},{-82,23},{-13,-5},{26,-86},{65,-32},{92,-128},{-57,45},{-73,53},{-42,19},{-67,100},{34,-62},{33,-43},{3,-48},{-63,34},{41,-17},{-64,53},{-75,17},{-44,59},{47,-84},{23,-25},{-56,54},{44,7},{-74,44},{-4,38},{-29,11},{-46,-14}};
const long long qU1_transp_l[][32] = {{-5,2,3,-4,9,-19,-2,12,11,1,4,4,2,-2,-4,-1,11,1,0,-4,0,-3,-2,3,-1,4,5,-1,6,3,8,12},{11,-5,1,-5,-6,5,5,-7,-7,-3,2,1,0,-1,-9,-2,1,1,-2,-2,-2,-4,-1,-4,17,22,-5,-2,-4,-3,4,13},{1,-12,-2,1,1,14,3,4,1,-1,1,-1,-2,1,4,-10,13,-3,3,-14,7,-9,-4,2,-1,-7,5,-1,2,3,-10,-1},{-7,2,1,-3,4,-6,1,3,6,3,1,2,10,1,8,1,-1,1,-1,2,-4,2,0,1,-7,-6,5,2,3,11,-6,0},{1,-2,-1,1,14,5,-8,14,3,0,-1,18,2,-1,-18,-1,-13,2,1,-7,-3,-7,-3,12,-19,14,0,2,6,2,10,-6},{1,-6,-2,4,4,-7,2,5,-15,-4,-1,-6,0,0,7,1,-8,2,2,-2,-4,-5,5,13,4,0,8,10,4,-9,-7,3}};
const long long qB_g_l[32] = {13,17,16,8,18,16,16,21,20,16,17,19,10,14,14,21,16,18,16,19,14,18,20,20,15,16,17,12,20,16,13,-7};
const long long qB_h_l[32] = {47,85,21,16,118,39,24,60,79,19,23,37,20,24,49,108,31,22,14,76,10,73,79,126,34,31,29,16,119,63,37,1};
const long long q_l= 15;

const long long I_l= 100000;

const long long mean_l[8] = {2027,2032,2027,2032,2027,2032,2027,2032};
const long long stdev_l[8] = {30,31,30,31,30,31,30,31};

const int wRank = 5;
const int uRank = 6;
const int inputDims = 8;
const int hiddenDims = 32;
const int timeSteps = 12;
const int numInstances = 8;
const int numClasses = 2;

ll I_l_vec[hiddenDims] = {I_l};

ll out_wRank[wRank] = {};
ll out_uRank[uRank] = {};
ll out_inputDims[inputDims] = {};
ll out_hiddenDims[hiddenDims] = {};

//Vector-scalar multiplication
void mulVecScal(ll* vec, ll scal, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec+i)*scal;
} 

//Vector-scalar division
void divVecScal(ll* vec, ll scal, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec+i)/scal;
} 

//Matrix-vector multiplication
void mulMatVec(ll* mat, ll* vec, int mat_rows, int vec_len, ll* out){
	for(int i=0; i < mat_rows; i++){
		out[i] = 0;
		for(int j=0; j < vec_len; j++)
			out[i] += *((mat+i*vec_len)+j)**(vec+j);
	}
}

//Vector-vector addition
void addVecs(ll* vec1, ll* vec2, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec1+i)+*(vec2+i);
}

//Vector-vector subtraction
void subVecs(ll* vec1, ll* vec2, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec1+i)-*(vec2+i);
}

//Vector-vector multiplication (Hadamard)
void mulVecs(ll* vec1, ll* vec2, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec1+i)**(vec2+i);
}

// In-place standardization with scaling
void stdScaleInput(ll* inp_vec, int vec_len){
	for(int i=0; i<vec_len; i++)
		*(inp_vec+i) = I_l*(*(inp_vec+i)-mean_l[i])/stdev_l[i];
}

// In-place quantTanh
void quantTanh(ll* vec, int vec_len, ll scale){
	for(int i=0; i<vec_len; i++)
		*(vec+i) = max(-scale, min(scale, *(vec+i)));
}

// In-place quantSigm
void quantSigm(ll* vec, int vec_len, ll scale){
	for(int i=0; i<vec_len; i++)
		*(vec+i) = max(min((*(vec+i)+scale)/2, scale),0);
}


int main(){
	int size = sizeof(qW1_transp_l) + sizeof(qFC_Bias_l) + sizeof(qW2_transp_l) + sizeof(qU2_transp_l) + sizeof(qFC_Weight_l) + sizeof(qU1_transp_l) + sizeof(qB_g_l) + sizeof(qB_h_l) + sizeof(q_l) + sizeof(I_l) + sizeof(mean_l) + sizeof(stdev_l);
	printf("Model size: %d KB\n", size/1000);

	long long x[] = {1, 4, 5, 1, 0, 4, 5, 1};
	
	mulMatVec((ll*)qW1_transp_l, x, wRank, inputDims, out_wRank);
	mulMatVec((ll*)qW2_transp_l, out_wRank, hiddenDims, wRank, out_hiddenDims);
	
	for(int i=0; i < hiddenDims; i++)
		printf("%lli\t", out_hiddenDims[i]);
	printf("\n");

	ll y[] = {5, 2, -1, -7, 9, 4, 5, 1};

	addVecs(x, y, inputDims, out_inputDims);

	for(int i=0; i < inputDims; i++)
		printf("%lli\t", out_inputDims[i]);
	printf("\n");

	subVecs(x, y, inputDims, out_inputDims);

	for(int i=0; i < inputDims; i++)
		printf("%lli\t", out_inputDims[i]);
	printf("\n");

	ll z[] = {5000, 6000, -1024, 7455, 2356, -2500, 7850, 2563};
	
	printf("Before standardization:\n");
	for(int i=0; i < inputDims; i++)
		printf("%lli\t", z[i]);
	printf("\n");

	stdScaleInput((ll*)z, inputDims);

	printf("After standardization:\n");
	for(int i=0; i < inputDims; i++)
		printf("%lli\t", z[i]);
	printf("\n");

}