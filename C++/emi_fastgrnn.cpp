#include <stdio.h>
#include <stdlib.h>

#define min(a,b) ((b)>(a))?a:b
#define max(a,b) ((a)>(b))?a:b

#define UPDATE_NL quantTanh
#define GATE_NL quantSigm

typedef long long ll;

/*************************** BEGIN PASTE MODEL PARAMS **************************/

static const ll qW1_transp_l[][8] = {{-1,2,-6,-4,5,1,4,6},{0,-5,-6,1,4,7,0,-6},{-2,-6,6,-1,-6,3,2,4},{-5,-1,6,7,6,1,-3,0},{-7,-6,2,-1,2,-3,-1,1}};
static const ll qFC_Bias_l[2] = {21,35};
static const ll qW2_transp_l[][5] = {{-1,-2,-1,-4,3},{-2,-10,0,2,1},{9,-3,7,-15,-5},{-5,2,-3,-8,9},{-1,0,-7,-6,-6},{-4,-2,-2,-1,-3},{2,2,2,1,3},{10,-4,-5,-1,4},{-6,2,8,-2,-7},{-4,9,-4,10,5},{-6,-17,-3,-4,-3},{-2,2,-4,-1,-4},{-4,0,-3,-7,10},{4,-1,2,2,-3},{2,1,0,0,2},{4,13,2,1,0},{2,0,1,3,-2},{9,-2,8,9,-15},{-12,-2,-7,4,0},{5,-1,7,-7,-2},{6,-2,0,6,-8},{-5,1,-6,8,1},{-4,-9,-4,-7,-5},{-11,5,4,7,-5},{2,1,2,-1,0},{0,-1,1,1,0},{-1,0,0,-2,5},{10,-2,3,13,-16},{7,2,12,6,5},{-6,1,-3,-8,10},{1,0,-1,2,-4},{0,0,1,3,-2}};
static const ll qU2_transp_l[][6] = {{-5,21,-12,16,15,3},{5,1,-1,1,-1,-1},{-4,3,-1,-3,9,-4},{4,-7,-6,-10,3,-5},{5,-2,-4,1,-5,3},{-9,-3,3,-1,11,-9},{-4,6,5,-2,-17,7},{-3,-6,-2,2,-2,-2},{-4,3,-1,-1,3,-2},{-5,-4,3,0,4,-3},{-2,2,6,-3,4,-1},{-4,-8,-3,1,8,-1},{4,2,-22,-13,13,5},{-16,5,3,-8,-3,14},{5,-5,-7,3,-11,6},{2,-5,2,0,6,-3},{6,-8,4,3,-7,1},{1,2,4,-3,2,-8},{17,6,-9,-16,0,-3},{1,-5,6,2,1,1},{-6,-11,-15,-18,12,9},{8,3,0,0,3,0},{2,0,1,1,-17,7},{5,4,-8,-3,-2,1},{10,12,-8,-3,1,2},{1,-1,-1,-6,7,-9},{8,-10,15,3,-7,8},{-13,-5,-13,0,-1,15},{-8,-3,-5,-2,8,-7},{2,-5,6,0,-2,-7},{-3,7,-12,-6,11,-8},{0,-15,2,0,-3,1}};
static const ll qFC_Weight_l[][32] = {{40,-22,35,83,-52,-1,33,-82,-13,26,65,92,-57,-73,-42,-67,34,33,3,-63,41,-64,-75,-44,47,23,-56,44,-74,-4,-29,-46},{-31,79,-47,-21,54,-23,-77,23,-5,-86,-32,-128,45,53,19,100,-62,-43,-48,34,-17,53,17,59,-84,-25,54,7,44,38,11,-14}};
static const ll qU1_transp_l[][32] = {{-5,2,3,-4,9,-19,-2,12,11,1,4,4,2,-2,-4,-1,11,1,0,-4,0,-3,-2,3,-1,4,5,-1,6,3,8,12},{11,-5,1,-5,-6,5,5,-7,-7,-3,2,1,0,-1,-9,-2,1,1,-2,-2,-2,-4,-1,-4,17,22,-5,-2,-4,-3,4,13},{1,-12,-2,1,1,14,3,4,1,-1,1,-1,-2,1,4,-10,13,-3,3,-14,7,-9,-4,2,-1,-7,5,-1,2,3,-10,-1},{-7,2,1,-3,4,-6,1,3,6,3,1,2,10,1,8,1,-1,1,-1,2,-4,2,0,1,-7,-6,5,2,3,11,-6,0},{1,-2,-1,1,14,5,-8,14,3,0,-1,18,2,-1,-18,-1,-13,2,1,-7,-3,-7,-3,12,-19,14,0,2,6,2,10,-6},{1,-6,-2,4,4,-7,2,5,-15,-4,-1,-6,0,0,7,1,-8,2,2,-2,-4,-5,5,13,4,0,8,10,4,-9,-7,3}};
static const ll qB_g_l[32] = {1300000,1700000,1600000,800000,1800000,1600000,1600000,2100000,2000000,1600000,1700000,1900000,1000000,1400000,1400000,2100000,1600000,1800000,1600000,1900000,1400000,1800000,2000000,2000000,1500000,1600000,1700000,1200000,2000000,1600000,1300000,-700000};
static const ll qB_h_l[32] = {4700000,8500000,2100000,1600000,11800000,3900000,2400000,6000000,7900000,1900000,2300000,3700000,2000000,2400000,4900000,10800000,3100000,2200000,1400000,7600000,1000000,7300000,7900000,12600000,3400000,3100000,2900000,1600000,11900000,6300000,3700000,100000};

static const ll mean_l[8] = {2027,2032,2027,2032,2027,2032,2027,2032};
static const ll stdev_l[8] = {30,31,30,31,30,31,30,31};

static const ll q_l= 15;
static const ll I_l= 100000;
static const ll q_times_I_l= 1500000;
static const ll I_l_vec[32] = {100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000};

static const int wRank = 5;
static const int uRank = 6;
static const int inputDims = 8;
static const int hiddenDims = 32;
static const int timeSteps = 12;
static const int numInstances = 8;
static const int numClasses = 2;

/*************************** END PASTE MODEL PARAMS **************************/

ll out_wRank[wRank] = {0};
ll out_uRank[uRank] = {0};
ll out_hiddenDims[hiddenDims] = {0};
ll out_numClasses[numClasses] = {0};

// Copy uint into ll array
void copyUIntVecToLL(uint* invec, ll* outvec, int vec_len)
{
	for(int i=0; i < vec_len; i++)
		*(outvec + i) = (ll)invec[i];
}

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
			out[i] += *((mat+i*vec_len)+j)*(*(vec+j));
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
		out[i] = *(vec1+i)*(*(vec2+i));
}

// Standardization with scaling
void stdScaleInput(ll* in_vec, int vec_len, ll* out_vec){
	for(int i=0; i<vec_len; i++)
		*(out_vec+i) = I_l*(*(in_vec+i)-mean_l[i])/stdev_l[i];
}

// quantTanh
void quantTanh(ll* vec, int vec_len, ll scale, ll* out_vec){
	for(int i=0; i<vec_len; i++)
		*(out_vec+i) = max(-scale, min(scale, *(vec+i)));
}

// quantSigm
void quantSigm(ll* vec, int vec_len, ll scale, ll* out_vec){
	for(int i=0; i<vec_len; i++)
		*(out_vec+i) = max(min((*(vec+i)+scale)/2, scale),0);
}

// Vector print utility
void util_printVec(ll* vec, int vec_len){
	for(int i=0; i < vec_len; i++)
		printf("%lli\t", vec[i]);
	printf("\n\n");
}

// Vector deep copy
void util_deepCopy(uint* src, uint*dst, int row_index, int vec_len){
	for(int j=0; j < vec_len; j++)
		*(dst+j) = *((src+row_index*vec_len)+j);
}

void util_deepCopy(ll* src, ll*dst, int row_index, int vec_len){
	for(int j=0; j < vec_len; j++)
		*(dst+j) = *((src+row_index*vec_len)+j);
}

int main(){
	int size = sizeof(qW1_transp_l) + sizeof(qFC_Bias_l) + sizeof(qW2_transp_l) + sizeof(qU2_transp_l) + sizeof(qFC_Weight_l) + sizeof(qU1_transp_l) + sizeof(qB_g_l) + sizeof(qB_h_l) + sizeof(q_l) + sizeof(I_l) + sizeof(mean_l) + sizeof(stdev_l) + sizeof(I_l_vec) + sizeof(q_times_I_l);
	printf("Model size: %d KB\n\n", size/1000);

	uint test_input[][8] = {{2032,2041,2019,2014,2034,2021,2038,2053},{2017,2051,2017,2041,2039,2050,2013,2056},{2016,2035,2021,2040,2008,2059,2026,2046},{2049,2050,2026,2067,2025,2041,2039,2025},{2025,2056,1994,2039,2015,2037,2029,2049},{2018,2060,2014,2046,2036,2029,2018,2045},{2020,2031,2037,2023,2001,2056,2002,2029},{2031,2028,2041,2059,2024,2060,2052,2036},{2045,2034,2012,2030,2019,1965,2039,2023},{2014,2073,2015,2031,2054,2026,2028,2060},{2023,2020,2046,2020,2035,2037,2012,2029},{2050,2019,2055,2042,2019,2063,2020,2043}};

	uint test_input2[][8] = {{2045,2034,2012,2030,2019,1965,2039,2023},{2014,2073,2015,2031,2054,2026,2028,2060},{2023,2020,2046,2020,2035,2037,2012,2029},{2050,2019,2055,2042,2019,2063,2020,2043},{2039,2043,2013,2059,2001,2020,2044,2034},{2029,2058,2002,2046,2040,2017,2051,2050},{2026,2070,2034,2020,2042,2019,2009,2042},{2019,2031,2054,2046,2040,2058,2033,2048},{2045,2027,2037,2070,2009,2076,2026,2026},{2051,2047,2005,2055,2009,2030,2043,2057},{2017,2059,2020,2049,2041,2052,2007,2049},{2021,2016,2028,2036,2015,2052,2007,2043}};

	ll h[hiddenDims] = {0};

	for(int t=0; t<timeSteps; t++){
		//util_printVec(h, hiddenDims);
		uint x_int[inputDims] = {0};
		util_deepCopy((uint*)test_input2, x_int, t, inputDims);

		ll x[8] = {};

		copyUIntVecToLL(x_int, x, inputDims);

		//printf("Current input array in ll\n");
		//util_printVec(x, inputDims);

		stdScaleInput(x, inputDims, x);

		//printf("Post-standardization input\n");
		//util_printVec(x, inputDims);

		// Precompute
		ll pre[hiddenDims] = {0};
		mulMatVec((ll*)qW1_transp_l, x, wRank, inputDims, out_wRank);
		mulMatVec((ll*)qW2_transp_l, out_wRank, hiddenDims, wRank, pre);

		//util_printVec(pre, hiddenDims);

		mulMatVec((ll*)qU1_transp_l, h, uRank, hiddenDims, out_uRank);
		mulMatVec((ll*)qU2_transp_l, out_uRank, hiddenDims, uRank, out_hiddenDims);

		addVecs(pre, out_hiddenDims, hiddenDims, pre);

		//util_printVec(pre, hiddenDims);

		divVecScal(pre, q_l, hiddenDims, pre);

		//printf("Pre at t=%d:\n", t);
		//util_printVec(pre, hiddenDims);

		// Create h_, z
		ll h_[hiddenDims] = {0};
		ll z[hiddenDims] = {0};

		addVecs(pre, (ll*)qB_h_l, hiddenDims, h_);
		addVecs(pre, (ll*)qB_g_l, hiddenDims, z);

		UPDATE_NL(h_, hiddenDims, q_times_I_l, h_);
		divVecScal(h_, q_l, hiddenDims, h_);

		GATE_NL(z, hiddenDims, q_times_I_l, z);
		divVecScal(z, q_l, hiddenDims, z);

		//printf("h_ at t=%d:\n", t);
		//util_printVec(h_, hiddenDims);

		//printf("z at t=%d:\n", t);
		//util_printVec(z, hiddenDims);

		// Create new h
		mulVecs(z, h, hiddenDims, h);

		subVecs((ll*)I_l_vec, z, hiddenDims, out_hiddenDims);
		mulVecScal(out_hiddenDims, I_l, hiddenDims, out_hiddenDims);
		mulVecs(out_hiddenDims, h_, hiddenDims, out_hiddenDims);
		divVecScal(out_hiddenDims, I_l, hiddenDims, out_hiddenDims);

		addVecs(h, out_hiddenDims, hiddenDims, h);
		divVecScal(h, I_l, hiddenDims, h);

		//printf("h at t=%d:\n", t);
		//util_printVec(h, hiddenDims);
	}

	// Classify
	mulMatVec((ll*)qFC_Weight_l, h, numClasses, hiddenDims, out_numClasses);
	addVecs(out_numClasses, (ll*)qFC_Bias_l, numClasses, out_numClasses);

	printf("Classification output:\n");
	util_printVec(out_numClasses, numClasses);
}