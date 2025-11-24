#include "mex.h"
#include "math.h"

#define NJUMP 10 
#define PI 3.141592653589793

/* for dims=2 only */

double factor[NJUMP+1] = {1,1,2,6,24,120,720,5040,40320,362880,3628800};

double normp(double *ut,double *Ht)
{
    double z;       
    z = -ut[0]*ut[0]/(2*Ht[0]);
    return exp(z)/sqrt(2*PI*Ht[0]);
}

double possp(int n,double lambda)
{
    return pow(lambda,n)*exp(-lambda)/factor[n];
}



void jump_LLF(double *data, double *par, double * H, int dims,int nobs, double *llf,double *lambdat)
{
	int i,j,t,dims2,ind;
    double *mu_J,*ut,*Ht,*HI,*prob;
    double probs,ksi;
    
    dims2 = dims*dims;

    mu_J  = mxMalloc(dims*sizeof(double)); 
    ut    = mxMalloc(dims*sizeof(double)); 
    Ht    = mxMalloc(dims2*sizeof(double)); 
    HI    = mxMalloc(dims2*sizeof(double)); 
    prob  = mxMalloc((NJUMP+1)*sizeof(double)); 
    
  
    ind = 0;
    /*jump size */
    for(i=0;i<dims;i++)
        mu_J[i]  = par[ind+i];
    
    ind = ind+dims;
    
    for(i=0;i<dims;i++) {
        for(j=0;j<dims;j++) {
            HI[i*dims+j] = par[ind+i*dims+j];
        }
    }   
   
        
    for(t=0;t<nobs;t++)
    { 
        probs = 0;
        for(i=0;i<=NJUMP;i++)
        {
            ut[0] = data[t*dims]+(lambdat[t]-i)*mu_J[0];
            Ht[0] = H[dims2*t]+HI[0]*i;  
            prob[i] = normp(ut,Ht)*possp(i,lambdat[t]);
            probs  += prob[i];              
        }        
        llf[t] = log(probs);
    }
    
    mxFree(mu_J);
    mxFree(HI);
    mxFree(Ht);
    mxFree(ut);
    mxFree(prob);
}


    

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *data, *par, *H,*llf,*lambdat;
	int  nobs,dims,tmp;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=4)  /*Three inputs are data,para,H*/
		mexErrMsgTxt("Four inputs required.");
	if(nlhs!=1)  /*two outputs are likelihoods and lambda(t) */
		mexErrMsgTxt("One output required.");	
	
	
  /*  Create a pointer to the input matrices . */
	data    = mxGetPr(prhs[0]);
	par     = mxGetPr(prhs[1]);
	H       = mxGetPr(prhs[2]);	
    lambdat = mxGetPr(prhs[3]);
		
	/*  Get the dimensions of the matrix input to make an output matrix. */
	dims = mxGetM(prhs[0]);
	nobs = mxGetN(prhs[0]);
    if(dims>nobs) {
        tmp  = nobs;
        nobs = dims;
        dims = tmp;
    }
           		
	
	/*  Set the output pointer to the output matrix. */
	plhs[0] = mxCreateDoubleMatrix(nobs,1, mxREAL);
	
	/*  Create a C pointer to a copy of the output matrix. */
	llf     = mxGetPr(plhs[0]);
	
	/*  Call the C subroutine. */
	jump_LLF(data,par,H,dims,nobs,llf,lambdat);	
}

			
