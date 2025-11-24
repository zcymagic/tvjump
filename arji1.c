#include "mex.h"
#include "math.h"


#define NJUMP 5 
#define PI 3.141592653589793

/* for dims=2 only */

double factor[11] = {1,1,2,6,24,120,720,5040,40320,362880,3628800};


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



void arji1(double *data, double *par, double * H, int dims,int nobs, double *llf,double *lambdat,double *ksit,double *lambda0,double *ksi0)
{
	int i,j,t,dims2,ind;
    double *mu_J,*ut,*Ht,*HI,*prob;
    double lambdap[3],probs;
    
    dims2 = dims*dims;

    mu_J  = mxMalloc(dims*sizeof(double)); 
    ut    = mxMalloc(dims*sizeof(double)); 
    Ht    = mxMalloc(dims2*sizeof(double)); 
    HI    = mxMalloc(dims2*sizeof(double)); 
    prob  = mxMalloc((NJUMP+1)*sizeof(double)); 
    
   
  
    /* time-varying jump intensity  */
    for(i=0;i<3;i++)    lambdap[i] = par[i];
    
    if(lambda0[0]<0) 
        lambdat[0] = lambdap[0]/(1-lambdap[1]);
    else
        lambdat[0] = lambdap[0]+lambdap[1]*lambda0[0]+lambdap[2]*ksi0[0];;
    
        
        
    
    
    ind = 3;
    /*jump size */
    for(i=0;i<dims;i++)
        mu_J[i]  = par[ind+i];
    
    ind = ind+dims;
    
    for(i=0;i<dims;i++) {
        for(j=0;j<dims;j++) {
            HI[i*dims+j] = par[ind+i*dims+j];
        }
    }
    
    probs = 0;    
    for(i=0;i<=NJUMP;i++)
    {
        ut[0] = data[0]+(lambdat[0]-i)*mu_J[0];
        Ht[0] = H[0]+HI[0]*i;  
        prob[i] = normp(ut,Ht)*possp(i,lambdat[0]);
        probs  += prob[i];
    }
    
    
    llf[0] = log(probs);
    
    /* get jump innovation */
    ksit[0] = 0;
    for(i=1;i<=NJUMP;i++)
        ksit[0] += prob[i]*i;
    ksit[0] /= probs;
    ksit[0] -= lambdat[0];
        
    for(t=1;t<nobs;t++)
    {        
        lambdat[t] = lambdap[0]+lambdap[1]*lambdat[t-1]+lambdap[2]*ksit[t-1];        
        probs = 0;
        for(i=0;i<=NJUMP;i++)
        {
            ut[0] = data[t*dims]+(lambdat[t]-i)*mu_J[0];
            Ht[0] = H[dims2*t]+HI[0]*i;  
            prob[i] = normp(ut,Ht)*possp(i,lambdat[t]);
            probs  += prob[i];
            
        }        
        llf[t] = log(probs);
        
        /* get jump innovation */
        ksit[t] = 0;
        for(i=1;i<=NJUMP;i++)
            ksit[t] += prob[i]*i;
        ksit[t] /= probs;
        ksit[t] -= lambdat[t];
    }
    
    mxFree(mu_J);
    mxFree(HI);
    mxFree(Ht);
    mxFree(ut);
    mxFree(prob);
}


    

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *data, *par, *H,*llf,*lambdat,*ksit,*lambda0,*ksi0;
	int  nobs,dims,tmp;
	
	/*  Check for proper number of arguments. */
	if(nrhs!=5)  /*Three inputs are data,para,H*/
		mexErrMsgTxt("Five inputs required.");
	if(nlhs!=3)  /*two outputs are likelihoods and lambda(t) */
		mexErrMsgTxt("Three output required.");	
	
	
  /*  Create a pointer to the input matrices . */
	data    = mxGetPr(prhs[0]);
	par     = mxGetPr(prhs[1]);
	H       = mxGetPr(prhs[2]);	
    lambda0 = mxGetPr(prhs[3]);	
    ksi0    = mxGetPr(prhs[4]);
		
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
    plhs[1] = mxCreateDoubleMatrix(nobs,1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(nobs,1, mxREAL);
	
	/*  Create a C pointer to a copy of the output matrix. */
	llf     = mxGetPr(plhs[0]);
    lambdat = mxGetPr(plhs[1]);
    ksit    = mxGetPr(plhs[2]);
	
	/*  Call the C subroutine. */
	arji1(data,par,H,dims,nobs,llf,lambdat,ksit,lambda0,ksi0);	
}

			
