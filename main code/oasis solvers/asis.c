#include "mex.h" 
 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 

    
    int m,n; // matrix dims
    int i,ell,r,c; // loop vars
    int L;    //  Number of Cols to Sample
    int verbose;  // display output?
    int maxInd;  //  Index with maximal error, i.e. next index to sample
    double *A, *R, *D, *C, *E, *I, *B,  d;
    double maxVal, shur;
   
    
    if(nrhs!=3)
       mexErrMsgTxt("Must have three inputs:\n\tmatrix(double array)\n\tnumColsToSample(double representation of an int)\n\tisVerbose (int representation of a double)"); 

    
    
    m = mxGetM(prhs[2]); 
    n = mxGetN(prhs[2]);
    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || !(m == 1 && n == 1)) {
        mexErrMsgTxt("isVerbose must be a noncomplex scalar double.");
    }
    verbose = (int) mxGetPr(prhs[2])[0];
    
    m = mxGetM(prhs[1]); 
    n = mxGetN(prhs[1]);
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !(m == 1 && n == 1)) {
        mexErrMsgTxt("L must be a noncomplex scalar double.");
    }
    L = (int) mxGetPr(prhs[1])[0];
    
    m = mxGetM(prhs[0]); 
    n = mxGetN(prhs[0]);
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || !(m == n)) {
        mexErrMsgTxt("A must be square.");
    }
    A =  mxGetPr(prhs[0]);

    
  
  
    
    /* Create an array for the output's data */ 
     I = (double *) mxMalloc(L * sizeof(double));    //  The indices chosen
     R = (double *) mxMalloc(m*L * sizeof(double));  // Representation, W^-1*C
     C = (double *) mxMalloc(m*L * sizeof(double));  //  The cols, C
     D = (double *) mxMalloc(m * sizeof(double));    //  The diagonal of A
     E = (double *) mxMalloc(m * sizeof(double));    //  Used 2X:  Stores B*rep first, and the error later in the loop
     
     for (c = 0; c < L; c++){
         I[c] = 0;
     }
     
     for (c = 0; c < m; c++) 
       D[c] = A[c+m*c];
     
     maxVal = D[0]*D[0];
     maxInd = 0;
     for (c = 0; c < m; c++){
         if(D[c]*D[c]>maxVal){
            maxVal = D[c]*D[c];
            maxInd = c;
         }
     }

     if(verbose){
         mexPrintf("\tasis (C implementation, mex interface)\n");
         mexPrintf("\t\tSelecting %i columns:",L);
     }
     
     for(ell=0;ell<L;ell++){
         //mexPrintf("------------%i\n",ell);
         //mexPrintf("\nInd = %d\n",maxInd);
         /* Find Shur complement */
         shur = 0;
         for(c=0;c<ell;c++)
            shur+=R[maxInd+c*m]*C[maxInd+c*m];
         shur = 1.0/(D[maxInd]-shur);
         
         
         /* Find b*rep */
         for (c = 0; c < m; c++)
           E[c] = 0; 
         for(r=0;r<ell;r++){
             d = C[r*m+maxInd];
              for (c = 0; c < m; c++)
                  E[c] += R[c+r*m]*d;
         }
            
         /* subtract newCol */
         for (c = 0; c < m; c++)
           E[c] -= A[maxInd*m+c];   /* Bis now brep-newCol*/
         
        
         /* update */
         for(r=0;r<ell;r++){
             d = R[r*m+maxInd];
              for (c = 0; c < m; c++)
                  R[r*m+c]+=d*shur*E[c];
         }
         for (c = 0; c < m; c++){
              R[c+m*ell]= -shur*E[c];
         }    
         
         
         /* copy new col */ 
         for (c = 0; c < m; c++)
           C[c+m*ell] = A[n*maxInd+c]; 
         I[ell] =  maxInd;
         
         
          /* Fill E with -D  */
         for (c = 0; c < m; c++)
            E[c] = -D[c];
         
         /* Compute Errors */
         for (c = 0; c < m*(ell+1); c++)
            E[c%m] += R[c]*C[c]; 
         
         
         
         /* Find Max Error */
         maxVal = E[0]*E[0];
         maxInd = 0;
         for (c = 0; c < m; c++){
             if(E[c]*E[c]>maxVal){
                maxVal = E[c]*E[c];
                maxInd = c;
             }
         }
         
         if(verbose){
             if(ell%50==0)
                 mexPrintf("\n\t");
             mexPrintf(".");
             if(ell%50==0 || m>50000)
             mexEvalString("drawnow;"); 
         }
     }
     
      if(verbose){
         mexPrintf("\n");
      }
     
     /* Assign the data array to the output array */
     mxFree(E);
     mxFree(D);
     mxFree(C);
     mxFree(R);
     plhs[0] = mxCreateDoubleMatrix(1 , L, mxREAL);
     mxSetPr(plhs[0], I); 


} 
