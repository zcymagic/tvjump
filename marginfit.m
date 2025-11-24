function [par1,LLF,ht,et,lambdat,ksit,std1,cov1] = marginfit(y,spec,X,lambda0,pari)

r =   spec.R;
m = spec.M;
p =  spec.P;
q = spec.Q; % p as GARCH and q as ARCH
o = spec.O;
vmodel = spec.VarianceModel;
dis = spec.Distribution;
jump = spec.jump;
tvjump = spec.tvjump;

constant=1;
lenx = size(X,2);

A1=[zeros(r,constant) eye(r)   zeros(r,m+lenx);...
        zeros(r,constant) -eye(r)   zeros(r,m+lenx);...
        zeros(m,constant) zeros(m,r) eye(m) zeros(m,lenx);...
        zeros(m,constant) zeros(m,r)  -eye(m) zeros(m,lenx)];
B1=ones(size(A1,1),1);

switch vmodel
    case 'GARCH'
        switch dis
            case 'Gaussian'
                A2 =  [-eye(1+p+q);...
                    0  ones(1,p+q)];  
                B2 =  [zeros(1+p+q,1); 1];               
            case 'ST'
                A2 =  [-eye(1+p+q) zeros(1+p+q,2);...
                      0  ones(1,p)  ones(1,q) zeros(1,2);...
                      [zeros(2,1+p+q) [-1;1] [0;0]];[zeros(2,1+p+q+1),[-1;1]]];

                B2 =  [zeros(1+p+q,1); 1 ; -2.1;200;ones(2,1)];      

        end
    case 'GJR' 
       A2 =  [-eye(1+p+q+o);...
           0  ones(1,p+q) 0.5*ones(1,o)];  
       B2 =  [zeros(1+p+q+o,1); 1];     

end

if jump
    Aj = [0 -1];
    Bj = 0;

    if tvjump
        Ai = [-1 0 0;0 1 0;0 -1 1;0 0 -1];
        Bi = [0;1;0;0];
    else
        Ai = -1;
        Bi = 0;
    end
else
    Ai = [];
    Bi = [];
    Aj = [];
    Bj = [];
end

if isempty(lambda0)
    A = blkdiag(A1,A2,Ai,Aj);
    B = [B1;B2;Bi;Bj];
else
    A = blkdiag(A1,A2,Aj);
    B = [B1;B2;Bj];
end

B = B-1e-6;
% 
opt = optimset('fmincon');
opt   = optimset(opt,'MaxIter',10000,'MaxFunEvals',90000,'Display','off');


para_mu=armaxfilter(y,1,r,m,[]);
if strcmp(vmodel,'GARCH')
    para_sigma = [0.001 0.1 0.9]';
else
    para_sigma = [0.001 0.1 -0.1 0.9]';
end
par0 = [para_mu';para_sigma];

if strcmp(dis,'ST')
    par0 = [par0;5;-0.1];
end

if jump
    if tvjump 
        par0 = [par0;0;0.9;0.01;-0.1;0.1];
    else
        par0 = [par0;0;0.01*ones(2,1)];
    end    
end   

[par1,LLF1] =  fmincon(@garcht_LLF, par0,A, B ,[] , [] , [] , [],[],opt,y,spec,X,lambda0);

if ~isempty(pari)
    [par2,LLF2] =  fmincon(@garcht_LLF, pari,A, B ,[] , [] , [] , [],[],opt,y,spec,X,lambda0);
    if LLF2<LLF1        
        par1=par2;
    end
end


[LLF, ht, et,lambdat, ksit ]  = garcht_LLF(par1, y, spec,X,lambda0);


if nargout>6
    [std1,cov1] = getRobuststd(@garcht_LLF, par1,y,spec,X,lambda0);
end
 