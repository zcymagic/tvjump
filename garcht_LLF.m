function [LLF,h,e,lambdat,ksit,likelihoods] = garcht_LLF(par, y, spec,X,lambda0)


if isreal(par)==0
   LLF = 1e8;
   return;
end 


r  = spec.R;
m = spec.M;
p  = spec.P;
q  = spec.Q;
o  = spec.O;
vmodel = spec.VarianceModel;
dis = spec.Distribution;
jump = spec.jump;
tvjump = spec.tvjump;

if tvjump % && par(end-3)==1
    par(end-3)=min(par(end-3),1-1e-6);
end;

lenx   = size(X,2);
parm   = par(1:r+m+lenx+1);
parh   = par(r+m+lenx+2:r+m+lenx+2+p+q+o);

ind = r+m+lenx+2+p+q+o;

switch dis
    case 'T'
        nu  = par(ind+1);
        ind = ind+1; 
    case 'ST'
        nu     = par(ind+1);
        lambdad = par(ind+2);
        ind = ind+2;
end


[regressand,lags] = newlagmatrix(y,r,1);
regressors        = lags;

lr               = size(regressors,2);

regressand  = regressand-regressors*parm(1:lr);
if lenx>0
    regressand = regressand-X(r+1:end)*parm(lr+1:end);
end

regressand = [zeros(r,1);regressand];


if m==0 
    e = regressand;
else    
    e = maxcore(regressand,parm(lr+1:end),m,length(regressand));
end

stdEstimate =  std(e(r+1:end),1); 

tau  = max([p,q,o]);
data = [stdEstimate(ones(tau+r,1));e(r+1:end)];
T    = size(data,1);

dataneg=(data<0).*data;
dataneg(1:tau)=data(1)/2;


switch vmodel;
    case 'GARCH'
        h = garchcore(data,parh,stdEstimate^2,p,q,tau,T);
    case 'GJR'
        h = tarchcore(data.^2,dataneg.^2,parh,stdEstimate^2,p,q,o,tau,T); 
    case 'TARCH';
        h = tarchcore(abs(data),abs(dataneg),parh,stdEstimate,p,q,o.tau,T); 
        h = h.^2;
    case 'EGARCH'
        h = egarchcore(data, parh, stdEstimate, p, q ,o,tau , T);
        temp=min(h(h>0))/100;
        h(isnan(h))=temp;
        h(isinf(h))=temp;
        h(h<=0)=temp;
end

% par
% min(h)
t = (tau+1:T)';
h = h(t);
e = data(t);

if min(h)<0
    LLF=1e6;
    return;
end

if strcmp(dis,'ST')
    stdresid = e./sqrt(h);
    likelihoods = log(skewtdis_pdf(stdresid,nu,lambdad)./sqrt(h));    
    lambdat = 1;
    ksit = 1;
elseif jump
    par(end)=par(end)^2;
    if tvjump        
        if isempty(lambda0)
            [likelihoods,lambdat,ksit]=arji1(e',par(ind+1:end),h,-1,0);
        else
            likelihoods = jump_LLF(e',par(ind+1:end),h,lambda0);
            lambdat = lambda0;
            ksit = 1;
        end
    else
        if isempty(lambda0)
            likelihoods = ji1(e,par(ind+1:end),h);
            lambdat = par(ind+1)*ones(length(e),1);
            ksit = ones(length(e),1);
        else
            likelihoods = jump_LLF(e',par(ind+1:end),h,lambda0*ones(size(e,1),1));
            lambdat = lambda0;
            ksit = 1;
        end
    end    
else   
    likelihoods = log(normpdf(e,0,sqrt(h)));
    lambdat = 1;
    ksit = 1;
end



LLF = -sum(likelihoods);

if min(lambdat)<0 
    LLF=1e8;
end

if isnan(LLF)
    LLF=1e6;
end

if isreal(LLF)==0
   LLF = 1e7;
end

if isfinite(LLF)==0
    LLF = 1e9;
end

dummy3 = 1;