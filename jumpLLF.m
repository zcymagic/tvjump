function LLF = jumpLLF(par,e,h,lambda0,tvjump)

ind = 0;
if tvjump
    if isempty(lambda0)
        [likelihoods,lambdat,~]=arji1(e',par(ind+1:end),h,-1,0);
    else
        likelihoods = jump_LLF(e',par(ind+1:end),h,lambda0);
    end
else
    if isempty(lambda0)
        likelihoods = ji1(e,par(ind+1:end),h);

    else
        likelihoods = jump_LLF(e',par(ind+1:end),h,lambda0*ones(size(e,1),1));

    end
end    



LLF = -sum(likelihoods);

if isnan(LLF)
    LLF=1e6;
end

if isreal(LLF)==0
   LLF = 1e7;
end

if lambdat(1)<0
    LLF = 1e8;
end

if isfinite(LLF)==0
    LLF = 1e9;
end