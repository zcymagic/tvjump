function main_est(matfile)
allprets = 0;
load(strcat(matfile,'.mat'));


%model specification
% r,m,p,q,o,'GARCH','Normal',jump,tvjump
specs{1} = setspec(1,0,1,1,0,'GARCH','Gaussian',0,0); % garch vol
specs{2} = setspec(1,0,1,1,0,'GARCH','Gaussian',1,0); % constant jump 
specs{3} = setspec(1,0,1,1,0,'GARCH','Gaussian',1,1); % time varying jump
specs{4} = setspec(1,0,1,1,0,'GARCH','ST',0,0); % GARCH-ST vol
specs{5} = setspec(1,0,1,1,1,'GJR','Gaussian',0,0); % GJR vol

for j=1:length(specs)
    [par0{j},LLF0(j),ht0{j},et0{j},lambda0{j},ksi0{j},std0{j},cov0{j}] = marginfit(allprets,specs{j},[],[],[]); %#ok<SAGROW> % market parameter
end

if strcmp(matfile,'sp500tr')
    save resus.mat
end

N = size(dates,1);
I = dates<datenum('2007/1/1');
insamples = sum(I);
interval = 1;  


parfor tt=1:N-insamples+1
    t = tt+insamples-1;
    data = allprets(tt:t,:);    
    
    [par1(:,tt),LLF1(tt),tmp1,tmp2] = marginfit(data,specs{1},[],[],[]); %#ok<SAGROW> % market parameter
    ht1(tt)=tmp1(end);
    et1(tt)=tmp2(end);
    stdresiduals(:,tt) = tmp2./sqrt(tmp1);  %standardized residuals

    [par2(:,tt),LLF2(tt),tmp1,tmp2,tmp3] = marginfit(data,specs{2},[],[],par0{2}); %#ok<SAGROW> % market parameter
    ht2(tt)=tmp1(end);
    et2(tt)=tmp2(end);
    lambdat2(tt)=tmp3(end);

    [par3(:,tt),LLF3(tt),tmp1,tmp2,tmp3,tmp4] = marginfit(data,specs{3},[],[],par0{3}); %#ok<SAGROW> % market parameter
    ht3(tt)=tmp1(end);
    et3(tt)=tmp2(end);
    lambdat3(tt)=tmp3(end);
    ksit3(tt)=tmp4(end);


    [par4(:,tt),LLF4(tt),tmp1,tmp2] = marginfit(data,specs{4},[],[],par0{4}); %#ok<SAGROW> % market parameter
    ht4(tt)=tmp1(end);
    et4(tt)=tmp2(end);


    [par5(:,tt),LLF5(tt),tmp1,tmp2] = marginfit(data,specs{5},[],[],par0{5}); %#ok<SAGROW> % market parameter
    ht5(tt)=tmp1(end);
    et5(tt)=tmp2(end);
end

save(strcat(matfile,"_est.mat"));

