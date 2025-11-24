function  mainvar(matfile)

addpath('johnson\')
load(strcat(matfile,"_est.mat"))

horizon = [1 5 10 20 30];
lenh     = length(horizon);

nspec    = length(specs);
pp       = [ 0.01 0.025 0.05];
lenp     = length(pp);

N = size(allprets,1);



nTests   = N-insamples;
mm   = zeros(4,nTests,lenh,nspec);
% varp     = zeros(nTests,lenp,lenh,nspec);

 
for t=insamples:N-1
    tt = t-insamples+1;
    kk = floor((t-insamples)/interval)+1; 
    for h=1:lenh
        for j=1:nspec              
            switch j
                case 1 
                    mm(:,tt,h,j) = getpredictn(par1(:,kk),specs{j},allprets(t),ht1(tt),et1(tt),1,1,horizon(h)); 
                case 2
                    mm(:,tt,h,j) = getpredictn(par2(:,kk),specs{j},allprets(t),ht2(tt),et2(tt),lambdat2(tt),1,horizon(h));
                case 3
                    mm(:,tt,h,j) = getpredictn(par3(:,kk),specs{j},allprets(t),ht3(tt),et3(tt),lambdat3(tt),ksit3(tt),horizon(h));  
                case 4
                    mm(:,tt,h,j) = getpredictn(par4(:,kk),specs{j},allprets(t),ht4(tt),et4(tt),0,0,horizon(h));  
                case 5
                    mm(:,tt,h,j)  = getpredictn(par5(:,kk),specs{j},allprets(t),ht4(tt),et4(tt),0,0,horizon(h));  
            end            
            varp(tt,:,h,j) = johnsonvar(mm(:,tt,h,j),pp);           
        end
    end
    % garch+fhs
    varp(tt,:,:,nspec+1) = fhsvar(par1(:,kk),allprets(t),ht1(tt),et1(tt),stdresiduals(:,tt),horizon,pp); 
end



zz = zeros(lenp,lenh,nspec+1);  %np*horizon*number of models

    for j=1:nspec+1
        zz(:,:,j) = permute(-mean(varp(:,:,:,j)),[2 1 3]);
    end


% [zz(:,:,1) zz(:,:,2) zz(:,:,3)]

pro2 = cell(1,nspec+1);
for i=1:nspec+1
    pro2{i} = deal(zeros(lenp,5,lenh));
end

[boot_stat,boot_pval,mcs_pval] = deal(zeros(nspec+1,lenp,lenh));

for h = 1:lenh
    hh = horizon(h);
    cumrets = zeros(N-hh-insamples+1,1);  %realzied cumulative return from t+1 to t+hh
    for t=insamples:N-hh
        tt = t-insamples+1;
        cumrets(tt) = sum(allprets(t+1:t+hh));                
    end

    for p=1:lenp 
        for j=1:nspec+1
            loss = cumrets<=varp(1:N-hh-insamples+1,p,h,j);           

            L    = sum(loss);
            p0   = mean(loss);


            pro2{j}(p,1,h) = p0;%size(loss,1);
            LR  = 2*L.*(log(p0)-log(pp(p)))+2*(size(loss,1)-L).*(log(1-p0)-log(1-pp(p)));
            pro2{j}(p,2,h) = coveragetest(loss,pp(p),hh);
            pro2{j}(p,3,h) = LR;

            lossi = lagmatrix(loss,[1 0]);
            lossi(1,:) = [];
            n00 = sum(lossi(:,1)==0 & lossi(:,2)==0);
            n01 = sum(lossi(:,1)==0 & lossi(:,2)==1);
            n10 = sum(lossi(:,1)==1 & lossi(:,2)==0);
            n11 = sum(lossi(:,1)==1 & lossi(:,2)==1);

            pi00 = min(max(n00/(n00+n01),eps),1-eps);
            pi10 = min(max(n10/(n10+n11),eps),1-eps);
            pi = min(max((n01+n11)/size(lossi,1),eps),1-eps);

            LRi = 2*(n00*log(pi00/(1-pi))+n01*log((1-pi00)/pi)+...
                n10*log(pi10/(1-pi))+n11*log((1-pi10)/pi));
            LRc = LR+LRi;
            pro2{j}(p,4,h) = LRi; 
            pro2{j}(p,5,h) = LRc;
            pro2{j}(p,6,h) = bootstrap_mean_test(loss,pp(p),hh);
        end  %spec
        %bootstrapping 
        [stat_obj,pval_obj,pval_mcs] = bootstraptest(permute(varp(1:N-hh-insamples+1,p,h,:),[1,4,2,3]),cumrets,pp(p),3,hh);
        boot_stat(:,p,h) = stat_obj;
        boot_pval(:,p,h) = pval_obj;
        mcs_pval(:,p,h) = pval_mcs;


    end %lenp

     
end



%horizon=1
[pro2{1}(:,[1 3:5],1);pro2{2}(:,[1 3:5],1);pro2{4}(:,[1 3:5],1);pro2{5}(:,[1 3:5],1);pro2{6}(:,[1 3:5],1); pro2{3}(:,[1 3:5],1)]
z1=[pro2{1}(:,[1  6],2:5);pro2{2}(:,[1  6],2:5); pro2{4}(:,[1  6],2:5); pro2{5}(:,[1  6],2:5); pro2{6}(:,[1  6],2:5); pro2{3}(:,[1  6],2:5)];
reshape(z1,size(z1,1),[])

reshape(permute(mcs_pval,[2,1,3]),[],5)


tmp1 = reshape(permute(boot_stat([1:2,4:6,3],:,:),[2,1,3]),[],lenh);  %tail probability*model*horizon
tmp2 = reshape(permute(boot_pval([1:2,4:6,3],:,:),[2,1,3]),[],lenh);

[m,n] = size(tmp1);

for i=1:m
    strline = "";
    for j=2:n
        if tmp2(i,j)<0.01
            s=strcat(num2str(tmp1(i,j),'%.2f'),'***&');
        elseif tmp2(i,j)<0.05
            s=strcat(num2str(tmp1(i,j),'%.2f'),'**&');
        elseif tmp2(i,j)<0.1
            s=strcat(num2str(tmp1(i,j),'%.2f'),'*&');
        else
            s=strcat(num2str(tmp1(i,j),'%.2f'),'&');
        end
        strline = strcat(strline,s);
    end
    strline = strcat(strline,'\\');
    display(strline);
end


save(strcat(matfile,"_var.mat"));