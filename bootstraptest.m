function [stat_obj,pval_obj,pval_mcs] = bootstraptest(varf,rets,pp,benchmarkid,hh) 


s = RandStream.getGlobalStream();
reset(s)

B = 10000; 
[Nb,dim] = size(varf);

idx = block_bootstrap((1:Nb)',B,hh);

hits = pp-(rets < varf);

losses = hits.*(rets-varf);
stat_obj = mean(losses,1);

boot_obj = zeros(dim,B);

for m=1:B
    bootrets = rets(idx(:,m),:);
    bootvarf = varf(idx(:,m),:);    
    boot_obj(:,m) = mean((pp-(bootrets < bootvarf)).*(bootrets-bootvarf),1);    
end

boot_indicator = boot_obj<boot_obj(benchmarkid,:);
pval_obj = mean(boot_indicator,2);


%mcs test
[included,pval]=mcs(losses,1e-10,B,hh,'BLOCK');
zz = sortrows([included,pval]);
pval_mcs = [zz(1:benchmarkid-1,2);zz(benchmarkid+1:end,2);zz(benchmarkid,2)];


