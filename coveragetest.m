function [stat,pval] = coveragetest(data,mu0,lag)


data = data(:);
mu = mean(data);
T = length(data);

gamma = zeros(lag, 1);
for j = 0:lag-1
    gamma(j+1) = (data(1:T-j) - mu)' * (data(j+1:T) - mu) / T;
end

% weights = 1-[1:lag-1]'/lag;
varval = gamma(1)+2*sum(gamma(2:end));

if varval>0
    stat = abs(mu-mu0)./sqrt(varval)*sqrt(length(data));
    pval = (1-normcdf(stat))*2;
else
    stat = nan;
    pval = 0;
end




