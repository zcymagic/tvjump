function varp=johnsonvar(mm,p)

res=f_johnson_fit(mm);
gamma = res.coef(1);
delta = res.coef(2);
xi = res.coef(3);
lambda = res.coef(4);

if lambda>0 
    u = (norminv(p)-gamma)./delta;
else
    u = (norminv(1-p)-gamma)./delta;
end

%  SL: Lognormal distribution = exponential transform
%  SU: Unbounded distribution = hyperbolic sine transform
%  SB: Bounded distribution   = logistic transforma
%  SN: Normal distribution    = identity transform
switch(res.type)
    case 'SL'
        tmpval = exp(u);
    case 'SU'
        tmpval = sinh(u);
    case {'SB','ST'}
        tmpval = 1./(1+exp(-u));
    case 'SN'
        tmpval = u;
end

varp = xi+lambda*tmpval;

        
