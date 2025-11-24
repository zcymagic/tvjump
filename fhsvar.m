function varp = fhsvar(parm,ret0,ht0,err0,stdresiduals,horizon,pp)

    s = RandStream.getGlobalStream();
    reset(s)

    T = length(stdresiduals);
    
    nTrials = 10000;                  % # of independent random trials
    maxhorizon = max(horizon);                     % VaR forecast horizon
    
    bootstrappedResiduals = stdresiduals(unidrnd(T,maxhorizon,nTrials));    
   
    fhsrets = simulateret(parm,ret0,ht0,err0,bootstrappedResiduals);
    lenh = length(horizon);
    varp = zeros(length(pp),lenh);
    for i=1:lenh
        cumrets = sum(fhsrets(1:horizon(i),:),1);
        varp(:,i) = quantile(cumrets,pp);
    end

end

function fhsrets = simulateret(par,ret0,ht0,err0,residuals)

    alpham = par(1);
    betam  = par(2);
    kappam = par(3); %K
    etam   = par(4); %ARCH
    psim   = par(5); %GARCH

    horizon = size(residuals,1);  
    fhsrets = residuals;
   
    ht = kappam+etam*err0^2+psim*ht0;
    et = sqrt(ht)*residuals(1,:);
    fhsrets(1,:) = alpham+betam*ret0+et;    
    for t=2:horizon
        ht = kappam+etam*et.^2+psim*ht;
        et = sqrt(ht).*residuals(t,:);
        fhsrets(t,:) = alpham+betam*fhsrets(t-1,:)+et;            
    end
end
