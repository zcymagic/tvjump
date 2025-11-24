function mm=getpredictnew(parm,spec,retp,htp,lambdap,nhorizon)
%predicted mean and variance in next period
% if nhorizon==1
%     mm = getpredict2(parm,spec,retp,htp,lambdap);
%     return;
% end

r = spec.R;
m = spec.M;
p =  spec.P;
q = spec.Q; % p as GARCH and q as ARCH
o = spec.O;
vmodel = spec.VarianceModel;
dis = spec.Distribution;
jump = spec.jump;
tvjump = spec.tvjump;
mind   = 1;

emret  = zeros(1,mind);
[hm,Elambda]  = deal(zeros(nhorizon,1));

horizonm = [1:nhorizon]';
horizonv = [1:nhorizon-1]';
lenx = 0;
ind  = 0;
for i=1:mind  %mind=1
    parm1(:,i)   = parm(1:1+r+m+lenx,i);  %#ok<*AGROW>
    parh1(:,i)   = parm(2+r+m+lenx:2+r+m+lenx+p+q+o,i);
    ind = 2+r+m+lenx+p+q+o;
    if jump
        parj1(:,i)   = parm(ind+1:ind+jump*(1+2*tvjump+2),i); %#ok<*NASGU>        
    end
    
    alpham = parm(1,i);
    betam  = parm(2,i); 
    
    %expected market return after nhorizon
    meanp = (1-betam.^horizonm)./(1-betam)*alpham+betam.^(horizonm-1)*(retp-alpham);
    emret(:,i) = sum(meanp);
%     mubar = alpham/(1-betam);
%     tmp   = mubar+betam.^horizonm*(lagret(i)-mubar);
%     emret(:,i) = sum(tmp);
%     emret2 = sum((1-betam.^horizonm)./(1-betam)*alpham+betam.^horizonm*lagret(i));
    
    y = (1-betam.^flipud(horizonm))/(1-betam);
    if p>0
        kappam = parh1(1,i);
        etam   = parh1(2,i);
        psim   = parh1(3,i);
        hm(1) = htp;
    else
        hm(1) = parh1(1,i);
    end
    
    
    if ~jump   %no jump, model 1
        varpbar = kappam/(1-etam-psim);
        hm(2:end) = varpbar+(etam+psim).^horizonv*(hm(1)-varpbar);
    else       %jump correlation, model 2  
        if tvjump
            %jump intensity  
            theta = parj1(1);
            rho = parj1(2);
            phi = parj1(3);
            Elambda(1,i) = lambdap;
            
            mu_J  = parj1(4);
            sig_J = parj1(5)^2;
            lambdabar = parj1(1)/(1-rho);
            Elambda(2:end,i) = lambdabar+rho.^horizonv*(Elambda(1,i)-lambdabar);
        else
            theta = parj1(1);
            rho = 0;
            phi = 0;
            
            Elambda = theta*ones(nhorizon,1);
            mu_J  = parj1(2);
            sig_J = parj1(3)^2;
        end
        Evarj = Elambda(:,i)*(mu_J^2+sig_J);
        for t=2:nhorizon
            hm(t) = kappam+(etam+psim)*hm(t-1)+etam*Evarj(t-1);
        end
    end   
end

% sigmaV = zeros(sind,sind);
% beta = [1 beta];
sigmaV = sum(y.^2.*hm);


if jump %for model 2, another terms related with jump
    VJ = Elambda*(mu_J^2+sig_J);
    k = [1:nhorizon-1]';
%     vareta = max(0.75*VJ(k)-0.25*hm(k),0);
%     vareta = (VJ(k)-0.5*min(VJ(k),hm(k)))/mu_J^2;
    
    if tvjump
        vareta = VJ(k)/mu_J^2;
        varlam = rho.^2.^(k-1).*vareta;
        Elambda2 = Elambda.^2+phi^2*cumsum([0;varlam]);
    else
        Elambda2 = Elambda.^2;
    end
        
    sigmaJ = sum(y.^2.*VJ); 
    skewt  = sum(Elambda.*y.^3)*mu_J*(mu_J^2+3*sig_J);
    
    %calculate fourth moment of residual
    kurt1  = 3*(Elambda2+Elambda)*(mu_J^2+sig_J)^2-2*Elambda*mu_J^4+6*hm.*Elambda*(mu_J^2+sig_J);
    h2 = zeros(nhorizon,1);
    for i=1:nhorizon
        if i==1
            h2(i) = hm(i)^2;
            kurt1(i) = kurt1(i)+3*h2(i);
        else
            h2(i)    = psim*(psim+2*etam)*h2(i-1)+etam^2*kurt1(i-1)+2*psim*etam*hm(i-1)*VJ(i-1)+2*kappam*hm(i)-kappam^2;
            kurt1(i) = kurt1(i)+3*h2(i);            
        end
    end            
    kurt   = sum(y.^4.*kurt1);
    h0 = kappam/(1-psim);
    
    for i=1:nhorizon    
        [skew0,skew1,kurt0,he2] = deal(zeros(nhorizon-i,1));
        for j=i+1:nhorizon 
            k = j-i;
            if k==1
                skew1(k) = Elambda(i)*mu_J*(mu_J^2+3*sig_J)*etam+phi*Elambda(i)*mu_J*(mu_J^2+sig_J);
            else
                skew1(k) = (etam+psim)*skew0(k-1)+rho^(k-2)*(rho-psim)*phi*Elambda(i)*mu_J*(mu_J^2+sig_J);
            end            
            B  = (etam+psim)^(k-1)*(etam*(mu_J^2+3*sig_J)+phi*(mu_J^2+sig_J))+...
                +(rho-psim)*phi*(mu_J^2+sig_J)*(rho^(k-1)-(etam+psim)^(k-1))/(rho-etam-psim);
            skew0(k) = B*Elambda(i)*mu_J;
%             skew0(k) = Elambda(i)*mu_J*(mu_J^2+3*sig_J)*etam*(psim+etam)^(j-i-1)+...
%                 Elambda(i)*mu_J*phi*rho^k*(mu_J^2+sig_J);
            skewt    = skewt+3*y(i)*y(j)^2*skew0(k);
            
            A = theta/(1-rho)*(1-rho^(j-i));
            tmp = A*(hm(i)+VJ(i))+rho^(j-i)*(hm(i)*Elambda(i)+(mu_J^2+sig_J)*Elambda2(i))+...
                rho^(j-i-1)*phi*Elambda(i)*(mu_J^2+sig_J);
            e22 = (mu_J^2+sig_J)*tmp;
            if k==1
                he2(k)   = kappam*(hm(i)+VJ(i))+psim*(h2(i)+VJ(i)*hm(i))+etam*kurt1(i);
                kurt0(k) = he2(k)+e22;
            else
                he2(k)   = kappam*(hm(i)+VJ(i))+psim*he2(k-1)+etam*kurt0(k-1);
                kurt0(k) = he2(k)+e22;
            end            
            kurt  = kurt+6*y(i)^2*y(j)^2*kurt0(k);
        end
    end
    
%     kurt1 = 0;
%     zz = phi^2*mu_J^2*(mu_J^2+sig_J);      
%     for i=1:nhorizon-2
%         for j=i+1:nhorizon-1
%             for k=j+1:nhorizon
%                 kurt1 = kurt1+12*y(i)*y(j)*y(k)^2*zz*Elambda(i)*rho.^(k-i-2);               
%             end
%         end
%     end
%     
%     kurt = 0;
    for i=1:nhorizon-2
        for j=i+1:nhorizon-1
            kurt3 = zeros(nhorizon-j,1);
            for k=j+1:nhorizon               
                kk = k-j;
                A  = (etam+psim)^(kk-1)*(etam*(mu_J^2+3*sig_J)+phi*(mu_J^2+sig_J))+...
                    +(rho-psim)*phi*(mu_J^2+sig_J)*(rho^(kk-1)-(etam+psim)^(kk-1))/(rho-etam-psim);
                kurt3(k-j) = A*rho.^(j-i-1)*mu_J^2*phi*Elambda(i);
                kurt = kurt+12*y(i)*y(j)*y(k)^2*kurt3(k-j);
            end
        end
    end
    
    zz = mu_J^2*(mu_J^2+3*sig_J)*phi;    
    for i=1:nhorizon-1
        kurt4 = zeros(nhorizon-i,1);
        for j=i+1:nhorizon
            kurt4(j-i) = zz*Elambda(i)*rho^(j-i-1);
            kurt = kurt+4*y(i)*y(j)^3*kurt4(j-i);
        end
    end
    
%     zz = phi^2*mu_J^2*(mu_J^2+sig_J)/(1-rho)^2;
%     t1 = [1:nhorizon-2]';
%     tmp = Elambda(t1).*(1-rho.^(nhorizon-1-t1)-(nhorizon-1-t1).*(1-rho).*rho.^(nhorizon-t1-1));
%     kurt = kurt+12*zz*sum(tmp);  
%     
%     zz = mu_J^2*(mu_J^2+3*sig_J)*phi/(1-rho);
%     t1 = [1:nhorizon-1]';
%     tmp = Elambda(t1).*(1-rho.^(nhorizon-t1));
%     kurt = kurt+4*zz*sum(tmp);
else
    sigmaJ = 0;
    skewt  = 0;
    
    
    kurt1 = zeros(nhorizon,1);
    for i=1:nhorizon
        if i==1            
            kurt1(i) = 3*hm(i)^2;
        else
            kurt1(i) = (psim^2+3*etam^2+2*psim*etam)*kurt1(i-1)+6*kappam*hm(i)-3*kappam^2;
%             kurt1(i) = (psim^2+3*etam^2)*kurt1(i-1)+6*kappam*hm(i)-3*kappam^2+...
%                 6*psim*etam*hm(i-1)^2;
        end
    end            
    kurt   = sum(y.^4.*kurt1);
    h0 = kappam/(1-psim);
    
    for i=1:nhorizon    
        kurt0 = deal(zeros(nhorizon-i,1));
        for j=i+1:nhorizon 
            k = j-i;       
            if k==1
                kurt0(k) = (psim/3+etam)*kurt1(i)+kappam*hm(i);
            else
                kurt0(k) = (psim+etam)*kurt0(k-1)+kappam*hm(i);
            end
            kurt  = kurt+6*y(i)^2*y(j)^2*kurt0(k);            
        end
    end
end

mm = zeros(4,1);
mm(1) = emret;
mm(2) = sigmaV+sigmaJ;
mm(3) = skewt/mm(2)^(1.5);
mm(4) = kurt/mm(2)^2-3;