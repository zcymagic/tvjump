function [rt,ht,lambdat]=simulateret_new(par,Npath,horizon,Z0)

njumps = 10;


alpham = par(1);
betam  = par(2);
kappam = par(3); %K
etam   = par(4); %GARCH
psim   = par(5); %ARCH

ind  = 5;
[rt,ht,lambdat] = deal(zeros(Npath,horizon));  %residual terms=diffusion+jump
ht(:,1) = Z0{1};


RandStream.setGlobalStream(RandStream('mt19937ar','seed',1));


lambdat(:,1) = Z0{2}*ones(Npath,1);


z = randn(Npath/2,1);
innovs = sqrt(ht(:,1)).*[z;-z];  

% jump component
lambda0 = par(ind+1);
rho     = par(ind+2);

ksi     = par(ind+3);
mu_J    = par(ind+4);
sig_J   = par(ind+5);
sig_J2  = sig_J*sig_J ;

JN = poissrnd(lambdat(:,1)); 
I = find(JN>0);
for k=1:length(I)
    j = I(k);
    innovs(j) = innovs(j)+sum(normrnd(mu_J,sig_J,JN(j),1));
end   
innovs = innovs-lambdat(:,1)*mu_J;
rt(:,1) = Z0{3}+innovs;   


for t=2:horizon
    ht0=ht(:,t-1);
    innov0=innovs;
    
    ht(:,t) = kappam+etam*innovs.^2+psim*ht(:,t-1);
    z = randn(Npath/2,1);
    innovs = sqrt(ht(:,t)).*[z;-z];
    
    %jump component
    prob   = zeros(Npath,njumps+1);
    probs = 0;
    u = innov0+lambdat(:,t-1)*mu_J;
    for nc=0:njumps                
        Hs = ht0+nc*sig_J2;
        prob(:,nc+1) = normpdf(u,nc*mu_J,sqrt(Hs)).*poisspdf(nc,lambdat(:,t-1));
        probs = probs+prob(:,nc+1);
    end
    filt = matdiv(prob,probs);
    ksit = sum(matmul(filt,(0:njumps)),2)-lambdat(:,t-1);
    lambdat(:,t) = lambda0+rho*lambdat(:,t-1)+ksi*ksit;  
    
    JN = poissrnd(lambdat(:,t)); 
    I = find(JN>0);
    for k=1:length(I)
        j = I(k);
        innovs(j) = innovs(j)+sum(normrnd(mu_J,sig_J,JN(j),1));
    end   
    innovs = innovs-lambdat(:,t)*mu_J;    
    rt(:,t) = alpham+betam*rt(:,t-1)+innovs;
   
end
    