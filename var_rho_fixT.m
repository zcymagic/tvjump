clear
addpath("johnson\")
load resus;

N = 1e5;
r1s = 0;
h1s = 1.6;
lambdas = [0.4 1];

maxh = 100;
hs =  (1:maxh);

horizon = [1 5 10];
pp  = 0.01;
rho_J = [0.8 0.95]';


varj = zeros(length(horizon),length(hs),length(lambdas),length(rho_J));
Zs = cell(3,1);
Zs{1} = h1s;
Zs{3} = r1s;   

for i=1:length(lambdas)
    Zs{2} = lambdas(i);
    for j=1:length(rho_J)        
        partmp = par0{3};
        partmp(7)=rho_J(j);        

        [rm,ht,lambdat] = simulateret_new(partmp,N,maxh,Zs);         
        
        for h2=1:maxh  %5-day VaR
            for hh=1:length(horizon)
                if h2==1                    
                    mmp = getpredictnew(partmp,specs{3},r1s,h1s,lambdas(i),horizon(hh));
                    varj(hh,h2,i,j) = johnsonvar(mmp,pp);
                else                   
                    tmpvarj = zeros(N,1);
                    for ii=1:N
                        mmp = getpredictnew(partmp,specs{3},rm(ii,h2),ht(ii,h2),lambdat(ii,h2),horizon(hh));                       
                        tmpvarj(ii) = johnsonvar(mmp,pp);
                    end
                    varj(hh,h2,i,j) = mean(tmpvarj);
                end
            end
        end
    end
end


for id1=1:length(rho_J)
    hf=figure;
    subplot(3,1,1)
    hh=1;
    plot(hs,-varj(hh,:,1,id1),'-',hs,-varj(hh,:,2,id1),'--');
    ylim([1,5])
    legend('\lambda_0=0.4','\lambda_0=1','Location','Northeast','Orientation','Horizontal');
    title('VaR of 1-day returns at 1% level')

    subplot(3,1,2)
    hh=2;
    plot(hs,-varj(hh,:,1,id1),'-',hs,-varj(hh,:,2,id1),'--');
    ylim([3,11]);
    legend('\lambda_0=0.4','\lambda_0=1','Location','Northeast','Orientation','Horizontal');
    title('VaR of 5-day returns at 1% level')

    subplot(3,1,3)
    hh=3;
    plot(hs,-varj(hh,:,1,id1),'-',hs,-varj(hh,:,2,id1),'--');
    legend('\lambda_0=0.4','\lambda_0=1','Location','Northeast','Orientation','Horizontal');
    xlabel('Day');
    ylim([5,15]);
    title('VaR of 10-day returns at 1% level')
    tname = sprintf('rhovar_fixT_%d.eps',mod(id1,3));
    saveas(hf,tname,'epsc');
end


