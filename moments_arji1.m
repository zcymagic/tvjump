addpath("johnson\")

clear

load resus;

N = 1e5;
r1s = 0;
h1s = 1.6;


lambdas = [1/20 0.4 1]; 
pp  = [0.01 0.05]';

maxh=1000;
hs = (1:maxh);

mmp = zeros(4,length(hs),length(lambdas));
[varsim,varsim_normal] = deal(zeros(length(pp),length(hs),length(lambdas)));
Zs = cell(3,1);
Zs{1} = h1s;
Zs{3} = r1s;



for i=1:length(lambdas)

    Zs{2} = lambdas(i);
    rm = simulateret_new(par0{3},N,maxh,Zs); 
    for hh=1:maxh               
        data = sum(rm(:,1:hh),2);                
        varsim(:,hh,i) = -quantile(data,pp);
        mmp(:,hh,i) = [mean(data); var(data); skewness(data); kurtosis(data)-3];          
        varsim_normal(:,hh,i) = -(mmp(1,hh,i)+sqrt(mmp(2,hh,i))*norminv(pp));        
    end           
end

ratio = (varsim-varsim_normal)./varsim;

hh=figure;
subplot(3,1,1)
plot(hs,mmp(3,:,1),'-',hs,mmp(3,:,2),'--',hs,mmp(3,:,3),'-.');
% xlabel('Days');
legend('\lambda_0=0.05','\lambda_0=0.4','\lambda_0=1','Location','Northeast','Orientation','Horizontal');
title('Skewness of T-day returns')

subplot(3,1,2)
plot(hs,mmp(4,:,1),'-',hs,mmp(4,:,2),'--',hs,mmp(4,:,3),'-.');
% xlabel('Days');
ylim([0,5]);
legend('\lambda_0=0.05','\lambda_0=0.4','\lambda_0=1','Location','Northeast','Orientation','Horizontal');
title('Excess kurtosis of T-day returns')


subplot(3,1,3)
plot(hs,ratio(1,:,1),'-',hs, ratio(1,:,2),'--',hs, ratio(1,:,3),'-.');
xlabel('Days');
% ylim([0,1]);
legend('\lambda_0=0.05','\lambda_0=0.4','\lambda_0=1','Location','Southeast','Orientation','Horizontal');
title('Contribution of higher-order moments to the total VaR')


saveas(hh,'skvar.eps','epsc');


    




