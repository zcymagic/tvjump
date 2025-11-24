
addpath("johnson\")
clear

load resus;

N = 1e5;


p =  [0.01 0.05]';
r1s = 0;
h1s = 1.6;
lambdas = [1/20 0.4 1]; 
maxh = 30;

Zs = cell(3,1);
Zs{1} = h1s;
Zs{3} = r1s;
[varp,varJ] = deal(zeros(length(p),maxh,length(lambdas)));


for i=1:length(lambdas)
    Zs{2} = lambdas(i);
    rm = simulateret_new(par0{3},N,maxh,Zs);     
    
    for hh=1:maxh
        data = sort(sum(rm(:,1:hh),2));                
        varp(:,hh,i) = -data(round(p*N));
        mmp = getpredictnew(par0{3},specs{3},r1s,h1s,lambdas(i),hh);
        varJ(:,hh,i) = -johnsonvar(mmp,p);          
    end
end

res1 = (varJ-varp)./(varp);
lenp = length(p);

for i=1:lenp
    subplot(lenp,1,i);
    plot(1:maxh,res1(i,:,1),'-',1:maxh,res1(i,:,2),'--',1:maxh,res1(i,:,3),'-.');
    title(sprintf('q=%.2g%%',p(i)*100))

    pos = get(gca, 'Position');
    new_height = pos(4)*0.85;  % 减少高度
    new_y = pos(2) + (pos(4) - new_height);  % 上移子图
    set(gca, 'Position', [pos(1) new_y pos(3) new_height])   

    if i==lenp
        xlabel('T(days)');
    end

end
legend('\lambda_0=0.05','\lambda_0=0.4','\lambda_0=1','Orientation', 'horizontal','Position',  [0.3 0.04 0.4 0.05]);

  
    



