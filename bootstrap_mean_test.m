function p_value = bootstrap_mean_test(data, mu0, horizon)
% Bootstrap均值检验（MATLAB实现）
% 输入：
%   data - 样本数据向量
%   mu0 - 假设的总体均值
%   B - Bootstrap次数（默认10000）
%   alpha - 显著性水平（默认0.05）
% 输出：
%   p_value - 计算得到的p值


B = 10000;

n = length(data);
x_bar = mean(data);

% 数据调整：使均值等于mu0（满足H0）
adjusted_data = data - x_bar + mu0;

% 预分配存储空间
boot_means = zeros(B, 1);

% Bootstrap抽样
rng('default'); % 设置随机种子保证可重复性


idx = block_bootstrap((1:n)',B,horizon);

for m = 1:B
    % 有放回抽样
    boot_sample = adjusted_data(idx(:,m));
    boot_means(m) = mean(boot_sample);
end

% 计算p值（双侧检验）
original_deviation = abs(x_bar - mu0);
boot_deviations = abs(boot_means - mu0);
p_value = mean(boot_deviations >= original_deviation);
