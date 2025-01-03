% SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
% SPDX-License-Identifier: MIT

close all;
clear all;

d = load("bench_data.mat", "z", "alpha", "beta");
z = d.z;
alpha = d.alpha;
beta = d.beta;
gamma = 1.0;

n = length(alpha);
nrepeats = 16;
result = zeros(n, 3);
times = zeros(nrepeats, 1);

for i = 1:n
    for j = 1:nrepeats
        tic();
        arrayfun(@(zk) ml(zk, alpha(i), beta, gamma), z);
        times(j) = toc();
    end

    result(i, 1) = min(times);
    result(i, 2) = mean(times);
    result(i, 3) = std(times);
end

save('bench_result.mat', 'alpha', 'result');
