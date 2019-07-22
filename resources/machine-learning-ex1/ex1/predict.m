%% Initialization
clear ; close all; clc

% 加载数据
data = csvread('myData.txt');

X = data(:, 1);
y = data(:, 2);

m = size(X,1);
X = X./24;
X = [X X.^2 X.^3];
X = [ones(m, 1) X];

theta = normalEqn(X, y);

fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);


people = zeros(24,1);
for i = 1 : 24
    people(i) = [1,(i-1)/24,(((i-1)/24)^2),((i-1)/24)^3] * theta;
end;
people

plot([0:23], people, '-');