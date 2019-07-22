clear ; close all; clc

% 加载数据
data = csvread('myData.txt');


X_original = data(:, 1);
y_original = data(:, 2);

y_test = csvread('myTest.txt');

X = zeros(15,1);
y = zeros(15,1);
for i = 0:14
    X(i+1) = i+5;
    y(i+1) = sum( y_original(i*7+1:i*7+7) ) / 7;
end;



% % 70%作为训练集，30%作为交叉验证
sel = randperm(size(X, 1));


% 集合个数
m = round(0.7 * size(X,1));
m_cv = size(X,1) - m;

X_data = X;
y_data = y;

% 选择训练集和交叉验证集
X = X_data(sel(1:m),:);
y = y_data(sel(1:m),:);

Xval = X_data(sel(m+1:end),:);
yval = y_data(sel(m+1:end),:);

% 多项式回归
p = 200;

% X变为多项式特征和标准化
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Xval变为多项式特征和标准化
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones


% 多项式回归学习曲线
lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);


% Plot training data and fit
figure(1);
plot(X, y, 'r.', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('time');
ylabel('flow');
% title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
axis([6 22 -10 40])
print /Users/alfred/Desktop/p1.bmp -dbmp
figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 m 0 200])
legend('Train', 'Cross Validation')
print /Users/alfred/Desktop/p2.bmp -dbmp

fprintf('Program paused. Press enter to continue.\n');
pause;



[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

figure();
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
axis([0 300 0 100])
fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;