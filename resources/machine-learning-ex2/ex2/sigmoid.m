function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
% fprintf('%d ',size(z));
% fprintf('\n');
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


% z也可以是矩阵，此处除法应使用 点除
g = 1 ./ ( 1 + exp(-z) );


% =============================================================

end
