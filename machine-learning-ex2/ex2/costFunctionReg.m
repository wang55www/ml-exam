function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


for i=1:m
  J = J + (-y(i) * log(sigmoid(X*theta)(i))-(1-y(i)) * log(1-sigmoid(X*theta)(i)));
end;
J = J / m + (lambda/(2*m))*sum(theta(2:size(X,2)).^2); ;

sum1 =zeros(size(X,2),1);

for i=1:m
    sum1 = sum1+(sigmoid(X*theta)(i)-y(i)).*X(i,:)';
end;

grad(1)= (1/m)*sum1(1);
grad(2:size(X,2))= (1/m)*sum1(2:size(X,2)) + (lambda/m).*theta(2:size(X,2));


% =============================================================

end
