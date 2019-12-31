function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%

%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = [ones(1, size(a2,2)); a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);
h = a3';

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end


J = (1/m)*sum(sum(-Y.*log(h) - (1-Y).*log(1-h)));

reg_theta1 = Theta1;
reg_theta1(:,1)=0;

reg_theta2 = Theta2;
reg_theta2(:,1)=0;

penalty = (lambda/(2*m))*(sum(sum(reg_theta1.^2)) + sum(sum(reg_theta2.^2)));

J = J + penalty;


z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = [ones(1, size(a2,2)); a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);

delta_3 = a3-Y';
delta_2 = (Theta2(:,2:end)'*delta_3).*sigmoidGradient(z2);

DELTA_2 = delta_3*a2';
DELTA_1 = delta_2*X;

Theta1_grad = (1/m)*DELTA_1 + (lambda/m)*reg_theta1;
Theta2_grad = (1/m)*DELTA_2 + (lambda/m)*reg_theta2;


DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
