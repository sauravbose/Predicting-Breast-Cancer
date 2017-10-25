function [weights,error_per_iter] = gradient_ascent_decay(Xtrain,Ytrain,initial_step_size,iterations)

    % Function to perform gradient descent with a decaying step size for
    % logistic regression.
    % Usage: [weights,error_per_iter] = gradient_descent(Xtrain,Ytrain,step_size,iterations)
    
    % The parameters to this function are exactly the same as the
    % parameters to gradient descent with fixed step size.
    
    % initial_step_size : This parameter refers to the initial value of the step
    % size. The actual step size to update the weights will be a value
    % that is (initial_step_size * some function that decays over time)
    % some good choices for this function might by 1/n or 1/sqrt(n).
    % Experiment with such functions, and initial step size until you get
    % good performance.
    
    initial_step_size = 0.004;
    weights = ones(size(Xtrain,2),1); % P x 1 vector of initial weights
    error_per_iter = zeros(iterations,1); % error_per_iter(i) records training error in iteration i of GD.
    % dont forget to update these values within the loop!
    Ytrain = double(Ytrain);
    %Ytrain(Ytrain == 0) = -1;
    
    n = size(Xtrain,1)
    step_size = initial_step_size;
    
    for iter = [1:iterations]
        Ytrain(Ytrain == 0) = -1;
        grad = sum(Xtrain.*Ytrain.*exp(Xtrain*weights.*Ytrain.*(-1))./(1+exp(Xtrain*weights.*Ytrain.*(-1))),1);
        grad = grad';
        weights = weights+grad*step_size;
                
        y_pred = 1./(1+exp(-Xtrain*weights));
        y_pred(y_pred>=0.5)=1;
        y_pred(y_pred<0.5)=0;
        Ytrain(Ytrain == -1) = 0;
        error_per_iter(iter) = sum(abs(y_pred - Ytrain))/length(Ytrain);
        
        step_size = step_size*(1/sqrt(iter));

    end
%  plot(1:500, error_per_iter);
%  
%  
% hold on;
% [~,err_fixed] = gradient_ascent_fixed(Xtrain,Ytrain,0.0001,500) 
% plot(1:500,err_fixed);
% 
% title('Error evolution : Original Data');
% xlabel('Iteration');
% ylabel('Error');
% legend('Decaying step size','Fixed step size');
% hold off;

end

