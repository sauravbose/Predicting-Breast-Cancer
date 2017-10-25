function [error] =  dt_xval_error(X, Y,d_lim,part)

N = max(part);
error_vect = [];

for i = 1:N
    test_rows = find(part == i);
    X_test = X(test_rows,:);
    Y_test = Y(test_rows);
    Y_test = double(Y_test);
         %Y_test = Y_test';
    train_rows = find(part~=i);
    X_train = X(train_rows,:);
    Y_train = Y(train_rows);
    
    p = [];
    root = dt_train(X_train,Y_train,d_lim);
    for j = 1:length(X_test)
        p(j) = dt_value(root, X_test(j,:));
    end
    p = p';
    labels = p;
    labels(labels>=0.5)=1;
    labels(labels<0.5)=0;
    
    y_diff = labels - Y_test;
    y_err = y_diff(y_diff~=0);

    err = length(y_err)/length(y_diff);
    error_vect = cat(2,error_vect,err);

end
     error = mean(error_vect);
    
  
end


