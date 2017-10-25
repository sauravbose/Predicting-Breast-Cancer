function [pred_labels] = SVM_train(test_data, kerneltype)
    % INPUT : 
    % test_data   - m X n matrix, where m is the number of test points and n is number of features
    % kerneltype  - one of strings 'poly', 'rbf'
    %               corresponding to polynomial, and RBF kernels
    %               respectively.
    
    % OUTPUT
    % returns a m X 1 vector predicted labels for each of the test points. The labels should be +1/-1 doubles

    
    % Default code below. Fill in your code on all the relevant positions

    m = size(test_data , 1);
    n = size(test_data, 2);

    %load train_data

    datadir = 'Breast-Cancer/';

    load(strcat(datadir,'train.mat'));


    %load cross-validation data

    %your code




    % Do cross-validation
    % For all c
    % For all kernel parameters
    % Calculate the average cross-validation error for the 5-folds

   if(strcmp(kerneltype,'poly'))
        options = '-t 1 -g 1 -r 1'
        options_init = options
    elseif(strcmp(kerneltype,'rbf'))
        options = '-t 2'
        options_init = options

   end

    c = [1, 10, 10^2 ,10^3, 10^4, 10^5];
    q = [1,2,3,4,5];
    sig = [0.01, 1, 10, 10^2, 10^3];

    %Train SVM on training data for the best parameters

    if(strcmp(kerneltype,'poly'))
        c_final = [];
        cv_err = [];
        for deg = 1:length(q)
            options = options_init;
            options = [options,' -d ',num2str(q(deg))];
            options_temp = options;
            error= [];
            for j = 1:length(c)
                options = options_temp;
                options = [options,' -c ',num2str(c(j))];
                err_cv = [];
                for i = 1:5           

                    datadircv = 'Breast-Cancer/CrossValidation/Fold';
                    datadircv = strcat(datadircv,num2str(i),"/");
                    load(strcat(datadircv,'cv-train.mat'));
                    load(strcat(datadircv,'cv-test.mat'));

                    X_train = cv_train(:,1:9);
                    Y_train = cv_train(:,10);

                    X_test = cv_test(:,1:9);
                    Y_test = cv_test(:,10);

                    model = svmtrain(Y_train, X_train, options);
                    predicted_label = svmpredict(Y_test, X_test, model);

                    err_cv(i) = classification_error(predicted_label, Y_test);


                end
                err = mean(err_cv);

                error(j) = err;
            end
        min_err = min(error);

        c_min_id = find(error == min_err);
        c_min_id = c_min_id(1);
        c_min = c(c_min_id);

        c_final(deg) = c_min;
        cv_err(deg) = min_err;
        
        end

      degree_opt_id = find(cv_err == min(cv_err));
      degree_opt = q(degree_opt_id);
      C_opt = c_final(degree_opt_id);
      
      Xtrain = train(:,1:9);
      Ytrain = train(:,10);
      
      Xtest = test_data(:,1:n);
      %Ytest = test_data(:,n);
      Ytest = ones(m,1);
      
      options = options_init;
      options = [options,' -d ',num2str(degree_opt)];
      options = [options,' -c ',num2str(C_opt)];
        
        
      model = svmtrain(Ytrain, Xtrain, options);
      pred_labels = svmpredict(Ytest, Xtest, model);
    
    elseif(strcmp(kerneltype,'rbf'))
        
        c_final = [];
        cv_err = [];
        for deg = 1:length(sig)
            options = options_init;
            options = [options,' -g ',num2str(sig(deg))];
            options_temp = options;
            error= [];
            for j = 1:length(c)
                options = options_temp;
                options = [options,' -c ',num2str(c(j))];
                err_cv = [];
                for i = 1:5           

                    datadircv = 'Breast-Cancer/CrossValidation/Fold';
                    datadircv = strcat(datadircv,num2str(i),"/");
                    load(strcat(datadircv,'cv-train.mat'));
                    load(strcat(datadircv,'cv-test.mat'));

                    X_train = cv_train(:,1:9);
                    Y_train = cv_train(:,10);

                    X_test = cv_test(:,1:9);
                    Y_test = cv_test(:,10);

                    model = svmtrain(Y_train, X_train, options);
                    predicted_label = svmpredict(Y_test, X_test, model);

                    err_cv(i) = classification_error(predicted_label, Y_test);


                end
                err = mean(err_cv);

                error(j) = err;
            end
        min_err = min(error);

        c_min_id = find(error == min_err);
        c_min_id = c_min_id(1);
        c_min = c(c_min_id);

        c_final(deg) = c_min;
        cv_err(deg) = min_err;
           

        end
      sigma_opt_id = find(cv_err == min(cv_err));
      sigma_opt = sig(sigma_opt_id);
      C_opt = c_final(sigma_opt_id);
      
      Xtrain = train(:,1:9);
      Ytrain = train(:,10);
      
      Xtest = test_data(:,1:n);
      %Ytest = test_data(:,n);
      Ytest = ones(m,1);
      
      options = options_init;
      options = [options,' -g ',num2str(sigma_opt)];
      options = [options,' -c ',num2str(C_opt)];
        
      model = svmtrain(Ytrain, Xtrain, options);
      pred_labels = svmpredict(Ytest, Xtest, model);
        
    end


    % Do prediction on the test data
    % pred_labels = your prediction on the test data
    % your code














end
