function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

first_fill_cap = floor(n/n_folds);
rem_first_fill = mod(n,n_folds);

for i = 1:n_folds
    cap_vect(i) = first_fill_cap
end

r = randi([1,n_folds])
cap_vect(r) = cap_vect(r) + rem_first_fill


id = 1
for i = 1:length(cap_vect)
    count = cap_vect(i)
    for j = 1:count
        bins(id) = i
        id = id+1
    end
end

part = bins(randperm(length(bins)))