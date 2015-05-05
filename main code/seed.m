% Compute Self-Expressive Decomposition (SEED) of X as Xout = D*V
% INPUT 
% X = data matrix with examples in columns of X (m x N) 
% opts = struct with options for seed
    % options for opts struct
    % numselect = number of initial columns to select for oasis
    % epsilon = target error for OMP (0 to 1)
    % kmax = target sparsity for OMP (1,2,...,L)
    % L = number of columns to select (if L is specified, then run oasis)
% OUPUT
% D = left factor matrix (m x L)
% V = right factor matrix (L x N)

% Example Usage:
% opts.kmax=10; 
% opts.epsilon = 0.05;
% [D,V] = seed(X,opts);

function [D,V] = seed(X,L,opts)

if nargin<3
    opts = setdefaultparam();
end

if isfield(opts,'kmax')
    kmax = opts.kmax;
else
    kmax = 10;
end

if isfield(opts,'epsilon')
    epsilon = opts.epsilon;
else
    epsilon = 0.05;
end

if ~isfield(opts,'numselect')
    opts.numselect = 10;
end

if isfield(opts,'ompmethod')
    ompmethod = opts.ompmethod;
else
    ompmethod = 'batch';
end

%%%%% Initialize algorithm %%%%% 
X = normcol(X); % normalize data

%%%%% Step 1. Form D %%%%%
if length(L)>1
    idxset = L; % pass in columns to sample (bypass oasis sampling) 
else
    G = X'*X;
    [ outs ] = nystrom(G,L,'p',opts);
    idxset = outs.selection;
end

D = normcol(X(:,idxset));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%% Step 2. Batch OMP to compute V, where Xhat = D*V %%%%%
Xtst = X;
DtX = D'*Xtst;
XtX = sum(Xtst.*Xtst);
G = D'*D;

if strcmp(ompmethod,'batch')
    V = omp2(DtX,XtX,G,epsilon,'maxatoms',kmax);
else
    V = OMP(X,D,kmax,epsilon);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end % end main function

function opts = setdefaultparam()
opts.kmax = 10;
opts.epsilon = 0.05;
opts.numselect = 10;
opts.ompmethod = 'batch';
    
end