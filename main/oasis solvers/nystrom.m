%  Build the Nystrom approximation of a symmetric matrix X by randomly
%  sampling columns.  The number of columns sampled is L.
function [ outs ] = nystrom( X, vecOfSampleSizes, method, opts )

 %% Check preconditions, fill missing optional entries in 'opts'
if ~exist('opts','var') % if user didn't pass this arg, then create it
 opts = [];
end
opts = fillDefaultOptions(opts);
L = max(vecOfSampleSizes); %  maximum number of cols to sample
if isnumeric(X) % if we are given a full matrix
    [rows,cols] = size(X);
    assert(rows==cols,'Input X must be square');
    assert(L<=cols,'Cannot have more samples than columns');
    N = cols;
else
    assert(strcmp(method,'p') | strcmp(method,'product') |strcmp(method,'r') | strcmp(method,'random'), 'Method must be product or random when function handle is used instead of full matrix.' )
end

if opts.verbose
    fprintf('Nystrom: mode = %s, L = %s\n',method,num2str(vecOfSampleSizes));
end

%%  to hold outputs
outs = [];

%%  Sample columns using the method chosen by the user
C = [];  % some sampling methods will overwrite this
if isempty(opts.selection) %  don't call sampler if columns are specified in opts
    switch lower(method)
        case {'r','random'}
            [selection, C]= randomSample(X,L,opts.verbose); %Grab columns at random
        case {'p','product'}
            [selection, C] = innerProductSample_fast(X,L,opts.numselect,opts.verbose);
        case {'n','norm'}
            [selection] = innerProductSample_norm(X,L,opts.numselect,opts.verbose);
        case {'pb','productbatch'} % eva changed this to 'pb' (was 'p')
            selection = innerProductSample_batch(X,L,opts.numselect,opts.numselect);
        case {'lev','leverage'} % added leverage score sampling
            %% check for leverage scores
            if isempty(opts.scores)
                outs.scores = leveragescores(X,opts.verbose,L);
            else
                outs.scores = opts.scores;
            end
            selection = leverageSample(outs.scores,L,opts.verbose);
        case {'f','farahat'} % added Farahat (2011) method for comparison
            selection = farahat(X,L, opts.verbose);
        otherwise
            error(['Invalid method: ' method]);
    end
else
    selection = opts.selection;  %  If columns were specified by user, then use there
end

assert(L<=numel(selection),'L is too big.  Not that many columns have been selected.');

%%  Rip out the cols
if isempty(C)
    if isnumeric(X)
        C = X(:,selection);
    else
        N = length(X('D','D'));  %  Problem size
        C = zeros(N,length(selection));
        for i=1:length(selection)
            C(:,i) = X([],selection(i));
        end
    end
end


%% Slice up the matrix
%%  Note:  We do not permute the rows of C.  This is so that the matrix we get back is an approximation of X, not of the permuated X
W = C(selection,:);       % Block 1-1 (Upper left) of permuted X

%%  Record Results

outs.C = C;
outs.W = W;
outs.selection = selection;


tic;
[Uw,Sw,Vw] = svd(W);
Splus = zeros(size(Sw));
Splus(Sw>1e-7) = Sw(Sw>1e-7).^-1; % invert singular values of W
outs.Wplus = Uw*Splus*Vw';   % The pseudo-inverse of W
if opts.verbose; fprintf('\tfactor: time = %d sec\n',toc); end;

%% Everything below this line only gets called when "opts" is set properly

%% Expand Nystrom Approximation
if opts.expandNystrom
      outs.K = C*outs.Wplus*C';
end


%%  Compute psuedo-inverse of W
%Do this explicitly (rather than calling 'pinv') and store SVD
if opts.computeApproxSVD
    % Compute approximate singular values of X
    outs.S = (N/L)*diag(Sw);
    % Compute approximate singular vectors of X
    outs.U = sqrt(N/L)*(C*Uw*Splus);
end


%%  If more than one sample size was chosen, compute them all
if length(vecOfSampleSizes)>1
    outs.nystroms = {};
    for Lsmall = vecOfSampleSizes
       thisNystrom = [];
       thisNystrom.C = C(:,1:Lsmall);
       thisNystrom.W = thisNystrom.C( selection(1:Lsmall),:); 
       thisNystrom.L = Lsmall;
       if opts.expandNystrom
           thisNystrom.K = thisNystrom.C*(thisNystrom.W\thisNystrom.C');
       end
       outs.nystroms(end+1) = {thisNystrom};
    end
end


return


%%  product sample columns.  USE C/MEX INTERFACE
function [indices, cols ] = innerProductSample_fast(X,L, startSize, verbose)
%rand('seed',0);
if verbose; fprintf('\tBegin Product\n'); tic; end
if isnumeric(X)
    if ~exist('asis')
        fprintf('Compiling asis executable.  You may get an error if you have not setup mex.');
        mex asis.c;
    end
    indices = asis(X,L, double(verbose));
    indices = indices+1; %  Convert from C 0-indexed array to matlab 1-indexing
    cols = [];
else
     [indices, cols ] = innerProductSample_dense(X,L, startSize, verbose);
end
if verbose; fprintf('\n\tEnd Product: time = %f secs\n',toc); end
return

%%  product sample columns
function [indices, cols ] = innerProductSample(X,L, startSize, verbose)
rand('seed',0);
if verbose; fprintf('\tBegin Product'); tic; end
if isnumeric(X)
    [indices, cols ] = innerProductSample_dense(X,L, startSize, verbose);
    return;
    %X = @(r,c) denseMatrixSampler(X,r,c);
end
%% Get the diagonal
D = X('D','D')';
N = length(D);  %  Problem size
%% Randomly choose some starting columns
startSize = min(startSize,L);
randomColPerm = randperm(N); 
indices = randomColPerm(1:startSize);  %  the indices of selected columns

%% Sample starting cols
cols = zeros(N,startSize);
for i=1:startSize
    cols(:,i) = X([],indices(i));
end

%% Compute representation (rep) of columns in the basis W
W = cols(indices,:); 
rep = W\cols';
error = sum(rep.*cols')-D; 
[m,newColIndex] = max(abs(error));
%%
while size(rep,1)<L
    %% Correct the matrix rep = W^{-1}*cols' to add columns we've already selected
    newCol = X([],newColIndex);
    b = newCol(indices);
    d = newCol(newColIndex);
    Ainvb = rep(:,newColIndex);
    shur = (d-b'*Ainvb)^-1;
    %% Use block matrix inverse formula to add row to W^{-1}*cols
    brep = b'*rep;
    rep = [rep+Ainvb*shur*(brep-newCol') ;...
         shur*(-brep+newCol')    ];
    %%  Update record to include what we just added %% C starts here
    cols = [cols newCol]; 
    indices = [indices newColIndex];
    %%  Select new column to be used on next iteration
    error = sum(rep.*cols')-D; 
    %%  Grab out the row with max norm
    [m,newColIndex] = max(abs(error)); 
    if verbose;
        fprintf('.'); 
        if mod(size(rep,1),50)==1
            fprintf('\n\t\t\t');
        end
    end
end
indices = indices(1:L);
if verbose; fprintf('\n\tEnd Product: time = %f secs\n',toc); end
return



%%  product sample columns
function [indices ] = innerProductSample_norm(X,L, startSize, verbose)
rand('seed',0);
if verbose; fprintf('\tBegin Product(normalized)'); tic; end
if isnumeric(X)
    %[indices, cols ] = innerProductSample_dense(X,L, startSize, verbose);
    %return;
    X = @(r,c) denseMatrixSampler(X,r,c);
end
%% Get the diagonal
D = X('D','D')';
N = length(D);  %  Problem size
%% Randomly choose some starting columns
startSize = min(startSize,L);
randomColPerm = randperm(N); 
indices = randomColPerm(1:startSize);  %  the indices of selected columns

%% Sample starting cols
cols = zeros(N,startSize);
for i=1:startSize
    cols(:,i) = X([],indices(i));
    cols(:,i) = cols(:,i)/norm(cols(:,i));
end

%% Compute representation (rep) of columns in the basis W
W = cols(indices,:); 
rep = W\cols';
error = sum(rep.*cols')-D; 
[m,newColIndex] = max(abs(error));
%%
while size(rep,1)<L
    %% Correct the matrix rep = W^{-1}*cols' to add columns we've already selected
    newCol = X([],newColIndex);
    newCol = newCol/norm(newCol);
    b = newCol(indices);
    d = newCol(newColIndex);
    Ainvb = rep(:,newColIndex);
    shur = (d-b'*Ainvb)^-1;
    %% Use block matrix inverse formula to add row to W^{-1}*cols
    brep = b'*rep;
    rep = [rep+Ainvb*shur*(brep-newCol') ;...
         shur*(-brep+newCol')    ];
    %%  Update record to include what we just added 
    cols = [cols newCol]; 
    indices = [indices newColIndex];
    %%  Select new column to be used on next iteration
    error = sum(rep.*cols')-D; 
    %%  Grab out the row with max norm
    [m,newColIndex] = max(abs(error)); 
    if verbose;
        fprintf('.'); 
        if mod(size(rep,1),50)==1
            fprintf('\n\t\t\t');
        end
    end
end
indices = indices(1:L);
if verbose; fprintf('\n\tEnd Product: time = %f secs\n',toc); end
return


%%  product sample columns
function [indices, cols ] = innerProductSample_dense(X,L, startSize, verbose)
rand('seed',0);
%% Get the diagonal
D = diag(X)';
N = length(D);  %  Problem size
%% Randomly choose some starting columns
randomColPerm = randperm(N); 
startSize = min(startSize,L);
indices = randomColPerm(1:startSize);  %  the indices of selected columns
cols = X(:,indices);           %  the selected columns themselves
W = cols(indices,:); 
rep = W\cols';
error = sum(rep.*cols')-D; 
[m,newColIndex] = max(abs(error));
while size(rep,1)<L
    %% Correct the matrix rep = W^{-1}*cols' to add columns we've already selected
    newCol = X(:,newColIndex);
    b = newCol(indices);
    d = newCol(newColIndex);
    Ainvb = rep(:,newColIndex);
    shur = (d-b'*Ainvb)^-1;
    %% Use block matrix inverse formula to add row to W^{-1}*cols
    brep = b'*rep;
    rep = [rep+Ainvb*shur*(brep-newCol') ;...
         shur*(-brep+newCol')    ];
    %%  Update record to include what we just added 
    cols = [cols newCol]; 
    indices = [indices newColIndex];
    %%  Select new column to be used on next iteration
    error = sum(rep.*cols')-D; 
    %%  Grab out the row with max norm
    [m,newColIndex] = max(abs(error)); 
    if verbose;
        fprintf('.'); 
        if mod(size(rep,1),50)==1
            fprintf('\n\t\t\t');
        end
    end
end
indices = indices(1:L);
if verbose; fprintf('\n\tEnd Product: time = %f secs\n',toc); end
return

%%  inverse sample columns
function indices = innerProductSample_batch(X,L, startSize, batch)
%rand('seed',0);
N = size(X,2);  %  Problem size
%% Randomly choose some starting columns
randomColPerm = randperm(N); 
indices = randomColPerm(1:startSize);  %  the indices of selected columns
cols = X(indices,:);           %  the selected columns themselves
W = cols(:,indices); 
rep = W\cols;
D = diag(X)';   %  This update rule needs the diagonal on X
error = sum(rep.*cols)-D; 
[m,newColIndex] = max(abs(error));
while size(rep,1)<L
    %% Correct the matrix rep = W^{-1}*cols' to add columns we've already selected
    newCol = X(newColIndex,:);
    b = newCol(:,indices);
    d = newCol(:,newColIndex);
    Ainvb = rep(:,newColIndex);
    shur = inv(d-b*Ainvb);
    %% Use block matrix inverse formula to add row to W^{-1}*cols
    brep = b*rep;
    rep = [rep+Ainvb*(shur*(brep-newCol)) ;...
         shur*(-brep+newCol)    ];
    %%  Update record to include what we just added 
    cols = [cols; X(newColIndex,:)]; 
    indices = [indices newColIndex];
    %%  Select new column to be used on next iteration
    error = sum(rep.*cols)-D; 
    %%  Grab out the row with max norm
    if batch == 1
        [m,newColIndex] = max(abs(error));
    else
        [s,ind] = sort(abs(error),'descend');
        newColIndex = ind(1:batch); 
    end
end
indices = indices(1:L);
return


%%  Slow (without rank-1 updates).  Code is kept here for instructional purposes
function indices = innerProductSample_slow(X,L, startSize)
rand('seed',0);
N = size(X,2);  %  Problem size
%% Randomly choose some starting columns
randomColPerm = randperm(N); 
indices = randomColPerm(1:startSize);  %  the indices of selected columns
cols = X(:,indices);           %  the selected columns themselves
D = diag(X)';

while size(cols,2)<L
    %% Sub-sample the selected rows (NOT just the topmost rows)
    W = cols(indices,:); 
    
    %%  Represent each row in the basis of W
    reps = W\cols';
    error = sum(reps.*cols')-D;
   
    %%  Grab out the row with max norm
    [m,newColIndex] = max(abs(error));
    indices = [indices newColIndex];
    cols = [cols X(:,newColIndex)]; 
end

indices = indices(1:L);
return

%%  Randomly sample columns
function [selection, cols]= randomSample(X,L, verbose)
rand('seed',0);
if verbose; fprintf('\tBegin Random...\n'); tic; end
if isnumeric(X)
    N = size(X,2);
    selection = randperm(N);
    selection = selection(1:L);
    cols = X(:,selection);
else   
    if verbose; fprintf('\t\tSelecting Cols'); end
    N = length(X('D','D'));
    selection = randperm(N);
    selection = selection(1:L);
    cols = zeros(N,L);
    for i=1:L
        cols(:,i) = X([],selection(i));
        if verbose 
            if mod(i,50)==1
                fprintf('\n\t\t\t');
            end
            fprintf('.');
        end
    end
    fprintf('\n');
end

if verbose; fprintf('\tEnd Random: time = %f secs\n',toc); end

return


%% sample L columns from X wrt leverage scores
function [selection]= leverageSample(scores,L, verbose)
if verbose; fprintf('\tBegin Leverage Sampling...\n'); end
% selct L samples according to p = abs(x)./sum(abs(x))
randnums = rand(L,1); 
vec = cumsum(abs(scores)./sum(abs(scores)));
% probably a faster way to do this
selection=zeros(1,L);
for i=1:L 
    % find element in vec thats closest to randnums(i) 
    [~,selection(i)] = min(abs(randnums(i)-vec));
end
if verbose; fprintf('\tEnd Sampling: time = %f secs\n',toc); end
return

%  Compute leverage scores
function scores = leveragescores(X,verbose,varargin)
% optional: pass in size of truncated SVD
% compute truncated SVD 
if verbose; fprintf('\tBegin Lev Scores...\n'); end
if nargin>2
    k = varargin{1};
    [Uk,~,~] = svds(X,k);
else
    [Uk,~,~] = svd(X);
end
% leverage scores
scores = norms(Uk').^2;
if verbose; fprintf('\tEnd Lev Scores: time = %f secs\n',toc); end
return


%% Greedy Method of Farahat: sample L columns from X iteratively with rank-1 updates over residual
function selection = farahat(X,L, verbose)
%initialize
Eq = X;
selection = [];
if verbose; fprintf('\tBegin Farahat'); end
for t = 1:L 
    %Criteria for new column is the largest column norm of residual scaled
    %by its diagonal. 
    crit = sum(Eq.^2)./(max(1e-6, diag(Eq)'));

    %This tends to repeat entries so we use a trick
    crit(selection) = 0;
    [~,colopts] = sort(crit,'descend');
    ind = 1;
    while intersect(selection,colopts(ind))
        ind = ind+1;
    end
    q = colopts(ind);
    selection(t) = q;
    
    %Take the column from the residual and subtract its component
    d = Eq(:,q); %selected column
    alpha = d(q); %diagonal entry of column
    w = d/sqrt(alpha); %scaling
    Eq = Eq - w*w'; %new residual
    if verbose;
        if mod(t,50)==1
            fprintf('\n\t\t\t',toc);
        end
        fprintf('.',toc); 
    end
end
if verbose; fprintf('\n\tEnd Farahat: time = %f secs\n',toc); end
return




function opts = fillDefaultOptions(opts)
  


% Display info?
if ~isfield(opts,'verbose')
    opts.verbose = false;
end


% Expand the Nystrom approximation to K = C*W^{-1}C?
if ~isfield(opts,'expandNystrom')
    opts.expandNystrom = true;
end

if ~isfield(opts,'numselect')
    opts.numselect = 1;
end

%  Compute the approximate singular vals/vectors
if ~isfield(opts,'computeApproxSVD')
    opts.computeApproxSVD = false;
end

%  leverageScores, if the user chooses to compute them ahead of time
if ~isfield(opts,'scores')
    opts.scores = [];
end

% columns to sample - if the user wants to hand this in explicitly
if ~isfield(opts,'selection')
    opts.selection = [];
end

return




