% Compare error, sparsity, and clustering for SEED
% Example Usage:
% Results = compute_err([],[1:5:150],[]);

%%% EXAMPLE %%%
% opts.labels = ones(N,1); % if no labels exist
% opts.epsilon = 0.3;
% opts.kmax = 20;

%%% EXAMPLE (SYNTHETIC DATA) %%%
% opts.numsub = 10; opts.ambdim=200; opts.subdim=10; 
% opts.n1=2.^[5:5+opts.numsubs/2]; opts.n2=2.^[5:5+opts.numsubs/2];
% opts.k1=10*ones(1,opts.numsubs/2); opts.k2=20*ones(1,opts.numsubs/2);
% opts.overlap = 5; opts.sampling = 'nonuniform';
% opts.kmax = opts.k2; opts.epsilon = 0.05;
% Results = compute_err([],[1:5:150],opts);

%%%% ORDERING OF OUTPUTS
% (OUTPUT) Results.Err = {asis, err, lev, rand}
% (OUTPUT) Results.Ncut0 = {gram, ssc, nn};

function Results = compare_cssmethods(X,L,opts)

[M,N] = size(X);
labels = ones(N,1);    

if ~isfield(opts,'labels')
    opts.labels = labels;
end

%%%%%% If X is empty, Generate synthetic data
if isempty(X)
   [X,labels] = gensynthdata(opts);
   [M,N] = size(X);
end

%%%%%% read in arguments (for sparse recovery)
if isfield(opts,'kmax')
    kmax = opts.kmax;
else
    kmax = 20;
end

if isfield(opts,'numselect')
    numselect = opts.numselect;
else
    numselect = 10;
    opts.numselect = 10;
end

if isfield(opts,'knn')
    knn = opts.knn;
else
    knn = kmax;
end

if isfield(opts,'epsilon')
    epsilon = opts.epsilon;
else
    epsilon = 0.05;
end

Xnm = sum(X(:).^2);
Lmax = max(L);
Errs = zeros(length(L),4);
Errseed = zeros(length(L),4);
Ncut = zeros(length(L),4);

if length(labels)~=N
    error('Not enough (or too many) labels!')
end

% Step 0. Generate Gram matrix
Xnorm = normc(X);
G=Xnorm'*Xnorm;

if isfield(opts,'Gnn')
    Gnn = opts.Gnn;
else
    Gnn = knngraph(abs(G),knn);
end

% Step 1. Run Different Sampling Methods
% 1. Oasis sampling
[ outs ] = nystrom(G,Lmax,'p',opts); t1 = toc;
asisset = outs.selection;

% 2. Error sampling
errset = errorsamp(X,Lmax,numselect); t2 = toc; % select one signal at a time

% 3. Leverage sampling
levset = leveragesamp(X,M,Lmax);

% 4. Random sampling
randset = uniformsamp(Lmax,N); t4 = toc;

for i=1:length(L)
    
    selection = unique(asisset(1:min(L(i),length(asisset))));
    [Errs(i,1),Ncut(i,1),NNz(i,1),VV{1},Errseed(i,1),VV2{1}] = computemetrics(X,Xnm,selection,opts);
    
    selection = unique(errset(1:min(L(i),length(errset))));
    [Errs(i,2),Ncut(i,2),NNz(i,1),VV{2},Errseed(i,2),VV2{2}] = computemetrics(X,Xnm,selection,opts);
    
    selection = unique(levset(1:min(L(i),length(levset))));
    [Errs(i,3),Ncut(i,3),NNz(i,1),VV{3},Errseed(i,3),VV2{3}] = computemetrics(X,Xnm,selection,opts);
    
    selection = unique(randset(1:min(L(i),length(randset))));
    [Errs(i,4),Ncut(i,4),NNz(i,1),VV{4},Errseed(i,4),VV2{4}] = computemetrics(X,Xnm,selection,opts);
    
    if i==length(L)
        Results.Vseed = VV;
        Results.VV2 = VV2;
    end
        
end

% compute entire SSC decomposition
if isfield(opts,'Vssc')
    Vssc = opts.Vssc;
else
    Vssc = createsuppmat(X,kmax,epsilon);
end
Gssc= abs(Vssc)+abs(Vssc');
    
% compute clustering metrics
ym = cocutmetric(G,[1:N],labels);
sscm = cocutmetric(Gssc,[1:N],labels);
nnm = cocutmetric(Gnn,[1:N],labels);

opts.M = M;
opts.N = N;
opts.kmax = kmax;
opts.epsilon = epsilon;
opts.Lmax = Lmax;
Results.opts = opts;
Results.X = X;
Results.G = G;
Results.Vssc = Vssc;
Results.Ncuts = Ncut; 
Results.Ncut0 = [ym; sscm; nnm];
Results.Errs = Errs;
Results.Errseed = Errseed;
Results.Sparsity = NNz;
Results.Labels = labels;
Results.ClustMethods = {'SEED','ErrorSamp','LeverageSamp',...
                        'RandomSamp','Gram','SSC', 'NN'};
Results.SamplesoASIS = outs.selection;

end


function  refset = sortbylabel(selection,labels)

[~,i,~] = unique(selection);
notid = setdiff((1:length(selection)),i);
selection(notid)=[];
[~,idd] = sort(labels(selection)); 
refset = selection(idd);

end


