function [Err,Ncut,nnz,VV,Errseed,VV2] = computemetrics(X,Xnm,selection,opts,varargin)
    
VV2=0;

labels = opts.labels;
select = sortbylabel(selection,labels);

D1 = normc(X(:,sort(select)));
sizeVV = size(D1,2)*size(X,2);

Err =  sum(sum((D1*pinv(D1)*X - X).^2))./Xnm;
    
% Compute V --- come back and fix this !!!
[D1,VV] = seed(X,selection,opts);

% Calculate cut ratios
Ncut = cocutmetric(VV,selection,opts.labels); 
nnz = sum(abs(VV(:))>1e-5)./sizeVV;
    
Errseed = sum(sum((D1*VV - X).^2))./Xnm;
    
if isfield(opts,'outliers')
    VV2 = createsuppmat(D1,opts.kmax,opts.epsilon); % compute ssc suppmat for outlier detection
    VV(:,selection) = VV2;
end
      
end   % end function


function  refset = sortbylabel(selection,labels)

[~,i,~] = unique(selection);
notid = setdiff((1:length(selection)),i);
selection(notid)=[];
[~,idd] = sort(labels(selection)); 
refset = selection(idd);

end


