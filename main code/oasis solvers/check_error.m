function [runningerror,numrandsamples,relerror,actualerror] =  check_error(M,Mt);

runningerror = Inf;
numrandsamples = 0;
G = [];
Gt = [];
numsamples = 1000;

count=0;
while runningerror > 0.01
    numrandsamples = numrandsamples+1000;
    
    numpoints = size(M,1); %need to know number of points in matrix
    G = [G; zeros(numsamples,1)];
    Gt = [Gt; zeros(numsamples,1)];
    pts = randi(numpoints,numsamples,2);
    
    for ind = 1:size(pts,1);
        Gt(count*numsamples+ind) = Mt(pts(ind,1),pts(ind,2));
        G(count*numsamples+ind) = M(pts(ind,1),pts(ind,2));
    end
    
    
    e = abs(G-Gt);
    E = mean(e);
    
    sig = (var(e)).^0.5;
    
    %zscore = E/sig*numrandsamples.^0.5
    runningerror = sig/(numrandsamples.^0.5*E);
    count = count+1;
    %disp([count E sig runningerror]);
end

relerror =          sum(e)/max(norm(G,1),norm(Gt,1));

actualerror = norm(M-Mt,1)/max(norm(M,1),norm(Mt,1));

return
