%%%%%%%% SEED demo %%%%%%%%
% 1. Generate synthetic dataset
% 2. Compute error for SEED, Leverage, Random, and SES sampling methods
% 3. Compute normalized cuts for SEED, Leverage, Random, SES, SSC, NN
% 4. Plot error curves and normalized cuts


% input parameters
opts.kmax = 5;
opts.epsilon = 0.05;

Results = compare_cssmethods([],10:10:100,opts); % Steps (1-3)

plot_err_ncuts(Results) % Step 4

