%% Using Fully Flexible Probabilities for Strategic Asset Allocation: 
%  an Empirical Experiment on a Global Minimum Variance Portfolio
%
% All scripts and function written by Marvin DENIS based on various sources
% mentionned in the associated thesis. 
% -------------------------------------------------------------------------
clc; clear; close all;

addpath([pwd '/Functions/']);
addpath([pwd '/Database/']);
load DB.mat
load hist_volatility.mat

%% DATA PROCESSING
% -------------------------------------------------------------------------

returns_dates=Dates(2:end);

% risk drivers for the stocks under consideration
risk_drivers = log(Prices);
% invariants (i.i.d. variables) from the timeseries analysis of the risk drivers (log-values)
invariants = diff(log(Prices));

% Here we switch to weekly data. Rationale vs. monthly or yearly : eliminates some of the noise from the daily prices without oversmoothing
% tau-days empirical observations:
tau = 5;            %weekly data
tau_dates = returns_dates(1:tau:end);
tau_prices = Prices(1:tau:end, 1:9);
tau_invariants = diff(log(tau_prices));
tau_SPprices = Prices(1:tau:end, 10);
tau_SPinv = diff(log(tau_SPprices));
tau_VIX=Prices(1:tau:end,11);
VIX_invariants=diff(log(tau_VIX));

% Recover the projected scenarios for the risk drivers at the tau-day horizon
proj_risk_drivers = log(tau_prices((1:end-1),:)) + tau_invariants;
proj_prices = exp(proj_risk_drivers);
PnL = proj_prices - tau_prices((1:end-1),:);    % ex-ante P&L of each stock

% Aggregation of the individual stock P&Ls into projected portfolio P&L for all scenarios
% Assume equally weighted protfolio at beginning of investment period
[J, N] = size(invariants(:,1:9));
selected_stocks=[1:9]';
% initial holdings (# of shares)
eqw = zeros(N,1);
eqw(selected_stocks)=1/length(selected_stocks);
h=eqw./Prices(end,1:9)';                        % quantity of each stock held in the portfolio
agg_PnL=PnL*h;                                  % portfolio aggregated P&L

% Variance risk premium trial
% histvol=hist5Dvol(30:30:end,:);
% histvol1=histvol(1);
% VIX2=Prices(1:30:end,11);
% VIX2_invariants=diff(log(VIX2));
% VP_dates=Dates(30:30:end);
% VP=histvol1-VIX2_invariants;

% figure
% subplot(3,1,1)
% plot(VP_dates, VP)
% subplot(3,1,2)
% plot(VP_dates, VIX2_invariants)
% subplot(3,1,3)
% plot(VP_dates, histvol1)

hist=hist20Dvol(5:tau:end,1);
test = hist - VIX_invariants;

figure(1)
plot(tau_dates(2:4:end), test(1:4:end), tau_dates(2:4:end), hist(1:4:end), 'b' , tau_dates(2:4:end), VIX_invariants(1:4:end), 'r', 'LineWidth',1)


%% GLOBAL MINIMUM VARIANCE TWO-STEP OPTIMISATION 
% -------------------------------------------------------------------------

% Step 0: data prep
% Historical prices and invariants for non-rolling optimisation (January 1999 - December 2003)
opt_invariants=tau_invariants(1:258,:);
opt_prices=tau_prices(258,1:9);
opt_window=tau_dates(1:258);
opt_VIX=tau_VIX(1:258);
Options.NumPortf=40;                      % number of portfolios forming the efficient frontier
Options.FrontierSpan=[.05 .95];           % range of normalized expected values spanned by efficient frontier

% ==============================
% NO Rolling Window + NO Entropy
% ==============================

% Step 1: MV quadratic optimization to determine one-parameter frontier of quasi-optimal solutions
eq_p=ones([length(opt_invariants),1])/length(opt_invariants);   % equal probabilities
[e1,s1,w1,M1,S1]=EfficientFrontier(opt_invariants,eq_p,Options);

% Step 2: Exact Satisfaction Computation (Case of a "Minimum Variance Portfolio")
satisfaction1=-s1;
[max_sat_value1, max_sat_index1]=max(satisfaction1);            % Choose the allocation that maximises satisfaction
optimal_allocation1=w1(max_sat_index1,:);

% ==============================
% NO Rolling Window + Entropy
% ==============================

% 3 step (Prior, Views, Posterior)
% Processing of the views - Conditioning framework
% -------------------------------------------------------------------------
T=size(opt_invariants,1);

% Time Crisp Conditioning
% -------------------------------------------------------------------------
rw_p=zeros(T,1);
%time_window=2*52;
time_window=0.5*52;
rw_p(end-time_window:end)=1;    % more weight on the end of the sample than the start
%rw_p(1:tau)=1;                 % opposite
rw_p=rw_p/sum(rw_p);            % re-normalising of the probabilities

% Exponential Smoothing Conditioning
% -------------------------------------------------------------------------
%half_life=2*52;                 % we test for a half-life of 6 months and 2 years
half_life=0.5*52;
lambda=log(2)/half_life;
t_p=exp(-lambda*(T-(1:T)'));
t_p=t_p/sum(t_p); 

% State Crisp Conditioning
% -------------------------------------------------------------------------
s_p=zeros(T,1);
cond_VIX=tau_VIX(1:258);        % based on expectation of VIX above or below [X] level
cond=cond_VIX<=20;              % change here to >=20; >=15; >=25 to replicate results in Section 6
s_p(cond)=1;
s_p=s_p/sum(s_p);  

% Kernel Smoothing Conditioning
% -------------------------------------------------------------------------
k_p=zeros(T,1);
VIX_level=20;                   % change here to 15; 25 to replicate results in Appendix B
cond=cond_VIX<=20;              % change here to >=20; >=15; >=25 to replicate results in Section 6
h2=cov(diff(cond_VIX));
%absdiff=cond_VIX-VIX_level;    % when VIX higher than [X]
absdiff=VIX_level-cond_VIX;     % when VIX lower than [X]
k_p=mvnpdf(absdiff,VIX_level,h2);
k_p=k_p/sum(k_p);

% Joint time and state conditioning through minimum relative entropy
% -------------------------------------------------------------------------
% 1. specify view on expectation and standard deviation
p_prior=t_p;                    % set the prior to exponentially decaying conditioned probabilities
expvalue=sum(s_p.*cond_VIX);
variance=sum(s_p.*cond_VIX.*cond_VIX)-expvalue.^2;
% 2. posterior market distribution using minimum relative entropy
p_post=TimeStateConditioning(cond_VIX,p_prior,expvalue,variance);


% 2-step optimisation (see above for explanations)
[e2,s2,w2,M2,S2] = EfficientFrontier(opt_invariants,p_post,Options);
satisfaction2=-s2;
[max_sat_value2, max_sat_index2]=max(satisfaction2);
optimal_allocation2=w2(max_sat_index2,:);

%% Other conditioning techniques - No rolling window
% ==================================================
[e_rw,s_rw,w_rw,M_rw,S_rw] = EfficientFrontier(opt_invariants,rw_p,Options);
[e_t,s_t,w_t,M_t,S_t] = EfficientFrontier(opt_invariants,t_p,Options); % frontier in terms of number of shares
[e_s,s_s,w_s,M_s,S_s] = EfficientFrontier(opt_invariants,s_p,Options);
[e_k,s_k,w_k,M_k,S_k] = EfficientFrontier(opt_invariants,k_p,Options);

satisfaction_rw=-s_rw; [max_sat_value_rw, max_sat_index_rw]=max(satisfaction_rw); optimal_allocation_rw=w_rw(max_sat_index_rw,:);
satisfaction_t=-s_t; [max_sat_value_t, max_sat_index_t]=max(satisfaction_t); optimal_allocation_t=w_t(max_sat_index_t,:);
satisfaction_s=-s_s; [max_sat_value_s, max_sat_index_s]=max(satisfaction_s); optimal_allocation_s=w_s(max_sat_index_s,:);
satisfaction_k=-s_k; [max_sat_value_k, max_sat_index_k]=max(satisfaction_k); optimal_allocation_k=w_k(max_sat_index_k,:);

% ------------------------------
% ROLLING WINDOW
% ==============================
% inputs for rolling window
s_start = tau_dates(1);
s_end = tau_dates(end);
window = 258;                           % size in weeks of the rolling window (=±5y)
ss=datefind(s_start,tau_dates);
se=datefind(s_end,tau_dates);
LastRoll=se-ss+1-window;                % last rolling window
counter = 0;
pr3=[]; pr4=[]; p_post_rr=[];

% inputs for backtest when using a rolling window
b_start= tau_dates(258);
b_end= s_end;
bs=datefind(b_start,tau_dates);
be=datefind(b_end,tau_dates)-1;
b_window = 52;

% Rolling Window loop

for j = 1 : b_window : LastRoll
    
    counter = counter +1;
    fprintf(1, 'Calculating Rolling Bootstrapped Portfolio: Year');
    fprintf(1, '%i', counter);
    
    ssRolling = ss + j - 1;  % Rolling Sample Start (First Raw)
    seRolling = ssRolling + window - 1; % Sample End (Last Raw) 
    ObsSampleRolling = seRolling-ssRolling+1; % Number of observations within Rolling Sample
    
    Rolling_Returns(:,:,counter) = tau_invariants((ssRolling:seRolling),:);
    Rolling_Prices(:,:,counter) = tau_prices(seRolling,:);

    Options.CurrentPrices=Rolling_Prices(:,:,counter)';
    
    % ------------------------------
    % Rolling Window + NO Entropy
    % ==============================
    
    % 2-step optimisation (see above for explanations)
    [e3(:,counter),s3(:,:,counter),w3(:,:,counter),M3(:,counter),S3(:,:,counter)]=EfficientFrontier(Rolling_Returns(:,:,counter),eq_p,Options);
    satisfaction3(:,:,counter)=-s3(:,:,counter);
    [max_sat_value3(:,:,counter), max_sat_index3(:,:,counter)]=max(satisfaction3(:,:,counter));
    optimal_allocation3(:,:,counter)=w3(max_sat_index3(:,:,counter),:, counter);
    
    bsRolling = bs+j-1;
    beRolling = bsRolling + b_window - 4;
    ObsOutSampleRolling = beRolling - bsRolling;
    
    % Backtest without Entropy
    rolling_ret(:,:,counter) = tau_invariants((bsRolling:beRolling),:);
    pr3(:,:,counter)=rolling_ret(:,:,counter)*optimal_allocation3(:,:,counter)';
    
    % ------------------------------
    % Rolling Window + Entropy
    % ==============================
    
    % t_p does not need to be made rolling but s_p does. 
    % "Crisp" Macro-economic conditioning
    s_pr=zeros(T,1);
    cond_VIXr=tau_VIX(ssRolling:seRolling);
    condr=cond_VIXr<=20;
    s_pr(condr)=1;
    s_pr=s_pr/sum(s_pr);     
    
    % Kernel Smoothing
    k_pr=zeros(T,1);
    h2r=cov(diff(cond_VIXr));
    %absdiffr=cond_VIXr-VIX_level;        % when VIX higher than X
    absdiffr=VIX_level-cond_VIXr;         % when VIX lower than X
    k_pr=mvnpdf(absdiffr,VIX_level,h2r);
    k_pr=k_pr/sum(k_pr);

    % Time & State conditioning via Entropy Pooling
    p_prior_r=t_p;
    ExpValue_r=sum(s_pr.*cond_VIXr);
    Variance_r=sum(s_pr.*cond_VIXr.*cond_VIXr)-ExpValue_r.^2;
    % posterior market distribution using the Entropy Pooling approach
    p_post_r=TimeStateConditioning(cond_VIXr,p_prior_r,ExpValue_r,Variance_r);
    p_post_rr(:,:,counter)=TimeStateConditioning(cond_VIXr,p_prior_r, ExpValue_r, Variance_r);
    
    % 2-step optimisation (see above for explanations)
    [e4(:,counter),s4(:,:,counter),w4(:,:,counter),M4(:,counter),S4(:,:,counter)]=EfficientFrontier(Rolling_Returns(:,:,counter),p_post_r,Options);
    satisfaction4(:,:,counter)=-s4(:,:,counter);
    [max_sat_value4(:,:,counter), max_sat_index4(:,:,counter)]=max(satisfaction4(:,:,counter));
    optimal_allocation4(:,:,counter)=w4(max_sat_index4(:,:,counter),:, counter);
    
    % Backtest with Entropy
    pr4(:,:,counter)=rolling_ret(:,:,counter)*optimal_allocation4(:,:,counter)';
    
    % ---------------------------------------------------
    % Other conditioning techniques - With rolling window
    % ===================================================
    
    % rw_p & t_p do not change
    % s_p & k_p do change
    % 2-step optimisation (see above for explanations)
    [e_sr(:,counter),s_sr(:,:,counter),w_sr(:,:,counter),M_sr(:,counter),S_sr(:,:,counter)] = EfficientFrontier(Rolling_Returns(:,:,counter),s_pr,Options);
    [e_kr(:,counter),s_kr(:,:,counter),w_kr(:,:,counter),M_kr(:,counter),S_kr(:,:,counter)] = EfficientFrontier(Rolling_Returns(:,:,counter),k_pr,Options);
    
    satisfaction_sr(:,:,counter)=-s_sr(:,:,counter); [max_sat_value_sr(:,:,counter), max_sat_index_sr(:,:,counter)]=max(satisfaction_sr(:,:,counter)); optimal_allocation_sr(:,:,counter)=w_sr(max_sat_index_sr(:,:,counter),:, counter);
    satisfaction_kr(:,:,counter)=-s_kr(:,:,counter); [max_sat_value_kr(:,:,counter), max_sat_index_kr(:,:,counter)]=max(satisfaction_kr(:,:,counter)); optimal_allocation_kr(:,:,counter)=w_kr(max_sat_index_kr(:,:,counter),:, counter);
    
    % backtest
    pr_sr(:,:,counter)=rolling_ret(:,:,counter)*optimal_allocation_sr(:,:,counter)';
    pr_kr(:,:,counter)=rolling_ret(:,:,counter)*optimal_allocation_kr(:,:,counter)';
end

%% BACKTESTING 
% -------------------------------------------------------------------------
bsend = 879; rbsend = 622;
% Backtesting of a simple buy and hold strategy

% Historical PnL for Backtest (Jan 2004 - December 2016)
BT_invariants = tau_invariants(258:bsend,1:9);
BT_SP = tau_SPinv(258:bsend);
BT_dates = tau_dates(258:bsend+1);

% Backtest with Equal Weights (no flex probs)
portfolio_returns0=BT_invariants*eqw;
portfolio_profits0=10*[1;exp(cumsum(portfolio_returns0))];

% Backtest without Entropy - No Rolling
portfolio_returns1=BT_invariants*optimal_allocation1';
portfolio_profits1=10*[1;exp(cumsum(portfolio_returns1))];

% Backtest with Entropy - No Rolling
portfolio_returns2=BT_invariants*optimal_allocation2';
portfolio_profits2=10*[1;exp(cumsum(portfolio_returns2))];

% Backtest S&P
portfolio_profits_SP=10*[1;exp(cumsum(BT_SP))];

% Backtesting of a buy and hold strategy with a Walk Forward Rolling Window
% No entropy profits - Rolling
portfolio_returns3 = num2cell(pr3, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns3 = vertcat(portfolio_returns3{:});    portfolio_returns3 = portfolio_returns3(1:rbsend);
portfolio_profits3=10*[1;exp(cumsum(portfolio_returns3))];

% Entropy profits - Rolling
portfolio_returns4 = num2cell(pr4, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns4 = vertcat(portfolio_returns4{:});    portfolio_returns4 = portfolio_returns4(1:rbsend);
portfolio_profits4=10*[1;exp(cumsum(portfolio_returns4))];

figure(2)
plot(BT_dates, portfolio_profits0, BT_dates, portfolio_profits1, BT_dates, portfolio_profits2, BT_dates, portfolio_profits3, BT_dates, portfolio_profits4, BT_dates, portfolio_profits_SP, 'LineWidth',3)
xlim([tau_dates(258) tau_dates(bsend)]);
ylim([5 32]);
set(gca, 'fontsize', 35);
datetick('x','yyyy','keeplimits','keepticks');
% legend('Equal Weights', 'GMV Without Entropy', 'GMV With Entropy', 'GMV Without Entropy Rolling','GMV With Entropy Rolling', 'S&P500', 'location', 'northwest');
% title('Performance', 'fontsize', 32);
xlabel('Year', 'fontsize', 32);
ylabel('Portfolio Balance (in Thousands)', 'fontsize', 32);
grid on


%% BACKTEST - Other conditioning techniques
% ========================================

% rolling window
portfolio_returns_rw=BT_invariants*optimal_allocation_rw';
portfolio_profits_rw=10*[1;exp(cumsum(portfolio_returns_rw))];

% time conditioning
portfolio_returns_t=BT_invariants*optimal_allocation_t';
portfolio_profits_t=10*[1;exp(cumsum(portfolio_returns_t))];

% state conditioning
portfolio_returns_s=BT_invariants*optimal_allocation_s';
portfolio_profits_s=10*[1;exp(cumsum(portfolio_returns_s))];
                   
% kernel smoothing
portfolio_returns_k=BT_invariants*optimal_allocation_k';
portfolio_profits_k=10*[1;exp(cumsum(portfolio_returns_k))];

% Views Performance
figure(3)
plot(BT_dates,portfolio_profits_rw, BT_dates,portfolio_profits_t, BT_dates, portfolio_profits_s, BT_dates, portfolio_profits_k, BT_dates, portfolio_profits2, 'LineWidth', 3)
xlim([tau_dates(258) tau_dates(bsend)]);
ylim([7 35]);
set(gca, 'fontsize', 35);
datetick('x','yyyy','keeplimits','keepticks');
legend('Time Crisp', 'Exponential Smoothing', 'State Crisp', 'Kernel Smoothing', 'Entropy', 'location', 'northwest');
title('Conditioned Portfolios Performance', 'fontsize', 32);
xlabel('Year', 'fontsize', 32);
ylabel('Portfolio Balance (in Thousands)', 'fontsize', 32);
grid on

% ------------------------------------------------------------
% BACKTEST - Other conditioning techniques with Rolling Window
% ============================================================

% state conditioning - Rolling
portfolio_returns_sr = num2cell(pr_sr, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns_sr = vertcat(portfolio_returns_sr{:});    portfolio_returns_sr = portfolio_returns_sr(1:rbsend);
portfolio_profits_sr=10*[1;exp(cumsum(portfolio_returns_sr))];

% kernel profits - Rolling
portfolio_returns_kr = num2cell(pr_kr, [1,2]);              % 3D Matrix to 2D [Nx1] Matrix
portfolio_returns_kr = vertcat(portfolio_returns_kr{:});    portfolio_returns_kr = portfolio_returns_kr(1:rbsend);
portfolio_profits_kr=10*[1;exp(cumsum(portfolio_returns_kr))];

%% Views Performance
figure(4)
plot(BT_dates,portfolio_profits_sr, BT_dates,portfolio_profits_kr, BT_dates, portfolio_profits4, 'LineWidth', 3)
xlim([tau_dates(258) tau_dates(bsend)]);
ylim([7 35]);
set(gca, 'fontsize', 35);
datetick('x','yyyy','keeplimits','keepticks');
legend('State Crisp Rolling', 'Kernel Rolling', 'Entropy Rolling', 'location', 'northwest');
title('Rolling Conditioned Portfolios Performance', 'fontsize', 32);
xlabel('Year', 'fontsize', 32);
ylabel('Portfolio Balance (in Thousands)', 'fontsize', 32); 
grid on


%% EX-POST ANALYSIS - Portfolio Performance Measures
% -------------------------------------------------------------------------

portfolio_returns = [portfolio_returns0, portfolio_returns1, portfolio_returns2, portfolio_returns3, portfolio_returns4, BT_SP]; 
portfolio_profits = [portfolio_profits0, portfolio_profits1, portfolio_profits2, portfolio_profits3, portfolio_profits4, portfolio_profits_SP];

total_return=Total_Return(portfolio_profits);           % Total Return
annual_return=mean(portfolio_returns)*52;               % Average Annual Return
monthly_return=mean(portfolio_returns)*4.3;             % Average Monthly Return
annual_volatility=std(portfolio_returns)*sqrt(52);      % Annual Volatility
positive_trades=Positive_Trades(portfolio_returns);     % Positive Trades
max_dd=maxdrawdown(portfolio_profits);                  % Max Drawdown
[sortino,inforatio] = Ratios(portfolio_returns,BT_SP);  % Sortino Ratio & Information Ratio

% Historical Value at Risk & Expected Shortfall
confidence_level = 0.99;
figure(5)
[VaR0, ES0] = HistVaR_ES(portfolio_returns0,confidence_level); [VaR1, ES1] = HistVaR_ES(portfolio_returns1,confidence_level); [VaR2, ES2] = HistVaR_ES(portfolio_returns2,confidence_level); [VaR3, ES3] = HistVaR_ES(portfolio_returns3,confidence_level); [VaR4, ES4] = HistVaR_ES(portfolio_returns4,confidence_level); [VaR5, ES5] = HistVaR_ES(BT_SP,confidence_level);
VaR=[-VaR0, -VaR1, -VaR2, -VaR3, -VaR4, -VaR5];
ES=[-ES0, -ES1, -ES2, -ES3, -ES4, -ES5];


%% EX-POST ANALYSIS - Other conditioning techniques
% ================================================

conditioning_returns=[portfolio_returns_rw, portfolio_returns_t, portfolio_returns_s, portfolio_returns_k, portfolio_returns_sr, portfolio_returns_kr];
conditioning_profits=[portfolio_profits_rw, portfolio_profits_t, portfolio_profits_s, portfolio_profits_k, portfolio_profits_sr, portfolio_profits_kr];

ctotal_return=Total_Return(conditioning_profits);           % Total Return
cannual_return=mean(conditioning_returns)*52;               % Average Annual Return
cmonthly_return=mean(conditioning_returns)*4.3;             % Average Monthly Return
cannual_volatility=std(conditioning_returns)*sqrt(52);      % Annual Volatility
cpositive_trades=Positive_Trades(conditioning_returns);     % Positive Trades
cmax_dd=maxdrawdown(conditioning_profits);                  % Max Drawdown
[csortino,cinforatio] = Ratios(conditioning_returns,BT_SP); % Sortino Ratio & Information Ratio

% Historical Value at Risk & Expected Shortfall
figure(6)
[VaRrw, ESrw] = HistVaR_ES(portfolio_returns_rw,confidence_level); [VaRt, ESt] = HistVaR_ES(portfolio_returns_t,confidence_level); [VaRs, ESs] = HistVaR_ES(portfolio_returns_s,confidence_level); [VaRk, ESk] = HistVaR_ES(portfolio_returns_k,confidence_level); [VaRsr, ESsr] = HistVaR_ES(portfolio_returns_sr,confidence_level); [VaRkr, ESkr] = HistVaR_ES(portfolio_returns_kr,confidence_level);
VaRc=[-VaRrw, -VaRt, -VaRs, -VaRk, -VaRsr, -VaRkr];
ESc=[-ESrw, -ESt, -ESs, -ESk, -ESsr, -ESkr];



%% Various other figures
% ================================================

prot=num2cell(p_post_rr, [1,2]);
prot=vertcat(prot{:});
prot2=prot(1:5:end);
figure(7)
area(prot2)

% figure
% PlotFrontier(e2,s2,w2)

% MV_PnL=agg_PnL(1:258,:);
% 
% figure
% subplot(3,1,1)
% stem(opt_window,MV_PnL)
% grid on
% set(gca,'xlim',[min(opt_window) max(opt_window)])
% datetick('x','mmmyy','keeplimits','keepticks');
% 
% 
% subplot(3,1,2)
% area(opt_window,rw_p);
% grid on
% set(gca,'xlim',[min(opt_window) max(opt_window)])
% datetick('x','mmmyy','keeplimits','keepticks');
% 
% 
% subplot(3,1,3)
% area(opt_window,t_p);
% grid on
% set(gca,'xlim',[min(opt_window) max(opt_window)])
% datetick('x','mmmyy','keeplimits','keepticks');
% 
% figure
% subplot(4,1,1)
% plot(opt_window,opt_VIX);
% grid on
% set(gca,'xlim',[min(opt_window) max(opt_window)])
% datetick('x','mmmyy','keeplimits','keepticks');
% 
% 
% subplot(4,1,2)
% bar(opt_window,s_p);
% grid on
% set(gca,'xlim',[min(opt_window) max(opt_window)])
% datetick('x','mmmyy','keeplimits','keepticks');
% 
% 
% subplot(4,1,3)
% area(opt_window,k_p);
% grid on
% set(gca,'xlim',[min(opt_window) max(opt_window)])
% datetick('x','mmmyy','keeplimits','keepticks');
% 
% 
% subplot(4,1,4)
% area(opt_window,p_post);
% grid on
% set(gca,'xlim',[min(opt_window) max(opt_window)])
% datetick('x','mmmyy','keeplimits','keepticks');

% figure
% plot(tau_dates, tau_VIX)
% xlim([min(tau_dates) max(tau_dates)]);
% datetick('x','mmmyy','keeplimits','keepticks');
% 
% figure
% histogram(portfolio_returns_post);


% ALLOCATION DRIFT
% equal probs
% port=zeros(673,N);
% port(:,1)=0.1111*exp(cumsum(BT_invariants(:,1)));
% port(:,2)=0.1111*exp(cumsum(BT_invariants(:,2)));
% port(:,3)=0.1111*exp(cumsum(BT_invariants(:,3)));
% port(:,4)=0.1111*exp(cumsum(BT_invariants(:,4)));
% port(:,5)=0.1111*exp(cumsum(BT_invariants(:,5)));
% port(:,6)=0.1111*exp(cumsum(BT_invariants(:,6)));
% port(:,7)=0.1111*exp(cumsum(BT_invariants(:,7)));
% port(:,8)=0.1111*exp(cumsum(BT_invariants(:,8)));
% port(:,9)=0.1111*exp(cumsum(BT_invariants(:,9)));
% 
% load eqw.mat
% figure
% area(eqw)
% set(gca,'ylim', [0 1])
% datetick('x','mmmyy','keeplimits','keepticks');
% legend(Names)
% 
% portini=zeros(673,N);
% portini(:,1)=0.0108*exp(cumsum(BT_invariants(:,1)));
% portini(:,2)=0.2322*exp(cumsum(BT_invariants(:,2)));
% portini(:,3)=0.0000*exp(cumsum(BT_invariants(:,3)));
% portini(:,4)=0.0000*exp(cumsum(BT_invariants(:,4)));
% portini(:,5)=0.0374*exp(cumsum(BT_invariants(:,5)));
% portini(:,6)=0.3291*exp(cumsum(BT_invariants(:,6)));
% portini(:,7)=0.2721*exp(cumsum(BT_invariants(:,7)));
% portini(:,8)=0.1182*exp(cumsum(BT_invariants(:,8)));
% portini(:,9)=0.0002*exp(cumsum(BT_invariants(:,9)));
% 
% load ini.mat
% figure
% area(ini)
% set(gca,'ylim', [0 1])
% datetick('x','mmmyy','keeplimits','keepticks');
% legend(Names)
% 
% portent=zeros(673,N);
% portent(:,1)=0.0072*exp(cumsum(BT_invariants(:,1)));
% portent(:,2)=0.1294*exp(cumsum(BT_invariants(:,2)));
% portent(:,3)=0.0023*exp(cumsum(BT_invariants(:,3)));
% portent(:,4)=0.0050*exp(cumsum(BT_invariants(:,4)));
% portent(:,5)=0.0097*exp(cumsum(BT_invariants(:,5)));
% portent(:,6)=0.5050*exp(cumsum(BT_invariants(:,6)));
% portent(:,7)=0.1659*exp(cumsum(BT_invariants(:,7)));
% portent(:,8)=0.1753*exp(cumsum(BT_invariants(:,8)));
% portent(:,9)=0.0002*exp(cumsum(BT_invariants(:,9)));
% 
% load ent.mat
% figure
% area(ent)
% set(gca,'ylim', [0 1])
% datetick('x','mmmyy','keeplimits','keepticks');
% legend(Names)