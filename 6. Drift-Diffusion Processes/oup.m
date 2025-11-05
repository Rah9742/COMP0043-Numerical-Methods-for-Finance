%% Monte Carlo simulation of the Ornstein-Uhlenbeck process
% dX = alpha*(mu-X)*dt + sigma*dW

% Define the parameters and the time grid
clear variables % clear all variables from workspace
npaths = 20000; % number of paths
T = 1; % time horizon
nsteps = 200; % number of time steps
dt = T/nsteps; % time step
t = (0:dt:T).'; % observation times
alpha = 5; mu = 0.07; sigma = 0.07; % model parameters
X0 = 0.03; % initial value

%% Monte Carlo

% Allocate and initialise all paths
X = [X0*ones(1,npaths);zeros(nsteps,npaths)];

% Sample standard Gaussian random numbers
N = randn(nsteps,npaths);

tic
% Compute the standard deviation for a time step
%sdev = sigma*sqrt(dt); % plain Euler-Maruyama
sdev = sigma*sqrt((1-exp(-2*alpha*dt))/(2*alpha)); % Euler-M. with analytic moments

% Compute and accumulate the increments
for i = 1:nsteps
   %X(i+1,:) = X(i,:) + alpha*(mu-X(i,:))*dt + sdev*N(i,:); % plain Euler-Maruyama
    X(i+1,:) = mu+(X(i,:)-mu)*exp(-alpha*dt) + sdev*N(i,:); % Euler-M. with a. m.
end
toc

%% Expected, mean and sample paths, long-term average
close all
figure(1)
EX = mu+(X0-mu)*exp(-alpha*t); % expected path
plot(t,EX,'k',t,mean(X,2),'k:',t,mu*ones(size(t)),'k--',t,X(:,1:1000:end),t,EX,'k',t,mean(X,2),'k:',t,mu*ones(size(t)),'k--')
legend('Expected path','Mean path','Long-term average')
xlabel('t')
ylabel('X')
sdevinfty = sigma/sqrt(2*alpha);
ylim([mu-4*sdevinfty,mu+4*sdevinfty])
title('Ornstein-Uhlenbeck process dX = \alpha(\mu-X)dt + \sigmadW')
print('-dpng','oupaths.png')

%% Variance = mean square deviation
figure(2)
plot(t,sigma^2/(2*alpha)*(1-exp(-2*alpha*t)),'r',t,sigma^2*t,'g', ...
    t,sigma^2/(2*alpha)*ones(size(t)),'b',t,var(X,0,2),'m',t,mean((X-EX).^2,2),'c--')
legend('Theory','\sigma^2t','\sigma^2/(2\alpha)','Sampled 1','Sampled 2','Location','SouthEast')
xlabel('t')
ylabel('Var(X) = E((X-E(X))^2)')
ylim([0 0.0006])
title('Ornstein-Uhlenbeck process: variance')
print('-dpng','ouvariance.png')

%% Mean absolute deviation
figure(3)
plot(t,sigma*sqrt((1-exp(-2*alpha*t))/(pi*alpha)),'r',t,sigma*sqrt(2*t/pi),'g', ...
    t,sigma/sqrt(pi*alpha)*ones(size(t)),'b',t,mean(abs(X-EX),2),'m')
legend('Theory','\sigma(2t/\pi)^{1/2}','Long-term average','Sampled','Location','SouthEast')
xlabel('t')
ylabel('E(|X-E(X)|) = (2Var(X)/pi)^{1/2}')
ylim([0 0.02])
title('Ornstein-Uhlenbeck process: mean absolute deviation')
print('-dpng','mad.png')

%% Probability density function at different times
x = linspace(-0.02,mu+4*sdevinfty,200).';
t2 = [0.05 0.1 0.2 0.4 1];
EX2 = mu+(X0-mu)*exp(-alpha*t2);
sdev = sigma*sqrt((1-exp(-2*alpha*t2))/(2*alpha));
fa = zeros(length(x),length(t2)); % analytical
fs = zeros(length(x),length(t2)); % sampled
for i = 1:length(t2)
    fa(:,i) = pdf('norm',x,EX2(i),sdev(i));
    fs(:,i) = hist(X(t2(i)*nsteps,:),x)/(npaths*(x(2)-x(1)));
end
figure(4)
plot(x,fa,x,fs)
legend('t = 0.05','t = 0.10','t = 0.20','t = 0.40','t = 1.00')
xlabel('x')
ylabel('f_X(x,t)')
title('Ornstein-Uhlenbeck process: PDF at different times')
print('-dpng','oudensities.png')

%% Autocovariance
C = zeros(2*nsteps+1,npaths);
for j = 1:npaths
    C(:,j) = xcorr(X(:,j)-EX,'unbiased');
end
C = mean(C,2);
figure(5)
plot(t,sigma^2/(2*alpha)*exp(-alpha*t),'r',t,C(nsteps+1:end),'g',0,sigma^2/(2*alpha),'go',0,mean(var(X,0,2)),'bo','Linewidth',1.5)
xlabel('\tau')
ylabel('C(\tau)')
legend('Theory for infinite t','Sampled','Var for infinite t','Sampled Var')
title('Ornstein-Uhlenbeck process: autocovariance')
print('-dpng','ouautocov.png')

%% Autocorrelation
figure(6)
plot(t,exp(-alpha*t),'r',t,C(nsteps+1:end)/C(nsteps+1),'g','Linewidth',1.5)
xlabel('\tau')
ylabel('c(\tau)')
legend('Theory for infinite t','Sampled')
title('Ornstein-Uhlenbeck process: autocorrelation')
print('-dpng','ouautocorr.png')
