%% Acceptance-rejection method for Gaussian from Laplace random variables

clear variables
n = 10^6; % number of samples
xmax = 4; % grid bound
deltax = 0.2; % grid step
x = -xmax:deltax:xmax; % grid
f = @(x) 1/sqrt(2*pi)*exp(-(x.^2)/2); % standard Gaussian PDF
g = @(x) 0.5*exp(-abs(x)); % standard Laplace PDF
c = sqrt(2*exp(1)/pi); % optimal value: f = cg in one point, c = max(f/g)

% Sample the standard Laplace or double-sided exponential distribution
U1 = rand(1,n);
L = log(2*U1).*(U1<0.5)-log(2*(1-U1)).*(U1>=0.5);

% Sample the normal distribution using the acceptance-rejection method
gL = g(L);
fL = f(L);
U2 = rand(1,n);
N = L(U2*c.*gL<=fL); % accept only the values of L that fulfil the condition

% Output to console
length(N)/n % acceptance ratio
format long
c % analytical value
max(fL./gL) % numerical check

close all
figure(1)
x2 = x-deltax/2; % left bin edges
histogram(N,x2,'normalization','pdf');
hold on
fx = f(x);
gx = g(x);
plot(x,fx,x,c*gx,'g')
xlabel('x')
legend('Sampled f(x)','Theoretical f(x)','Majorant function cg(x)')
title('Standard normal distribution using the acceptance-rejection algorithm')

figure(2)
plot(x,x.^2-2*x+1,x,fx./gx,x,c*ones(size(x)),'--g')
xlim([0 3])
xlabel('x')
legend('x^2-2x+1','f/g','c = (2e/pi)^{1/2}','Location','northwest')
title('x^2-2x+1 = 0 where f/g = max = c')