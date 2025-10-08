%% Acceptance-rejection method for normal from Laplace random variables

clear variables
n = 10^6; % number of samples
c = sqrt(2*exp(1)/pi); % max(f/g)
deltax = 0.1; % grid spacing
xmax = 4; % grid bound
x = -xmax:deltax:xmax; % grid

% Sample the standard Laplace or double-sided exponential distribution
U1 = rand(1,n);
L = log(2*U1).*(U1<0.5)-log(2*(1-U1)).*(U1>=0.5);

% Sample the normal distribution using the acceptance-rejection method
g = 0.5*exp(-abs(L));
f = 1/sqrt(2*pi)*exp(-L.^2/2);
U2 = rand(1,n);
N = L(U2*c.*g<=f);

% Output to console
length(N)/n %#ok<NOPTS> acceptance ratio
format long
c %#ok<NOPTS> analytical value
max(f./g) % numerical check

close all
figure(1)
x2 = [x-deltax/2 x(end)+deltax/2]; % bin edges
histogram(N,x2,'normalization','pdf');
hold on
fx = 1/sqrt(2*pi)*exp(-(x.^2)/2);
gx = 0.5*exp(-abs(x));
plot(x,fx,x,c*gx,'g')
xlabel('x')
legend('Sampled f(x)','Theoretical f(x)','Majorant function cg(x)')
title('Standard normal distribution using the accept-reject algorithm')

figure(2)
plot(x,x.^2-2*x+1,x,fx./gx,x,c*ones(size(x)),'--g')
xlim([0 3])
xlabel('x')
legend('x^2-2x+1','f/g','c = (2e/pi)^{1/2}','Location','northwest')
title('x^2-2x+1 = 0 where f/g = max = c')
