%% Linear congruential random number generator
% Seydel, Course Notes, Chapter 2

close all

% Good example, page 203
a = 1597; b = 51749; M = 244944;
seed = 12345;
% seed = tic;
nsample = 10000;
nbins = 50;

% Pathologic example, page 208
% a = 2; b = 0; M = 11;
% seed = 1;
% nsample = 11;
% nbins = 11;

% Pathologic example, page 209
% a = 1229; b = 1; M = 2048;
% seed = 1;
% nsample = 2048;
% nbins = 20;

% Pathologic example, page 210: RANDU
% a = 2^16+3; b = 0; M = 2^31;
% seed = 1;
% nsample = 10000;
% nbins = 50;

% Generator, page 202
U = zeros(1,nsample);
U(1) = mod(seed,M);
for n = 2:nsample
    U(n) = mod(a*U(n-1)+b,M);
end
U = U/M;

%% Quality checks

% Probability density function
figure(1)
h = histogram(U,nbins,'normalization','pdf');
hold on;
x = h.BinEdges;
plot(x,ones(size(x)))
xlim([-0.2 1.2])
ylim([0 2])
xlabel('x')
ylabel('f_U')
legend('Sample','Theory')
title('Probability density function')

% 2D scatter plot
figure(2)
plot(U(1:nsample-1),U(2:nsample),'.')
xlabel('U_i')
ylabel('U_{i+1}')
title('2D scatter plot')

% 3D scatter plot
figure(3)
plot3(U(1:nsample-2),U(2:nsample-1),U(3:nsample),'.')
xlabel('U_i')
ylabel('U_{i+1}')
zlabel('U_{i+2}')
title('3D scatter plot')