%% Fibonacci random number generator (Knuth)
% Seydel, Course Notes, Chapter 2, pages 211-212

% Parameters
mu = 5;
nu = 17;
M = 714025;
%seed = 12345;
seed = mod(tic,M);
nsample = 10000;

% Linear congruential generator to start the sequence
a = 1366; b = 150889;
U = zeros(1,nu);
U(1) = seed;
for n = 2:nu
    U(n) = mod(a*U(n-1)+b,M);  
end

U

% % Fibonacci generator
% for n = nu+1:nsample
%     U(n) = mod(U(n-nu)-U(n-mu),M);  
% end
% 
% U = U/M;


% %% Tests
% 
% % Bin the random variables in a histogram and normalise it
% nbins = 100;
% [h,x] = hist(U,100);
% h = h*nbins/nsample; % normalisation
% close all
% 
% % Plot of the probability density function
% figure(1)
% plot(x,h,x,ones(size(x)))
% xlim([-0.2 1.2])
% ylim([0 2])
% xlabel('x')
% ylabel('f')
% legend('Sampled','Theory')
% title('Uniform distribution')
% 
% % 2D scatter plot
% figure(2)
% plot(U(1:nsample-1),U(2:nsample),'.')
% xlabel('U_{i-1}')
% ylabel('U_i')
% title('Scatter plot')
% 
% % 3D scatter plot
% figure(3)
% plot3(U(1:nsample-2),U(2:nsample-1),U(3:nsample),'.')
% xlabel('U_i')
% ylabel('U_{i+1}')
% zlabel('U_{i+2}')
% title('3D scatter plot')