% Check numerically the Fourier pair Laplace <-> Lorentzian

clear all

N = 1024;
Dt = 0.1;
t = Dt*(-N/2:N/2-1);
T = N*Dt;
% Nyquist relation: Dt*Dw = 2*pi/N
Dw = 2*pi/T;
w = Dw*(-N/2:N/2-1);
W = N*Dw; % = 2*pi/Dt

a = 1;
fa = a/2*exp(-a*abs(t)); % Laplace (or double exponential)
Fa = a^2./(a^2+w.^2); % Lorentzian (or Cauchy)

Fn  = fftshift(ifft(ifftshift(fa)))*T;
fn  = fftshift( fft(ifftshift(Fa)))/T;
Fn1 = fftshift( fft(ifftshift(fa)))*Dt;
fn1 = fftshift(ifft(ifftshift(Fa)))/Dt;

figure(1), clf, hold on
plot(t,real(fn),'r')
plot(t,imag(fn),'g')
plot(t,fa,'k:')
axis([-10 10 0 1])
xlabel('t')
ylabel('f')
legend('Re(fn)','Im(fn)','fa')

figure(2), clf, hold on
plot(w,real(Fn),'b')
plot(w,imag(Fn),'m')
plot(w,Fa,'c:')
axis([-20 20 0 1])
xlabel('\omega')
ylabel('F')
legend('Re(Fn)','Im(Fn)','Fa')

figure(3), clf, hold on
plot(t,real(fn1),'r')
plot(t,imag(fn1),'g')
plot(t,fa,'k:')
axis([-10 10 0 1])
xlabel('t')
ylabel('f')
legend('Re(fn1)','Im(fn1)','fa')

figure(4), clf, hold on
plot(w,real(Fn1),'b')
plot(w,imag(Fn1),'m')
plot(w,Fa,'c:')
axis([-20 20 0 1])
xlabel('\omega')
ylabel('F')
legend('Re(Fn1)','Im(Fn1)','Fa')