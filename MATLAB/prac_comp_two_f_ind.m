clc;
clearvars;
% close all;

load Y_1
load Y_2

Fs = 20e6;
Ts = 1/Fs;

lt = [1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 0,...
                1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1,...
                1 1 1 1];
lt = [lt(27:end) lt(1:26)];
lt(1) = [];

y1 = Y_1;
y2 = Y_2;

lt11 = y1(193:256);
lt21 = y1(257:320);

lt12 = y2(193:256);
lt22 = y2(257:320);

off1 = angle(sum(lt11.*conj(lt21)))/(pi*2*64*Ts);
off2 = angle(sum(lt12.*conj(lt22)))/(pi*2*64*Ts);

ang1 = exp(sqrt(-1)*(909382:909382+399)*2*pi*Ts*off1);
ang2 = exp(sqrt(-1)*(915909:915909+399)*2*pi*Ts*off2);

y1 = y1.*ang1;
y2 = y2.*ang2;



lt11 = y1(193:256);
lt21 = y1(257:320);

lt12 = y2(193:256);
lt22 = y2(257:320);



lt11 = fft(lt11);
lt11([1, 28:38]) = [];

lt21 = fft(lt21);
lt21([1, 28:38]) = [];



lt12 = fft(lt12);
lt12([1, 28:38]) = [];

lt22 = fft(lt22);
lt22([1, 28:38]) = [];



c11 = lt11./lt;
c21 = lt21./lt;
c1 = 0.5*(c11 + c21);




c12 = lt12./lt;
c22 = lt22./lt;
c2 = 0.5*(c12 + c22);


s1 = y1(337:end);
s2 = y2(337:end);


s1 = fft(s1);
s1([1, 28:38]) = [];

s2 = fft(s2);
s2([1, 28:38]) = [];


s1 = s1./c1;
s2 = s2./c2;


p1 = sum(s1([32 46 7 21])./[1 1 1 -1])/4;
s1 = s1.*conj(p1);
s1 = s1./abs(p1);

p2 = sum(s2([32 46 7 21])./[1 1 1 -1])/4;
s2 = s2.*conj(p2);
s2 = s2./abs(p2);


%%

r11 = s1(real(s1) > 0);
m11 = sum((real(r11) - 1).^-2);
r21 = s1(real(s1) < 0);
m21 = sum((real(r21) + 1).^-2);
m1 = .5*(m11 + m21)/sum(abs(s1).^2)


r12 = s1(real(s2) > 0);
m12 = sum((real(r12) - 1).^-2);
r22 = s1(real(s2) < 0);
m22 = sum((real(r22) + 1).^-2);
m2 = .5*(m12 + m22)/sum(abs(s2).^2)



%% Plots

m = 3;

subplot(m,2,1)
plot(y1, '.')
subplot(m,2,2)
plot(y2, '.')

subplot(m,2,3)
plot(unwrap(angle(c1)))
subplot(m,2,4)
plot(unwrap(angle(c2)))

subplot(m,2,5)
plot(s1, '.')
subplot(m,2,6)
plot(s2, '.')

% whos
















