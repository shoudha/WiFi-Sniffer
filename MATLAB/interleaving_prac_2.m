

%% SETTING UP parameters required for Interleaver module to function 
clc;
clearvars;

Ndatasc = 48; % Number of data subcarriers in the IFFT symbol
mod_type = input('Enter the modulation type[1 for BPSK,2 for QPSK,4 for 16QAM,6 for 64QAM]: ');
ncbps = Ndatasc*mod_type; %Number of coded bits per symbol, based on number of data carriers in OFDM symbol
ncpc = mod_type; %Number of coded bits per carrier
x = randi([0, 1], [ncbps, 1]); % generating binary data ones and zeros
interleaver_input = x'; % Interleaver binary input

%% Interleaver MATLAB Code
% Interleaver PART
s = ceil(ncpc/2);
k = 0:ncbps-1;
% 
% %First permutation of interleaver
m = (ncbps/16)*mod(k,16)+floor(k/16);
% %Second permutation of interleaver
n = s*floor(m/s)+mod(m+ncbps-floor(16*m/ncbps),s);
% interleaved_data_out(n+1) = interleaver_input(k+1); % OUTPUT of interleaver

%% Deinterleaver MATLAB Code
% Deinterleaver PART
% k = 0:ncbps - 1;
% s = ceil(ncpc/2);
% %First de-permutation of de-interleaver
% i = s*floor(k/s)+mod(k+floor(12*k/ncbps),s);
% %second de-permutation of de-interleaver
% j = 12*i - (ncbps - 1)*floor(12*i/ncbps);
% deinterleaver_data_out(e + 1) = interleaved_data_out(j + 1);

%% OUTPUT Plots in MATLAB
% Plotting data
% subplot(4,1,1),plot(1:length(interleaver_input),interleaver_input);
% title('INTERLEAVER INPUT');xlabel('binary input index');ylabel('binary data');
% subplot(4,1,2),plot(1:length(interleaved_data_out),interleaved_data_out);
% title('INTERLEAVER OUTPUT');xlabel('binary input index');ylabel('binary data');
% subplot(4,1,3),plot(1:length(deinterleaver_data_out),deinterleaver_data_out);
% title('DE-INTERLEAVER OUTPUT');xlabel('binary input index');ylabel('binary data');
% subplot(4,1,4),plot(1:length(deinterleaver_data_out),(interleaver_input-deinterleaver_data_out));
% title('Difference between Interleaver-IN and Deinterleaver-OUT/');


table(k', m')



