clc;
clearvars;
% close all;


load 'Collected Data_Current\wifiData_index_915909_code_3_4_54qam'

if size(Y, 1) ~= 1
    Y = Y.';
end


Fs = 20e6;
Ts = 1/Fs;

lt = [1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 0,...
                1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1,...
                1 1 1 1];
lt = [lt(27:end) lt(1:26)];
lt(1) = [];

load lt

%% FRAME SYNCHRONIZATION

len_y = length(Y);

upper_limit = len_y - 320 - 1;
comp_sig = zeros(1, upper_limit);
comp_sig_norm = comp_sig;


parfor k = 1:upper_limit
    
    upper_limit - k
    frame = Y(k:k + 320 - 1);

    f1 = frame(1:144);
    f2 = frame(17:160);
    
    comp_sig(k) = sum((f1).*conj(f2));
    comp_sig_norm(k) = abs(comp_sig(k)).^2/(160*sum(abs(frame).^2));
       
end

% MPH = 0.5*1e-4;
% [~, ind] = findpeaks(comp_sig_norm, 'MinPeakHeight', MPH);
% len_pc = length(ind);
% 
% ind_rem = zeros(size(ind));
% 
% ind_diff = ind - [ind(2:end) 0];
% 
% 
% parfor i = 1:len_pc
%     
%     len_pc - i
%     if ind_diff(i) < 4255
%         ind_rem(i) = ind(i);
%         ind(i) = 0;
%     end
%     
% end
% 
% 
% c = find(ind_rem == 0, 1);
% ind_rem(c:end) = [];
% 
% comp_sig_norm(ind_rem) = -comp_sig_norm(ind_rem);

% [c_up, c_low] = envelope(comp_sig_norm);

[~, f_ind] = max(comp_sig_norm)

figure
plot(comp_sig_norm)
hold on
a = 1:length(comp_sig_norm);
plot(a(915909), comp_sig_norm(915909), '*r')
hold off


Y_2 = Y(909384:909384+399);
Y_1 = Y(915909:915909+399);

