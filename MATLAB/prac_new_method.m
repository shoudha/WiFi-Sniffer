clc;
clearvars;
% close all;


% load 'Collected Data_Current\wifiData_index_184091_code_2_3_54qam'
% load 'Collected Data_Current\wifiData_index_751455_code_3_4_54qam'
% load 'Collected Data_Current\wifiData_index_915909_code_3_4_54qam'

% load wiFiData

% load labviewWiFiData

% load labviewWiFiDataNew



if size(Y, 1) ~= 1
    Y = Y.';
end

%% Normalizing the signal

Y = Y./max(abs(Y));

%% Determinig signal positions

ro = 0.9;
p = zeros(size(Y));

len_p = length(p);
p(1) = abs(Y(1))^2;
for i = 2:len_p
    
    p(i) = (1 - ro)*abs(Y(i))^2 + ro*p(i - 1);
    
end

%Selected threshold
thr = 0.003;
margin = thr*ones(size(p));
m = p - margin;
m = -(0.5)*[sign(m(1:end-1)) - sign(m(2:end)) 0];

%main index pointer
sig_start = find(m == 1);
sig_end = find(m == -1);

if sig_end(1) < sig_start(1)
    sig_end(1) = [];
end

% length(sig_start)
% length(sig_end)

% Siscard signals smaller than 400 samples or 400*Ts seconds
sig_duration = sig_end - sig_start;
sig_start(sig_duration < 400) = [];
sig_end(sig_duration < 400) = [];
sig_duration(sig_duration < 400) = [];

% length(sig_start)
% length(sig_end)

% m_new = zeros(ones(size(p)));
% m_new(sig_start) = 1;
% m_new(sig_end) = -1;





%% Apply AC
if sig_start(1) > 10
    upper_limit = sig_start(1) - 10
else
    upper_limit = 1;
end
    
lower_limit = sig_end(1)


Y_temp = Y(upper_limit:lower_limit);


len_y = length(Y_temp);
ac_bound = len_y - 160 - 1;
comp_sig = zeros(1, ac_bound);
comp_sig_norm = comp_sig;


for k = 1:ac_bound
    
    frame = Y_temp(k:k + 160 - 1);

    f1 = frame(1:144);
    f2 = frame(17:160);
    
    comp_sig(k) = sum((f1).*conj(f2));
    comp_sig_norm(k) = abs(comp_sig(k)).^2/(160*sum(abs(frame).^2));
       
end

[~, f_ind_temp] = max(comp_sig_norm)

f_ind = upper_limit + f_ind_temp - 1





plot(comp_sig_norm)




























%% Plots
% pl = 3;
% subplot(pl,1,1)
% plot(real(Y))
% hold on
% plot(m, 'r')
% hold off
% 
% 
% subplot(pl,1,2)
% plot(real(Y))
% hold on
% plot(m_new, 'r')
% hold off
% 
% 
% subplot(pl,1,3)
% plot(p)
% hold on
% plot(margin)
% hold off

