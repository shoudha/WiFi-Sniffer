clc;
clearvars;
% close all;

load test_OFDM_data
% load wiFiData

% 
Y = I_OFDM1 + (sqrt(-1))*Q_OFDM1;
% p = 10*log10(I_OFDM1.^2 + Q_OFDM1.^2);


%% Parameters

%Lookup table
rate_tb = [1101; 1111; 0101; 0111; 1001; 1011; 0001; 0011];
data_rate_tb = [6, 9, 12, 18, 24, 36, 48, 54]';
modulation = [{'BPSK'}, {'BPSK'}, {'QPSK'},{'QPSK'},{'16-QAM'},{'16-QAM'},{'64-QAM'},{'64-QAM'}]';
coding_rates = [1/2, 3/4, 1/2, 3/4, 1/2, 3/4, 2/3, 3/4]';
nbpsc_tb = [1 1 2 2 4 4 6 6]';
ncbps_tb = [48, 48, 96, 96, 192, 192, 288, 288]';
ndbps_tb = ncbps_tb.*coding_rates;

wifi_tb = table(rate_tb,...
                data_rate_tb,...
                modulation,...
                coding_rates,...
                nbpsc_tb,...
                ncbps_tb,...
                ndbps_tb);
wifi_tb.Properties.VariableNames = {'Rate',...
                                    'Data_Rate',...
                                    'Modulation',...
                                    'Coding_Rate',...
                                    'NBPSC',...
                                    'NCBPS',...
                                    'NDBPS'};

%Receiver Parameters
Fs = 20e6;
Ts = 1/Fs;
os = 1;
Mt = 64;
Nt = 2*Mt;
Mt_new = 160;
offset_new = 16;

%Frame lengths in time
LT_len_t = 3.2e-6;
GI_len_t = .8e-6;
sig_len_t = 3.2e-6;
sig_off_t = 7.2e-6;
ofdm_sym_t = 4e-6;

%Frame lengths in sample
LT_len = round(LT_len_t*Fs);
GI_len = round(GI_len_t*Fs);
sig_len = round(sig_len_t*Fs);
sig_off = round(sig_off_t*Fs);
ofdm_sym = round(ofdm_sym_t*Fs);
len_ofdm = 80;

%Pilot Polarity
pilot_polarity = [  1 1 1 1,...
                    -1 -1 -1 1,...
                    -1 -1 -1 -1,...
                    1 1 -1 1,...
                    -1 -1 1 1,...
                    -1 1 1 -1,...
                    1 1 1 1,...
                    1 1 -1 1,...
                    1 1 -1 1,...
                    1 -1 -1 1,...
                    1 1 -1 1,...
                    -1 -1 -1 1,...
                    -1 1 -1 -1,...
                    1 -1 -1 1,...
                    1 1 1 1,...
                    -1 -1 1 1,...
                    -1 -1 1 -1,...
                    1 -1 1 1,...
                    -1 -1 -1 1,...
                    1 -1 -1 -1,...
                    -1 1 -1 -1,...
                    1 -1 1 1,...
                    1 1 -1 1,...
                    -1 1 -1 1,...
                    -1 -1 -1 -1,...
                    -1 1 -1 1,...
                    1 -1 1 -1,...
                    1 1 1 -1,...
                    -1 1 -1 -1,...
                    -1 1 1 1,...
                    -1 -1 -1 -1,...
                    -1 -1 -1]';
                
pilot_wifi = [1 1 1 -1]';

%BPSK Signal Field
signal_52 = [-1 1 1 -1 1 1 -1 -1 -1 1 -1 -1 1 1 1 1 1 -1 -1 1 1,...
            -1 -1 -1 1 1 1 1 1 -1 -1 -1 1 -1 1 1 -1 1 1 -1 -1 1,...
            -1 1 -1 -1 -1 -1 1 -1 1 1];
% signal_52 = [signal_52(27:end) signal_52(1:26)];

%Given BSSID
BSSID_check = [{'1'} {'2'} {'0'} {'1'} {'0'} {'B'} {'5'} {'1'} {'B'} {'F'} {'5'} {'7'}];
BSSID_check_bin = hexToBinaryVector(BSSID_check);
BSSID_check_bin = BSSID_check_bin';
BSSID_check_bin = BSSID_check_bin(:)';


%% FRAME SYNCHRONIZATION AND CFO COMPENSATION

len_y = length(Y);

%carrier offset and frame synchronization simultaneously
upper_limit = len_y - Mt_new - os;

comp_sig = zeros(1, upper_limit);
comp_sig_norm = comp_sig;
for k = 1:upper_limit
    
%     Frame Synch using Long Preamble
%     frame = r(k:k + 2*Mt*os - 1);
%     half_1 = frame(1:os*Mt);
%     half_2 = frame(os*Mt + 1:end);
%     comp = half_1(1:os:end).*conj(half_2(1:os:end));
%     comp_sig(k) = sum(comp);
%     comp_sig(k) = abs(comp_sig(k)).^2./(length(frame)*sum(abs(frame).^2));

    %Frame Synch using Short Preamble
    frame = Y(k:k + Mt_new - 1);
    half_1 = frame(1:Mt_new - offset_new);
    half_2 = frame(offset_new + 1:end);
    comp = (half_1).*conj(half_2);
    comp_sig(k) = sum(comp);
    comp_sig_norm(k) = abs(comp_sig(k)).^2./(160*sum(abs(frame).^2));
 
end

%correlation signal
corr_sig = comp_sig_norm;

%Short preamble frame starting point
[maOFDM_corr, f_ind] = max(comp_sig_norm);

f_ind = f_ind - 1;

v_max = comp_sig(f_ind);
plot(comp_sig_norm)

%CFO estimation
f_ind_LT = f_ind + Mt_new + 2*GI_len;

LT_frame_1 = Y(f_ind_LT:f_ind_LT + LT_len - 1);
LT_frame_2 = Y(f_ind_LT + LT_len:f_ind_LT + 2*LT_len - 1);

Y_long = sum(LT_frame_1.*conj(LT_frame_2));
f_offset = angle(Y_long)/(pi*2*LT_len*Ts);

LT_frame = [LT_frame_1; LT_frame_2];

%CFO compensation starting from LT frame
if size(Y,2) == 1
    r_sig_SP = Y(f_ind_LT:end);
else
    r_sig_SP = Y(f_ind_LT:end).';
end
    
f_offset_angle = exp(sqrt(-1)*(f_ind_LT:len_y)...
                    *2*pi*Ts*f_offset).';
r_sig_CFO = r_sig_SP.*f_offset_angle;

LT_frame_1 = r_sig_CFO(1:64);
LT_frame_2 = r_sig_CFO(65:128);

%Channel estimation from LT frame
LT_frame_org = [1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 0,...
                1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1,...
                1 1 1 1];
LT_frame_re = [LT_frame_org(27:end) LT_frame_org(1:26)]';
LT_frame_re(1) = [];

%Making 52 length LT frames
LT_frame_1_52 = fft(LT_frame_1);
LT_frame_1_52([1, 28:38]) = [];

LT_frame_2_52 = fft(LT_frame_2);
LT_frame_2_52([1, 28:38]) = [];

%Estimating channel coefficients
channel_coeff_LT_1 = LT_frame_1_52./LT_frame_re;
channel_coeff_LT_2 = LT_frame_2_52./LT_frame_re;
channel_coeff_LT = 0.5*(channel_coeff_LT_1 + channel_coeff_LT_2);

%% EXTRACTING SIGNAL BLOCK
sig_ind = f_ind_LT + sig_off;
ERP = r_sig_CFO(sig_off + 1:sig_off + sig_len);
ERP_freq = fft(ERP);

%Removing Null subcarriers
ERP_freq([1, 28:38]) = [];

%% RF AND PHASE OFFSET COMPENSATION TO SIGNAL BLOCK

%RF compensation
ERP_freq_RF = ERP_freq./channel_coeff_LT;

%Phase offset compensation
ERP_pilot = [1 1 1 -1]';
ERP_freq_pilot = ERP_freq_RF([32 46 7 21]);
sig_phase = sum(ERP_freq_pilot./ERP_pilot)/4;
ERP_freq_RF_po = ERP_freq_RF.*conj(sig_phase);
ERP_freq_RF_po = ERP_freq_RF_po./abs(sig_phase);

%% Rearranging the signal from [1-52] to [-26-26]
ERP_freq_RF_po = [ERP_freq_RF_po(27:end).' ERP_freq_RF_po(1:26).'];

% plot(ERP_freq_RF_po, 'o', 'LineWidth', 3)

%% Demodulation of BPSK symbols to bit
ERP_th = real(ERP_freq_RF_po);
ERP_th(ERP_th > 0) = 1;
ERP_th(ERP_th < 0) = 0;

%% Checking with BPSK original
% signal_52(signal_52 == -1) = 0;
% error_bits_signal = sum(ERP_th ~= signal_52)

%% DE-INTERLEAVER

%Removing the pilots
ERP_th([6 20 33 47]) = [];

%Deinterleaver
%Parameters
Ndatasc = 48;
ncbps = Ndatasc*1;
ncpc = 1;

s = max(ncpc/2, 1);
j = 0:ncbps - 1;

%First de-permutation of de-interleaver
i = s*floor(j/s) + mod(j + floor(16*j/ncbps), s);

%second de-permutation of de-interleaver
k = 16*i - (ncbps - 1)*floor(16*i/ncbps);

ERP_dei = zeros(1, length(k));
ERP_dei(k + 1) = ERP_th(j + 1);

%% VITERBI DECODER
trellis = poly2trellis(7, [133 171]);
decoded = vitdec(ERP_dei, trellis, 1, 'trunc', 'hard');

%Parity check
parity_even = ~mod(sum(decoded(1:end - 6)), 2);
disp(char(parity_even*'Parity is Even.' +...
         ~parity_even*'Parity is Odd. '));

len_dec = length(decoded);
rate = decoded(1:4)
reserved_bit = decoded(5);
len_OFDM_bin = decoded(17:-1:6);
len_OFDM = bin2dec(num2str(len_OFDM_bin))
parity_bit = decoded(18);
signal_tail = decoded(19:end);

%% Calculating total length of Data and extracting the signal
T_preamble = 16e-6;
T_signal = 4e-6;
data_rate = 54e6;
T_TX = T_preamble + T_signal + (16 + 8*len_OFDM + 6)/data_rate;
n_TX = T_TX*Fs;

no_OFDM_symbols = ceil((16 + 8*len_OFDM + 6)/216);
% no_OFDM_symbols = ceil(n_TX/80);
n_TOFDM_r = no_OFDM_symbols*ofdm_sym;
ind_end = f_ind + no_OFDM_symbols*ofdm_sym;

%Partial frame detection - for future use
partial_frame = len_y < ind_end;
disp(char(partial_frame*'Detected Frame is incomplete.' +...
         ~partial_frame*'Detected Frame is complete.  '));


%% Extracting the OFDM symbols
data_ind = 2*LT_len + len_ofdm + 1;
val_OFDM = r_sig_CFO(data_ind:data_ind + no_OFDM_symbols*len_ofdm - 1);
val_OFDM = reshape(val_OFDM, len_ofdm, no_OFDM_symbols);

%% Constellation of the DATA OFDM blocks

%Removing CP
val_OFDM(1:16, :) = [];

%Taking FFT
OFDM_freq = fft(val_OFDM);

%Removing Null subcarriers
OFDM_freq([1, 28:38],:) = [];

%RF compensation
OFDM_RF = OFDM_freq./channel_coeff_LT;

%Phase offset compensation
pilot_wifi_rep = repmat(pilot_wifi, 1, no_OFDM_symbols);
pilot_polarity_rep = repmat(pilot_polarity(2:no_OFDM_symbols + 1).', 4, 1);
pilot_OFDM = pilot_wifi_rep.*pilot_polarity_rep;
OFDM_pilot = OFDM_RF([32 46 7 21],:);
OFDM_phase = sum(OFDM_pilot./pilot_OFDM)/4;
OFDM_RF_po = OFDM_RF.*conj(OFDM_phase);
OFDM_RF_po = OFDM_RF_po./abs(OFDM_phase);


%% Rearranging the data from [1-52] to [-26-26]
OFDM_RF_po = [OFDM_RF_po(27:end,:); OFDM_RF_po(1:26,:)];

%Removing the pilots
OFDM_RF_po([6 20 33 47], :) = [];

%Normalize
OFDM = OFDM_RF_po.*sqrt(42);

% plot(OFDM, 'x')

%% Demodulation of Symbols to bits

data = zeros(6, 48*no_OFDM_symbols);

%Quantize the symbols
levels = -7:2:7;

OFDM_freq_re = real(OFDM(:));
OFDM_freq_im = imag(OFDM(:));

OFDM_re_diff = abs(OFDM_freq_re - levels);
OFDM_im_diff = abs(OFDM_freq_im - levels);

[~, OFDM_re_min_ind] = min(OFDM_re_diff, [], 2);
[~, OFDM_im_min_ind] = min(OFDM_im_diff, [], 2);

OFDM_re_q = levels(OFDM_re_min_ind);
OFDM_im_q = levels(OFDM_im_min_ind);

%Demodulate using encode table
QAM_16_en_tb = [0,0,0; 0,0,1; 0,1,1; 0,1,0; 1,1,0; 1,1,1; 1,0,1; 1,0,0]';

levels = repmat(levels.', 1, length(OFDM_re_q));

OFDM_re_q_mod = repmat(OFDM_re_q, size(levels, 1), 1);
OFDM_im_q_mod = repmat(OFDM_im_q, size(levels, 1), 1);

re_ind = find(OFDM_re_q_mod == levels);
im_ind = find(OFDM_im_q_mod == levels);

[re_ind, ~] = ind2sub(size(OFDM_re_q_mod), re_ind);
[im_ind, ~] = ind2sub(size(OFDM_im_q_mod), im_ind);

data(1:3,:) = QAM_16_en_tb(:, re_ind);
data(4:6,:) = QAM_16_en_tb(:, im_ind);

data = data(:);

%% Deinterleave DATA bits

data = reshape(data, Ndatasc*6, no_OFDM_symbols);

mod_typ = 6;
ncbps = Ndatasc*mod_typ;
ncpc = 6;

k = 0:ncbps - 1;
s = ceil(ncpc/2);

%First de-permutation of de-interleaver
i = s*floor(k/s) + mod(k + floor(16*k/ncbps), s);

%second de-permutation of de-interleaver
j = 16*i - (ncbps - 1)*floor(16*i/ncbps);

data_dei(j + 1,:) = data(k + 1,:);

%% Puncturing and Viterbi decoding

decoder_input = data_dei;
cols = numel(decoder_input)/12;
decoder_input = reshape(decoder_input(:), 12, cols);

%Decode
trellis = poly2trellis(7, [133 171]);
puncpat = [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1];
decoder_output = vitdec(decoder_input(:), trellis, 35, 'trunc', 'hard', puncpat);
decoder_output = reshape(decoder_output, 9, cols);

size(decoder_output);
% decoder_output = fliplr(decoder_output);

%% Descramble data symbol

min_val = Inf;
for shamman = 47:47

data_scrambled = decoder_output(:)';
data_scrambled(1:7) = [];

b = str2double(dec2bin(shamman));
c = str2double(regexp(num2str(b), '\d', 'match'));

len_pad = 7 - length(c);
init_state = [zeros(1, len_pad) c];
    
n = length(data_scrambled);
data_descrambled = zeros(1, n);
for i = 1:n
    init_state_4 = init_state(4);
    init_state_7 = init_state(7);
    data_descrambled(i) = xor(data_scrambled(i), xor(init_state_4, init_state_7));
    init_state = circshift(init_state, 1);
    init_state(1) = xor(init_state_4, init_state_7);
end

bits_service = data_descrambled(1:7);

error_bits = sum(bits_service);

if error_bits < min_val
    min_val = error_bits;
    state = shamman;
    data_saved = data_descrambled;
end

end

descrambler_init_state = state
service_bit_summation = min_val

data_est = data_saved;

%% Extract data fields
% n_data = length(data_est);
% n_pad = n_data - (16 + 8*len_OFDM + 6);
% 
% % Pad bits
% bits_padded = data_est(end - n_pad + 1:end)
% data_est(end - n_pad + 1:end) = [];
% 
% % Tail bits
% bits_tail = data_est(end - 5:end)
% data_est(end - 5:end) = [];
% 
% % Service bits
% bits_service = data_est(1:16)
% data_est(1:16) = [];
% 
% 
% %Protocol bits
% data_hex_dump = binaryVectorToHex(data_est)
% 
% data_hex_dump2 = reshape(data_hex_dump, length(data_hex_dump)/2, 2);
% data_dec = hex2dec(data_hex_dump2);
% data_char = char(data_dec)'
% 
% 
% % data_est(1:16)
% % data_est(17:32)
% % data_est(33:80)
% % data_est(81:128)
% % bits_protocol = data_est(1:2)
% 
% %Frame bits
% % data_est(1:16) = fliplr(data_est(1:16));
% frame_type = data_est(3:4)
% frame_subtype = data_est(5:8)
% 
% %Duration bits
% duration = data_est(17:17+16-1)
% duration = reshape(duration, 4, 4)';
% duration = binaryVectorToHex(duration)';
% 
% %Destination Address
% dest_add = data_est(33:33+48-1);
% 
% dest_add = reshape(dest_add, 4, 12)';
% % dest_add = binaryVectorToHex(dest_add)';
% 
% %Source Address
% src_add = data_est(81:81+48-1);
% sum(BSSID_check_bin == src_add)*100/48
% 
% 
% src_add = reshape(src_add, 4, 12)';
% % src_add = binaryVectorToHex(src_add)';
% 
% %BSS ID
% bss_id = data_est(129:129+48-1);
% 
% bss_id = reshape(bss_id, 4, 12)';
% % bss_id = binaryVectorToHex(bss_id)';
% 
% data_char = reshape(data_descrambled, 8, numel(data_descrambled)/8).';
% data_char = binaryVectorToHex(data_char)';
% data_char = hex2dec(data_char);
% data_char = char(data_char)';


%%

% 
% % test = [0 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 1 1 0 1 0 0 0 1 1 0 0 1 1 0 0];
% % test = [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
% test = [0 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 1 1 0 1 0 0 0 1 1 0 0 1 1 0 0 1 1 1 0 0 1 0 1 0 1 0 0 0 0 1 0];
% r = data_est;
% len_sig = length(data_est);
% Mt_new = length(test);
% upper_limit = len_sig - Mt_new - 1;
% 
% t1 = zeros(1, upper_limit);
% t2 = t1;
% for k = 1:upper_limit
% 
%     %Frame Synch using Short Preamble
%     frame = r(k:k + Mt_new - 1);
%     comp = (frame == test);
%     t1(k) = sum(comp);
%     
%     s2 = sum(frame);
%     
%     if s2 ~= 0
%         t2(k) = abs(t1(k)).^2./(Mt_new*s2.^2);
%     end
% end
% 
% close all
% plot(t2)
% [v2, i2] = max(t2)
% data_hex_dump = binaryVectorToHex(data_est(i2:end))


%% Plots

subplot(311)
plot(real(Y))
axis tight
title('Real value of received signal')

subplot(312)
plot(corr_sig)
axis tight
title('Correlation signal')

subplot(313)
plot(ERP_freq_RF_po, 'o', 'LineWidth', 3)
axis tight
title('Power spectrum of received signal')











