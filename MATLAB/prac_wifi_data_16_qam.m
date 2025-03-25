clc;
% clearvars;
close all;

load labviewWiFiData

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


%% FRAME SYNCHRONIZATION AND CFO COMPENSATION

len_y = length(Y);

%carrier offset and frame synchronization simultaneously
upper_limit = len_y - Mt_new - os;

comp_sig = zeros(1, upper_limit);
comp_sig_norm = comp_sig;
for k = 1:upper_limit
    
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

f_ind = f_ind - 2;

v_max = comp_sig(f_ind);
% plot(comp_sig_norm)

%CFO estimation
f_ind_LT = f_ind + Mt_new + 2*GI_len;

LT_frame_1 = Y(f_ind_LT:f_ind_LT + LT_len - 1);
LT_frame_2 = Y(f_ind_LT + LT_len:f_ind_LT + 2*LT_len - 1);

Y_long = sum(LT_frame_1.*conj(LT_frame_2));
f_offset = angle(Y_long)/(pi*2*LT_len*Ts);

LT_frame = [LT_frame_1; LT_frame_2];

%CFO compensation starting from LT frame
if size(Y, 2) == 1
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
rate_wifi = decoded(1:4)
reserved_bit = decoded(5);
len_OFDM_bin = decoded(17:-1:6);
len_OFDM = bin2dec(num2str(len_OFDM_bin));
parity_bit = decoded(18);
signal_tail = decoded(19:end);

%% Calculating total length of Data and extracting the signal
T_preamble = 16e-6;
T_signal = 4e-6;
data_rate = 24e6;
T_TX = T_preamble + T_signal + (16 + 8*len_OFDM + 6)/data_rate;
n_TX = T_TX*Fs;

no_OFDM_symbols = ceil((16 + 8*len_OFDM + 6)/96);
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
OFDM = OFDM_RF_po.*sqrt(10);

plot(OFDM, 'x')

%% Demodulation of Symbols to bits

data = zeros(4, 48*no_OFDM_symbols);

%Quantize the symbols
levels = -3:2:3;

OFDM_freq_re = real(OFDM(:));
OFDM_freq_im = imag(OFDM(:));

OFDM_re_diff = abs(OFDM_freq_re - levels);
OFDM_im_diff = abs(OFDM_freq_im - levels);

[~, OFDM_re_min_ind] = min(OFDM_re_diff, [], 2);
[~, OFDM_im_min_ind] = min(OFDM_im_diff, [], 2);

OFDM_re_q = levels(OFDM_re_min_ind);
OFDM_im_q = levels(OFDM_im_min_ind);

%Demodulate using encode table
QAM_16_en_tb = [0,0; 0,1; 1,1; 1,0]';

levels = repmat(levels.', 1, length(OFDM_re_q));

OFDM_re_q_mod = repmat(OFDM_re_q, size(levels, 1), 1);
OFDM_im_q_mod = repmat(OFDM_im_q, size(levels, 1), 1);

re_ind = find(OFDM_re_q_mod == levels);
im_ind = find(OFDM_im_q_mod == levels);

[re_ind, ~] = ind2sub(size(OFDM_re_q_mod), re_ind);
[im_ind, ~] = ind2sub(size(OFDM_im_q_mod), im_ind);

data(1:2,:) = QAM_16_en_tb(:, re_ind);
data(3:4,:) = QAM_16_en_tb(:, im_ind);

data_check_2 = data;

data = data(:);



%% Deinterleave DATA bits

data = reshape(data, Ndatasc*4, no_OFDM_symbols);

mod_typ = 4;
ncbps = Ndatasc*mod_typ;
ncpc = 4;


k = 0:ncbps - 1;
s = ceil(ncpc/2);

%First de-permutation of de-interleaver
i = s*floor(k/s) + mod(k + floor(16*k/ncbps), s);

%second de-permutation of de-interleaver
j = 16*i - (ncbps - 1)*floor(16*i/ncbps);

data_dei(j + 1,:) = data(k + 1,:);


%% Viterbi decoding

trellis = poly2trellis(7, [133 171]);
decoder_output = vitdec(data_dei(:)', trellis, 35, 'term', 'hard');

size(decoder_output);


%% Descrambler

decoder_output = decoder_output(:)';

init = (decoder_output(1:7));
seq = zeros(1, 127);

for i = 1:127
    
    init_4 = init(4);
    init_7 = init(1);
    
    init_4_7_xor = xor(init_7, init_4);
    
    seq(i) = init_4_7_xor;
    
    init(1:end-1) = init(2:end);
    init(7) = init_4_7_xor;
    
end


N = length(decoder_output) - 7;
n = floor(N/127);
n_extra_bits = N - n*127;

seq_repeated = [init repmat(seq, 1, n) seq(1:n_extra_bits)];

data_est = bitxor(decoder_output, seq_repeated);




%% Extract data fields

n_data = length(data_est);
n_pad = n_data - (16 + 8*len_OFDM + 6);

% Pad bits
bits_padded = data_est(end - n_pad + 1:end)
data_est(end - n_pad + 1:end) = [];

% Tail bits
bits_tail = data_est(end - 5:end)
data_est(end - 5:end) = [];

% Service bits
bits_service = data_est(1:16)
data_est(1:16) = [];

data_est

data_bin = logical(reshape(data_est, 8, len_OFDM)).';
data_hex = binaryVectorToHex(data_bin, 'LSBfirst');
data_hex = cell2mat(data_hex)';
data_hex = data_hex(:)';


%CRC extraction
CRC_rec = fliplr(flipud(reshape(data_est(end-31:end), 4, 8)'));
CRC_rec = binaryVectorToHex(CRC_rec);
CRC_rec = cell2mat(CRC_rec)'




%% CRC-32

poly = [1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1];
len_poly = length(poly);

data_hex_no_crc = data_hex(1:end - 8);
msg = hexToBinaryVector(data_hex_no_crc);

%Processing Message for CRC calculation
data = [zeros(1, ceil(length(msg)/8)*8-length(msg)) msg];
data = fliplr(reshape(data, 8, length(data)/8)');
data = data';
data = data(:)';
data = [data zeros(1, len_poly - 1)];

%XORing first 32 bits with 0xFFFFFFFF
data = [xor(data(1:32), ones(1, 32)) data(33:end)];

%Binary division
xor_with_poly = bitxor(data(1:len_poly), poly);
data(1:len_poly) = [];

stop = 0;
while(~stop)
    
    cur_data_len = length(data);
    num_zeros_xor = find(xor_with_poly == 1, 1) - 1;
    
    if ~sum(xor_with_poly)
        num_zeros_xor = length(xor_with_poly);
    end
    
    xor_with_poly(1:num_zeros_xor) = [];
    
    len_xor = length(xor_with_poly);
    diff_poly_xor = len_poly - len_xor;
    
    if diff_poly_xor > cur_data_len
        
        xor_with_poly = [xor_with_poly data];
        stop = 1;
        break
        
    else
                
        append_xor = data(1:diff_poly_xor);
        new_xor = [xor_with_poly append_xor];

    end
    
    xor_with_poly = bitxor(poly, new_xor);
    data(1:diff_poly_xor) = [];
    
end

CRC = xor_with_poly;
CRC = [zeros(1, len_poly - length(CRC) - 1) CRC];

%XORing CRC with 0xFFFFFFFF and reflecting
CRC = fliplr(xor(CRC, ones(1, 32)));
% CRC = reshape(CRC, 4, 8)';

CRC_hex_calc = binaryVectorToHex(CRC);







%% Plots
% 
% subplot(311)
% plot(real(Y))
% axis tight
% title('Real value of received signal')
% 
% subplot(312)
% plot(corr_sig)
% axis tight
% title('Correlation signal')
% 
% subplot(313)
% plot(ERP_freq_RF_po, 'o', 'LineWidth', 3)
% axis tight
% title('Power spectrum of received signal')











%% MATLAB Toolbox

nht = wlanNonHTConfig('MCS', 4, 'PSDULength', len_OFDM)

rxPPDU = Y.';

%% Receiver
fieldInd = wlanFieldIndices(nht)
fieldInd.LSTF = fieldInd.LSTF + f_ind;
fieldInd.LLTF = fieldInd.LLTF + f_ind;
fieldInd.LSIG = fieldInd.LSIG + f_ind;
fieldInd.NonHTData = fieldInd.NonHTData + f_ind


numSamples = fieldInd.LSIG(2);

rxLLTF = rxPPDU(fieldInd.LLTF(1):fieldInd.LLTF(2),:);
demodLLTF = wlanLLTFDemodulate(rxLLTF, nht);
chEstLLTF = wlanLLTFChannelEstimate(demodLLTF, nht);

rxLSIG = rxPPDU(fieldInd.LSIG(1):fieldInd.LSIG(2),:);
recLSIG = wlanLSIGRecover(rxLSIG, chEstLLTF, 1, 'CBW20');
rate = recLSIG(1:4)'

cfgRec = wlanRecoveryConfig('EqualizationMethod', 'ZF');

rxPSDU = rxPPDU(fieldInd.NonHTData(1):fieldInd.NonHTData(2),:);
[recPSDU, ~, eqSym] = wlanNonHTDataRecover(rxPSDU, chEstLLTF, 1, nht, cfgRec);


rec_bin = logical(reshape(recPSDU, 8, 276)).';
rec_hex = binaryVectorToHex(rec_bin, 'LSBfirst');
rec_hex = cell2mat(rec_hex)';
rec_hex = rec_hex(:)';
[rec_hex(65:65+7); rec_hex(73:80); rec_hex(81:88)]



numErr = biterr(data_est.', recPSDU)

check = [data_est.' recPSDU (data_est.' ~= recPSDU)];
min(find((check(:,3) == 1)))
stem(check(:,3))
axis tight
















%% CRC CHECK
% 
% recPSDU = logical(recPSDU');
% 
% poly = ([1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1]);
% 
% % poly = fliplr(flipud(reshape(poly(2:end), 8, 4)'));
% % poly = [1 poly(:)']
% 
% data = recPSDU(1:end-32);
% 
% len_poly = length(poly);
% 
% data_temp = [data zeros(1, len_poly-1)];
% 
% CRC_1 = crc_custom(data_temp, poly);
% 
% CRC_1 = [zeros(1, len_poly - length(CRC_1) - 1) CRC_1]
% CRC_check = ~sum(crc_custom(recPSDU, poly))
% 
% 
% 
% 
% function CRC = crc_custom(data_temp, poly)
% 
% len_poly = length(poly);
% xor_with_poly = bitxor(data_temp(1:len_poly), poly);
% data_temp(1:len_poly) = [];
% 
% stop = 0;
% while(~stop)
%     cur_data_len = length(data_temp);
%     num_zeros_xor = min(find(xor_with_poly == 1)) - 1;
%     
%     if ~sum(xor_with_poly)
%         num_zeros_xor = length(xor_with_poly);
%     end
%     
%     xor_with_poly(1:num_zeros_xor) = [];
%     
%     len_xor = length(xor_with_poly);
%     diff_poly_xor = len_poly - len_xor;
%     
%     if diff_poly_xor > cur_data_len
%         
%         xor_with_poly = [xor_with_poly data_temp];
%         stop = 1;
%         break
%         
%     else
%                 
%         append_xor = data_temp(1:diff_poly_xor);
%         new_xor = [xor_with_poly append_xor];
% 
%     end
%     
%     xor_with_poly = bitxor(poly, new_xor);
%     data_temp(1:diff_poly_xor) = [];
%     
% end
% 
% CRC = xor_with_poly;
% 
% end


