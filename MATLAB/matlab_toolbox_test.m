clc;
clearvars;
% close all;

no_OFDM_symbols = 58;
len_OFDM = 1558;

cfg = wlanNonHTConfig('Modulation', 'OFDM', 'MCS', 7);
cfg.PSDULength = len_OFDM;

psdu = randi([0 1], cfg.PSDULength*8, 1);




SCRAMINIT = 126;
y = wlanNonHTData(psdu, cfg, SCRAMINIT);
size(y);

%% Channel effect


snr = 25;
y = awgn(y, snr);


%% Received signal

y = reshape(y, 80, no_OFDM_symbols);

%Removing CP
y(1:16, :) = [];

y_freq = fft(y);

%Removing Null subcarriers
y_freq([1, 28:38],:) = [];

%Rearranging the signal from [1-52] to [-26-26]
y_freq = [y_freq(27:end,:); y_freq(1:26,:)];

%Removing the pilots
y_freq([6 20 33 47], :) = [];

%Normalize
% y_freq = y_freq.*sqrt(42);
scaling = 7/max(max(real(y_freq)));
OFDM = y_freq.*scaling;

plot(OFDM, 'x')


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

Ndatasc = 48;

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

size(data_dei);


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


%% Descramble data symbol

data_scrambled = decoder_output(:)';




% min_val = 10e12;
% 
% for shamman = 1:127
% 
% b = str2num(dec2bin(shamman));
% c = str2double(regexp(num2str(b),'\d','match'));
% 
% len_pad = 7 - length(c);
% c = [zeros(1, len_pad) c];

init_state = [0 1 1 1 1 1 1];
% init_state = c;
    
n = length(data_scrambled);
data_descrambled = zeros(1, n);
for i = 1:n
    init_state_4 = init_state(4);
    init_state_7 = init_state(7);
    data_descrambled(i) = xor(data_scrambled(i), xor(init_state_4, init_state_7));
    init_state = circshift(init_state, 1);
    init_state(1) = xor(init_state_4, init_state_7);
end

data_est = data_descrambled;

size(data_est);



%% Extract data fields

n_data = length(data_est);
n_pad = n_data - (16 + 8*len_OFDM + 6);

%Pad bits
bits_padded = data_est(end - n_pad + 1:end);
data_est(end - n_pad + 1:end) = [];

%Tail bits
bits_tail = data_est(end - 5:end);
data_est(end - 5:end) = [];

%Service bits
bits_service = data_est(1:16);
data_est(1:16) = [];

check_service_bits = any(bits_service)

error_bits = sum(data_est ~= psdu')
% 
% 
% if error_bits < min_val
%     min_val = error_bits;
%     state = shamman;
%     service_at_min = sum(bits_service);
% end
% 
% end
% min_val
% service_at_min
% init_state = SCRAMINIT
% opt_state = state

%Protocol bits
bits_protocol = data_est(1:2);

%Frame bits
frame_type = data_est(3:4);
frame_subtype = data_est(5:8);

%Duration bits
duration = data_est(17:17+16-1);
duration = reshape(duration, 4, 4)';
duration = binaryVectorToHex(duration)';

%Destination Address
dest_add = data_est(33:33+48-1);
dest_add = reshape(dest_add, 4, 12)';
dest_add = binaryVectorToHex(dest_add)';

%Source Address
src_add = data_est(81:81+48-1);
src_add = reshape(src_add, 4, 12)';
src_add = binaryVectorToHex(src_add)';

%BSS ID
bss_id = data_est(129:129+48-1);
bss_id = reshape(bss_id, 4, 12)';
bss_id = binaryVectorToHex(bss_id)';







