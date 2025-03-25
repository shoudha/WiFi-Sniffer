clc
clearvars
close all

load 'entire_packet'

entire_packet = x;

%% Removing Short and Long Preamble, Signal Field and Last sample
x(1:400) = [];
x(end) = [];

%% Data Symbol extract
x = reshape(x, 80, 6);

%Remove CP
x(1:16,:) = [];

%FFT
x_freq = fft(x);

%Removing null SC
x_freq([1, 28:38],:) = [];

% Rearranging the signal from [1-52] to [-26-26]
x_freq = [x_freq(27:end,:); x_freq(1:26,:)];

%Remove Pilots
x_freq([6 20 33 47],:) = [];

%% Demodulate from symbols to databit
data = zeros(4, 48*6);

%Normalize
x_freq = x_freq.*sqrt(10);

%Quantize the symbols
levels = -3:2:3;

x_freq_re = real(x_freq(:));
x_freq_im = imag(x_freq(:));

x_re_diff = abs(x_freq_re - levels);
x_im_diff = abs(x_freq_im - levels);

[~, x_re_min_ind] = min(x_re_diff, [], 2);
[~, x_im_min_ind] = min(x_im_diff, [], 2);

x_re_q = levels(x_re_min_ind);
x_im_q = levels(x_im_min_ind);

%Demodulate using encode table
QAM_16_en_tb = [0,0; 0,1; 1,1; 1,0].';

levels = repmat(levels.', 1, length(x_re_q));

x_re_q_mod = repmat(x_re_q, size(levels, 1), 1);
x_im_q_mod = repmat(x_im_q, size(levels, 1), 1);

re_ind = find(x_re_q_mod == levels);
im_ind = find(x_im_q_mod == levels);

[re_ind, ~] = ind2sub(size(x_re_q_mod), re_ind);
[im_ind, ~] = ind2sub(size(x_im_q_mod), im_ind);

data(1:2,:) = QAM_16_en_tb(:,re_ind);
data(3:4,:) = QAM_16_en_tb(:,im_ind);

data = data(:);

%% Deinterleave

data = reshape(data, 192, 6);

no_OFDM_symbols = 6;
Ndatasc = 48*1;
mod_typ = 4;
ncbps = Ndatasc*mod_typ;
ncpc = 4;

k = 0:ncbps - 1;
s = ceil(ncpc/2);

%First de-permutation of de-interleaver
i = s*floor(k/s) + mod(k + floor(16*k/ncbps), s);

%second de-permutation of de-interleaver
j = 16*i - (ncbps - 1)*floor(16*i/ncbps);

x_dei(j + 1,:) = data(k + 1,:);

%% Puncturing and Viterbi Decode

decoder_input = x_dei;
cols = numel(decoder_input)/12;
decoder_input = reshape(decoder_input(:), 12, cols);

%Decode
trellis = poly2trellis(7, [133 171]);
puncpat = [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1];
decoder_actual_output = vitdec(decoder_input(:), trellis, 35, 'trunc', 'hard', puncpat);
decoder_actual_output = reshape(decoder_actual_output, 9, cols);

%% Descrambler

x_scrambled = decoder_actual_output(:)';
init_state = [1 0 1 1 1 0 1];
    
n = length(x_scrambled);
x_descrambled = zeros(1, n);
for i = 1:n
    
    x_descrambled(i) = xor(x_scrambled(i), xor(init_state(4), init_state(1)));
    
    state_4 = init_state(4);
    state_7 = init_state(1);
    
    init_state = circshift(init_state, -1);
    init_state(7) = xor(state_4, state_7);
end

x_descrambled = reshape(x_descrambled, 144, 6);


%% THE RAW DATA
service_bits = x_descrambled(1:16);
tail_bits = x_descrambled(end - 47:end - 42);
appended_bits = x_descrambled(end - 41:end);

%Remove Service, tail and appended bits
x_descrambled(1:16) = [];
x_descrambled = x_descrambled(1:800);

%Convert the databits to hexa and then character
x_raw_data_bin = reshape(x_descrambled, 8, 100).'
x_raw_hex = binaryVectorToHex(x_raw_data_bin, 'LSBFirst');

x_raw_dec = hex2dec(x_raw_hex);
x_raw_char = char(x_raw_dec)';
x_poem = x_raw_char(25:25 + 72 - 1);

display(x_raw_char)
display(x_poem)

x_mac_bin = x_descrambled(1:96);
x_mac_bin = reshape(x_mac_bin, 16, 6)'
x_mac_hex = x_raw_hex(1:24)
x_mac_hex_2 = binaryVectorToHex(x_mac_bin, 'LSBFirst')

x_mac_bin = x_mac_bin';
x_mac_bin = x_mac_bin(:)';
frame_bits = x_mac_bin(1:16)
frame_hex = binaryVectorToHex(frame_bits, 'LSBFirst')

check_hex = cell2mat(x_raw_hex);

x_hex = check_hex';
x_hex = x_hex(:);
x_hex = x_hex';
x_hex(end-7:end) = [];




