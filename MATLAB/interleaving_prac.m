clc;
clearvars;
close all;

rate = [1 0 1 1];
res = 0;
len = [0 0 0 0 0 1 1 0 0 1 0 0];
par = 0;
tail = zeros(1, 6);
org = [rate res fliplr(len) par tail];

org_con = [1 1 0 1 0 0 0 1 1 0 1 0,...
            0 0 0 1 0 0 0 0 0 0 1 0,...
            0 0 1 1 1 1 1 0 0 1 1 1,...
            0 0 0 0 0 0 0 0 0 0 0 0];
        
org_int_out = [1 0 0 1 0 1 0 0 1 1 0 1,...
                0 0 0 0 0 0 0 1 0 1 0 0,...
                1 0 0 0 0 0 1 1 0 0 1 0,...
                0 1 0 0 1 0 0 1 0 1 0 0];

signal_52 = [-1 1 1 -1 1 1 -1 -1 -1 1 -1 -1 1 1 1 1 1 -1 -1 1 1,...
            -1 -1 -1 1 1 1 1 1 -1 -1 -1 1 -1 1 1 -1 1 1 -1 -1 1,...
            -1 1 -1 -1 -1 -1 1 -1 1 1];
signal_52(signal_52 < 0) = 0;
signal_52([6 20 33 47]) = [];

ERP_dei = [0 0 0 1 1 1 0 0 1 1 0 0 0 0 0 0 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 1 1 0 0 1 0 1 0 1 0 1 1 1 1 0 0];            

%% Convolutional coding and decoding
trellis1 = poly2trellis(7, [133 171]);
code1 = convenc(org, trellis1);
decoded = vitdec(org_con, trellis1, 1, 'trunc', 'hard');

% table(org_con', code1')
% table(decoded', org')

error_bits_encoding = sum(code1 ~= org_con)
error_bits_decoding = sum(decoded ~= org)

%% Interleaver

%Parameters
Ndatasc = 48;
ncbps = Ndatasc;
ncpc = 1;

s = max(ncpc/2, 1);
k = 0:ncbps-1;

%First permutation of interleaver
i = (ncbps/16)*mod(k,16) + floor(k/16);

%Second permutation of interleaver
j = s*floor(i/s) + mod(i + ncbps - floor(16*i/ncbps), s);

interleaved_data_out(j + 1) = ERP_dei(k + 1);

error_bits_inter = sum(interleaved_data_out ~= signal_52)

%% Deinterleaver
Ndatasc = 48;
ncbps = Ndatasc;
ncpc = 1;

s = max(ncpc/2, 1);
j = 0:ncbps-1;

%First de-permutation of de-interleaver
i = s*floor(j/s) + mod(j + floor(16*j/ncbps), s);

%second de-permutation of de-interleaver
k = 16*i - (ncbps - 1)*floor(16*i/ncbps);

deinterleaver_data_out(k + 1) = org_int_out(j + 1);

error_bits_deinter = sum(deinterleaver_data_out ~= org_con)














