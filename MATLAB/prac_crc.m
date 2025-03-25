clc;
clearvars;
close all;


poly = [1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1];
len_poly = length(poly);


msg = [0 1 1 0 0 0 0 1 0 1 1 0 0 0 1 0 0 1 1 0 0 0 1 1];

%Processing Message for CRC calculation
data = [zeros(1, ceil(length(msg)/8)*8-length(msg)) msg];
data = fliplr(reshape(data, 8, length(data)/8)');
data = data';
data = data(:)';
data = [data zeros(1, len_poly - 1)];

reshape(data, 8, 7)'