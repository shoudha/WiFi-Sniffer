clc;
clearvars;
close all;


poly = [1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1];
len_poly = length(poly);

% data = [1 randi([0 1], 1, 35)];
% msg = [1 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 0 1 1 0 0 1 1 0 1 0 0 0 0 1 1 0 1 0 1 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 0 0 1];

msg = '123456789';

%Converting msg from character to binary
msg = dec2hex(double(msg))';
msg = hexToBinaryVector(msg(:)');


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
CRC = reshape(CRC, 4, 8)'














