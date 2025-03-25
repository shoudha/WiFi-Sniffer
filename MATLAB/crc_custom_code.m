clc;
clearvars;
close all;



poly = [1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1];
len_poly = length(poly)

% data = [1 randi([0 1], 1, 35)];
data = [1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 0 1 1 0 0 1 1 1 1 0 1 0 1 1 1 0 1];

%Hex data
data_hex = reshape(data, 4, 8)';
data_hex = binaryVectorToHex(data_hex);
data_hex = cell2mat(data_hex)'

%Division
data_temp = [data zeros(1, len_poly-1)];

CRC_1 = crc_custom(data_temp, poly);

CRC_1 = [zeros(1, len_poly - length(CRC_1) - 1) CRC_1];

%Hex CRC
CRC_1_hex = reshape(CRC_1, 4, (len_poly-1)/4)';
CRC_1_hex = binaryVectorToHex(CRC_1_hex);
CRC_1_hex = cell2mat(CRC_1_hex)'


CRC_check = ~sum(crc_custom([data CRC_1], poly));


%Bit-by-bit method

% Load the register with zero bits.
crc_reg = ones(1, len_poly - 1);

% Augment the message by appending W zero bits to the end of it.
data_temp = [data zeros(1, len_poly-1)];

% While (more message bits)
len_msg = length(data_temp);
while(len_msg ~= 0)
    
    % Shift the register left by one bit, reading the next bit of the augmented message into register bit position 0.
    
    pop_out = crc_reg(1);
    crc_reg(1:end-1) = crc_reg(2:end);
    crc_reg(end) = data_temp(1);
    data_temp(1) = [];

    % If (a 1 bit popped out of the register during step 3)
    if pop_out

        % Register = Register XOR Poly.
        crc_reg = bitxor(crc_reg, poly(2:end));
        
    end
    
    len_msg = length(data_temp);

end

% The register now contains the remainder.
% crc_reg
crc_reg = xor(crc_reg, ones(1, 32));



%Hex CRC
crc_reg_hex = reshape(crc_reg, 4, (len_poly - 1)/4)';
crc_reg_hex = binaryVectorToHex(crc_reg_hex);
crc_reg_hex = cell2mat(crc_reg_hex)'


isequal(crc_reg, CRC_1)











function CRC = crc_custom(data_temp, poly)

len_poly = length(poly);
xor_with_poly = bitxor(data_temp(1:len_poly), poly);
data_temp(1:len_poly) = [];

stop = 0;
while(~stop)
    cur_data_len = length(data_temp);
    num_zeros_xor = min(find(xor_with_poly == 1)) - 1;
    
    if ~sum(xor_with_poly)
        num_zeros_xor = length(xor_with_poly);
    end
    
    xor_with_poly(1:num_zeros_xor) = [];
    
    len_xor = length(xor_with_poly);
    diff_poly_xor = len_poly - len_xor;
    
    if diff_poly_xor > cur_data_len
        
        xor_with_poly = [xor_with_poly data_temp];
        stop = 1;
        break
        
    else
                
        append_xor = data_temp(1:diff_poly_xor);
        new_xor = [xor_with_poly append_xor];

    end
    
    xor_with_poly = bitxor(poly, new_xor);
    data_temp(1:diff_poly_xor) = [];
    
end

CRC = xor_with_poly;

end


