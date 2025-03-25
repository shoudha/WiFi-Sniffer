clc;
clearvars;
close all;


%Polynomial for CRC
poly = [1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1];
len_poly = length(poly)

% data = [1 randi([0 1], 1, 35)];
% data is MSB first
data = [1 1 0 1 0 0 1 0];
len_data = length(data);

%Reflect Input
data = fliplr(data);

% Load the register with initial state
crc_reg = zeros(1, len_poly - 1);

for i = 1:32
    
    pop_out = crc_reg(1);
    crc_reg(1:end-1) = crc_reg(2:end)
    
    
    if i > len_data
        crc_reg(end) = 0;
    else
        crc_reg(end) = data(i);
    end
        
    
    
    if pop_out == 1

        crc_reg = xor(crc_reg, poly(2:end));
        
    end
    
end

%Xor with final state
crc_reg = xor(crc_reg, ones(1, 32));

%Reflect Output
crc_reg = fliplr(crc_reg);

%Display CRC
crc_reg = reshape(crc_reg, 4, 8)'





