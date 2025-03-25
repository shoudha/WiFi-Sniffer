clc;
clearvars;
close all;

% msg = reshape(de2bi(49:57, 8, 'left-msb')', 72, 1);
msg = [1 1 0 1 0 0 1 0]';

gen = crc.generator('Polynomial' , '0x04C11DB7',...
                    'ReflectInput' , true,...
                    'ReflectRemainder' , true,...
                    'InitialState' , '0x00000000',...
                    'FinalXOR' , '0xFFFFFFFF');

% The message below is an ASCII representation of the digits 1-9
% msg = ([0 0 1 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 0 1 1 0 0 1 1 0 1 0 0 0 0 1 1 0 1 0 1 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 0 0 1])';
encoded = generate(gen, msg);

crc_32 = encoded(end-31:end);

crc_32 = reshape(crc_32, 4, 8)'
