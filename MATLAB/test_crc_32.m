clc;
clearvars;

data = ['aa'; 'aa'; '03'; '00'; '00'; '00'; '08'; '00'; '45'; '00';...
        '00'; '4e'; '66'; '1a'; '00'; '00'; '80'; '11'; 'be'; '64';...
        '0a'; '00'; '01'; '22'; '0a'; 'ff'; 'ff'; 'ff'; '00'; '89';...
        '00'; '89'; '00'; '3a'; '00'; '00'; '80'; 'a6'; '01'; '10';...
        '00'; '01'; '00'; '00'; '00'; '00'; '00'; '00'; '20'; '45';...
        '43'; '45'; '4a'; '45'; '48'; '45'; '43'; '46'; '43'; '45';...
        '50'; '46'; '45'; '45'; '49'; '45'; '46'; '46'; '43'; '43';...
        '41'; '43'; '41'; '43'; '41'; '43'; '41'; '43'; '41'; '41';...
        '41'; '00'; '00'; '20'; '00'; '01'; '1b'; 'd0'; 'b6'; '04'];
    

data_hex = data';
data_hex = data_hex(:);
data_hex = data_hex';


data_bin = hexToBinaryVector(data)';
data_bin = (data_bin(:)');

crc_hex = ['1b'; 'd0'; 'b6'; '04'];
crc_bin = hexToBinaryVector(crc_hex)';
crc_bin = fliplr(crc_bin(:)');


GenPoly = [1 1 1 0 1 1 0 1 1 0 1 1 1 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0]; 
%G(x) = x32 + x26 + x23 + x22 + x16 + x12 + x11 + x10 + x8 + x7 + x5 + x4 + x2 + x + 1

BufferInit = ones(1, 32);
Input = [data_hex zeros(1, 32)];
for i = 1:length(Input)
    
    temp1 = BufferInit(end);
    temp2 = temp1*GenPoly;
    
    for j = length(BufferInit):-1:2
        BufferInit(j) = xor(temp2(j), BufferInit(j - 1));
    end
    
    BufferInit(1) = xor(Input(i), temp2(1));
end

Output = fliplr(BufferInit);

sum(crc_bin == Output)



