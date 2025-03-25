clc;
clearvars;


init = [1 0 1 1 1 1 1];
seq = zeros(1, 127);

for i = 1:127
    
    init_4 = init(4);
    init_7 = init(7);
    
    init_4_7_xor = xor(init_4, init_7);
    
    seq(i) = init_4_7_xor;
    
    init(2:end) = init(1:end-1);
    init(1) = init_4_7_xor;
    
end


reshape(seq(1:end-7), 8, 15)
seq(end-6:end)'



N = 500;
n = floor(N/127);
n_extra_bits = N - n*127;

seq_repeated = [repmat(seq, 1, n) seq(1:n_extra_bits)];




data = randi([0 1], 1, N);
data(1:7) = zeros(1, 7);

data_scrambled = bitxor(data, seq_repeated);

data_descrambled = bitxor(data_scrambled, seq_repeated);

sum(data ~= data_descrambled)

