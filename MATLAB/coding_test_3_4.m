clc;
clearvars;
close all;


%% Interleaver
input_interleaver = randi([0 1], 1, 144);

trellis = poly2trellis(7, [133 171]);
[output_no_shape_no_steal, final_state] = convenc(input_interleaver, trellis);

rows = 18;
cols = numel(output_no_shape_no_steal)/rows;
output_no_steal = reshape(output_no_shape_no_steal, rows, cols);

output_encoder = output_no_steal([1 2 3 6 7 8 9 12 13 14 15 18], :);
output_encoder = output_encoder(:)';

puncpat = [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1];
output_encoder_2 = convenc(input_interleaver, trellis, puncpat);

check_encoder = sum(output_encoder_2 ~= output_encoder)

%% Deinterleaver
dei_output = vitdec(output_encoder, trellis, 1, 'trunc', 'hard', puncpat);


check_decoder = sum(dei_output ~= input_interleaver)