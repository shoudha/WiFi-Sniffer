%%
%________Clearing and closing all the files and data________
clear all;
close all;
clc;
%________Opening/Reading/closing the file________
fileid_inphase = fopen('I_OFDM1.txt', 'r');
inphase = fscanf(fileid_inphase, '%f');
fclose(fileid_inphase);
inphase = inphase';
inphase_length = length(inphase);
fileid_qphase = fopen('Q_OFDM1.txt', 'r');
qphase = fscanf(fileid_qphase, '%f');
fclose(fileid_qphase);
qphase = qphase';
qphase_length = length(qphase);
%________Creating the I-Q signal________
r = complex(inphase, qphase);
r_length = length(r);
fs = 20e6;
ts = 1/fs;
figure();
plot(abs(r));
xlabel('Number of samples');
ylabel('Amplitude of signal');
title('Received I-Q signal');    
%%
%________coarse Fram Synchronization________
%Finding the starting index of the frame where the correlation is maximum
pcoarse = zeros(1, r_length-144-16); %Initializing pcoarse
for d = 1 : (r_length-144-16)
    pcoarse(d) = sum((conj(r(d+(0:143)))).*(r(d+(0:143)+16))); %Performing correlation between the received data.
    R(d) = (1/160)*(sum(abs(r(d+(0:159)).^2)));
    Mcoarse(d) = (abs(pcoarse(d)).^2)./R(d);
end
[corr_max,dindex] = max(Mcoarse); %Assigned the frame index in the variable dindex
dindex = dindex-4 ; %Frame index is slightly tuned to avoid the data to be scattered heavily.
figure();
plot(Mcoarse);
%%
%________fine CFO estimation________
%Performing CFO estimation
Ylong = sum((conj(r(dindex+(0:63)+192))).*(r(dindex+(0:63)+192+64)));
Ylong_phase = (angle(Ylong)*fs)/ (pi*128);
CFOcorrect = exp(-1*(sqrt(-1))*2*pi*Ylong_phase*ts*(0:(length(r)-1))).*r;
%%
%________channel coefficient estimation________
long_train_1 = [1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1]; 
long_train_2 = [1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1];
training_sequence = [long_train_1 long_train_2];
y1 = CFOcorrect((dindex+160+32+1) : (dindex+160+32+64));
Y1 = fftshift(fft(y1));
y2 = CFOcorrect((dindex+160+32+65) : (dindex+160+32+128));
Y2 = fftshift(fft(y2));
Y = (Y1+Y2);
Y_remove_null = [Y(7:32) Y(34:59)];
H = 0.5*(Y_remove_null./(training_sequence));
%%
%________Signal block________
signal_field = CFOcorrect((dindex+160+32+64+64+16+1):(dindex+160+32+64+64+16+64));
signal_field_fft = fftshift(fft(signal_field));
signal_field_compensated = [signal_field_fft(7:32) signal_field_fft(34:59)]./H;
signal_field_only_pilots = [signal_field_compensated(6) signal_field_compensated(20) signal_field_compensated(33) signal_field_compensated(47)];
phase_offset = signal_field_only_pilots./[1 1 1 -1];
phase = phase_offset./(abs(phase_offset));
phase_avg = mean(phase);
signal_field_compensated = signal_field_compensated*conj(phase_avg);
signal_field_compensated1 = [signal_field_compensated(1:5) signal_field_compensated(7:19) signal_field_compensated(21:32) signal_field_compensated(34:46) signal_field_compensated(48:52)];
scatterplot(signal_field_compensated);
title('BPSK Signal constellation');
demodulated_bits_w_pilot = (sign(real(signal_field_compensated))+1)/2;
Actual_prof_bits = [-1 1 1 -1 1 1 -1 -1 -1 1 -1 -1 1 1 1 1 1 -1 -1 1 1 -1 -1 -1 1 1 1 1 1 -1 -1 -1 1 -1 1 1 -1 1 1 -1 -1 1 -1 1 -1 -1 -1 -1 1 -1 1 1];
Actual_prof_bits = (sign(Actual_prof_bits)+1)/2;
if demodulated_bits_w_pilot == Actual_prof_bits
    display('The demodulated bits matched the professor displayed bits');
else
    display('The demodulated bits did not match the professor displayedbits');
end
demod_bits = (sign(real(signal_field_compensated1))+1)/2;
%%
%________Deinterleaving________
NBPS = 1;
NCBPS = 48;
S = max(NBPS/2,1);
for deint_j = 0:(NCBPS-1)    
    deint_i = (S)* floor(deint_j/S) + mod((deint_j + floor(16 * deint_j/NCBPS)), S);    
    deint_k(deint_j+1) = 16 * deint_i - (NCBPS-1) * floor(16 * deint_i/NCBPS);  
end
demod_bits(deint_k+1)= demod_bits;
%%
%________viterbi decoding________
trellis = poly2trellis([7], [133 171]);
sig_bits = vitdec(demod_bits, trellis, 24, 'term', 'hard');
ratebits = sig_bits(1:4);
%%
%________Rate Dependent parameters________
if ratebits ==  [1 1 0 1]
    Rate = 6;
    codingrate = '1/2';
    modulation = 'BPSK';
    NBPSC = 1;
    NCBPS = 48;
    NDBPS = 24;
elseif ratebits ==  [1 1 1 1]
    Rate = 9;
    codingrate = '3/4';
    modulation = 'BPSK';
    NBPSC = 1;
    NCBPS = 48;
    NDBPS = 36;
elseif ratebits == [0 1 0 1]
    Rate = 12;
    codingrate = '1/2';
    modulation = 'QPSK';
    NBPSC = 2;
    NCBPS = 96;
    NDBPS = 48;
elseif ratebits == [0 1 1 1]
    Rate = 18;
    codingrate = '13/4';
    modulation = 'QPSK';
    NBPSC = 2;
    NCBPS = 96;
    NDBPS = 72;
elseif ratebits == [1 0 0 1]
    Rate = 24;
    codingrate = '1/2';
    modulation = '16-QAM';
    NBPSC = 4;
    NCBPS = 192;
    NDBPS = 96;
elseif ratebits == [1 0 1 1]
    Rate = 36;
    codingrate = '3/4';
    modulation = '16-QAM';
    NBPSC = 4;
    NCBPS = 192;
    NDBPS = 144;
elseif ratebits == [0 0 0 1]
    Rate = 48;
    codingrate = '2/3';
    modulation = '64-QAM';
    NBPSC = 6;
    NCBPS = 288;
    NDBPS = 192;
elseif ratebits ==  [0 0 1 1]
    Rate = 54;
    codingrate = '3/4';
    modulation = '64-QAM';
    NBPSC = 6;
    NCBPS = 288;
    NDBPS = 216;
end
%%
signal_field_length_bits = fliplr(sig_bits(6:17));
an = 0;
bn = 0;
for i  = 1:18
    if sig_bits(i) == 1
        an = an+1;
    elseif sig_bits(i) == 0
        bn = bn+1;
    end
end
%Even/odd number
if mod(an,2) == 0
    cn = 1;
elseif mod(an,2) == 1
    cn = 0;
end 
Reserved_bit = sig_bits(5);
if (cn == Reserved_bit)
    display('We have even number of 1s');
else
    display('We have odd number of 1s');
end
Length_indicator_bit = sig_bits(18);
signal_tail_bits = sig_bits(19:24);
if sum(signal_tail_bits) == 0
    display('All the signal tail bits are zeros');
else
    display('All the signal tail bits are non zeros');    
end
%%
%________Length of OFDM data________
length_OFDM=bi2de(sig_bits(6:17));
Nsym = ceil((16+8*length_OFDM+6)/NDBPS);
Ndata = Nsym * NDBPS;
Npad = Ndata - (16+8*length_OFDM+6);
pilot_default = [1 1 1 -1];
pilot_sequence = [1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1];
pilot_sequence = pilot_sequence';
pilot_sequence = repmat(pilot_sequence,1,4);
pilot_Tx = repmat(pilot_default,(Nsym+1),1);
Final_pilot_polarities = pilot_Tx.*pilot_sequence(1:(Nsym+1),:);
%%
%________OFDM data________
OFDM_data=CFOcorrect((dindex+160+32+64+64+16+64+1):(dindex+160+32+64+64+16+64+1+(Nsym*80)));
CP_length=16;
OFDM_channel_compensated_total=[];
OFDM_fft_before_compensation=[];
m=1;
OFDM_final_output = [];
Total_sequence = [];
Total_final_sequence = [];
for ia=1:Nsym      
    OFDM_with_CP=OFDM_data(m:m+79);
    OFDM_without_CP=OFDM_with_CP(17:80);
    OFDM_fft= fftshift(fft(OFDM_without_CP));
    OFDM_channel_compensated= [ OFDM_fft(7:32) OFDM_fft(34:59) ] ./H;
    OFDM_channel_compensated_total=[OFDM_channel_compensated_total OFDM_channel_compensated]; 
    pilots_OFDM= [ OFDM_channel_compensated(6) OFDM_channel_compensated(20) OFDM_channel_compensated(33) OFDM_channel_compensated(47) ];
    t1=pilots_OFDM./Final_pilot_polarities(ia+1,:);
    t1 = t1./(abs(t1));
    t1_avg = mean(t1);
    t1_compensated = OFDM_channel_compensated*conj(t1_avg);
    OFDM_final_output_pilotsremoved=[ t1_compensated(1:5) t1_compensated(7:19) t1_compensated(21:32) t1_compensated(34:46) t1_compensated(48:end) ]; 
    OFDM_fft_before_compensation=[OFDM_fft_before_compensation OFDM_fft];
    m=m+80;
    if strcmp(modulation, '64-QAM')
        kmod = 1/sqrt(42);
        OFDM_final_output_pilotsremoved = OFDM_final_output_pilotsremoved/kmod;      
        for z = 1:(length(OFDM_final_output_pilotsremoved))
            if (-10<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=-6)
                b0(z) = 0; b1(z) = 0;b2(z) = 0;
            elseif (-6<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=-4)
                b0(z) = 0; b1(z) = 0;b2(z) = 1;
            elseif (-4<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=-2)
                b0(z) = 0; b1(z) = 1;b2(z) = 1;
            elseif (-2<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=0)
                b0(z) = 0; b1(z) = 1;b2(z) = 0;
            elseif (0<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=2)
                b0(z) = 1; b1(z) = 1;b2(z) = 0;
            elseif (2<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=4)
                b0(z) = 1; b1(z) = 1;b2(z) = 1;
            elseif (4<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=6)
                b0(z) = 1; b1(z) = 0;b2(z) = 1;
            elseif (6<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=10)
                b0(z) = 1; b1(z) = 0;b2(z) = 0;
            else
                display('The OFDM complex signal is not within the limits of desired constellation');
                %break;
            end
            if (-10<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=-6)
                b3(z) = 0; b4(z) = 0;b5(z) = 0;
            elseif (-6<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=-4)
                b3(z) = 0; b4(z) = 0;b5(z) = 1;
            elseif (-4<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=-2)
                b3(z) = 0; b4(z) = 1;b5(z) = 1;
            elseif (-2<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=0)
                b3(z) = 0; b4(z) = 1;b5(z) = 0;
            elseif (0<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=2)
                b3(z) = 1; b4(z) = 1;b5(z) = 0;
            elseif (2<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=4)
                b3(z) = 1; b4(z) = 1;b5(z) = 1;
            elseif (4<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=6)
                b3(z) = 1; b4(z) = 0;b5(z) = 1;
            elseif (6<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=10)
                b3(z) = 1; b4(z) = 0;b5(z) = 0;
            else 
                display('The OFDM complex signal is not within the limits of desired constellation');
            end
            sequence = [b0(z)  b1(z) b2(z)  b3(z)  b4(z) b5(z)];
            Total_sequence = [Total_sequence sequence];
        end
        %________Deinterleaving________
        S = max(NBPSC/2,1);
        for j = 0:(NCBPS-1)    
            i = (S)* floor(j/S) + mod((j + floor(16 * j/NCBPS)), S);    
            k(j+1) = 16 * i - (NCBPS-1) * floor(16 * i/NCBPS);  
        end
        ia = ia-1;
        Final_sequence(k+1)= Total_sequence((ia*NCBPS)+1:(ia*NCBPS)+NCBPS);
        Total_final_sequence = [Total_final_sequence Final_sequence];
        ia = ia+1;
    elseif strcmp(modulation, 'BPSK')
        kmod = 1;
        OFDM_final_output_pilotsremoved = OFDM_final_output_pilotsremoved/kmod;     
        for z = 1:(length(OFDM_final_output_pilotsremoved))
            if (OFDM_final_output_pilotsremoved(z))>0
                b0(z)=1;
            else
                b0(z)=0;
            end
            sequence = b0(z);
            Total_sequence = [Total_sequence sequence];
        end
        %________Deinterleaving________
        S = max(NBPSC/2,1);
        for j = 0:(NCBPS-1)    
            i = (S)* floor(j/S) + mod((j + floor(16 * j/NCBPS)), S);    
            k(j+1) = 16 * i - (NCBPS-1) * floor(16 * i/NCBPS);  
        end
        ia = ia-1;
        Final_sequence(k+1)= Total_sequence((ia*NCBPS)+1:(ia*NCBPS)+NCBPS);
        Total_final_sequence = [Total_final_sequence Final_sequence];
        ia = ia+1;            
    elseif strcmp(modulation,'QPSK')
        kmod = 1/sqrt(2);
        OFDM_final_output_pilotsremoved = OFDM_final_output_pilotsremoved/kmod;          
        for z = 1:(length(OFDM_final_output_pilotsremoved))
            if (-2<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=0)
                b0(z) = 0;
            elseif (0<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=2)
                b0(z) = 1;
            end
            if (-2<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=0)
                b1(z) = 0;
            elseif (0<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=2)
                b1(z) = 1;
            end
            sequence = [b0(z)  b1(z)];
            Total_sequence = [Total_sequence sequence];
        end
        %________Deinterleaving ________
        S = max(NBPSC/2,1);
        for j = 0:(NCBPS-1)    
            i = (S)* floor(j/S) + mod((j + floor(16 * j/NCBPS)), S);    
            k(j+1) = 16 * i - (NCBPS-1) * floor(16 * i/NCBPS);  
        end
        ia = ia-1;
        Final_sequence(k+1)= Total_sequence((ia*NCBPS)+1:(ia*NCBPS)+NCBPS);
        Total_final_sequence = [Total_final_sequence Final_sequence];
        ia = ia+1;            
    elseif strcmp(modulation, '16-QAM')
        kmod = 1/sqrt(10);
        OFDM_final_output_pilotsremoved = OFDM_final_output_pilotsremoved/kmod;             
        for z = 1:(length(OFDM_final_output_pilotsremoved))
            if (-4<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=-2)
                b0(z) = 0; b1(z) = 0;
            elseif (-2<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=0)
                b0(z) = 0; b1(z) = 1;
            elseif (0<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=2)
                b0(z) = 1; b1(z) = 1;
            elseif (2<(real(OFDM_final_output_pilotsremoved(z)))) && ((real(OFDM_final_output_pilotsremoved(z)))<=6)
                b0(z) = 1; b1(z) = 0;
            end
            if (-4<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=-2)
                b2(z) = 0; b3(z) = 0;
            elseif (-2<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=0)
                b2(z) = 0; b3(z) = 1;
            elseif (0<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=2)
                b2(z) = 1; b3(z) = 1;
            elseif (2<(imag(OFDM_final_output_pilotsremoved(z)))) && ((imag(OFDM_final_output_pilotsremoved(z)))<=6)
                b2(z) = 1; b3(z) = 0;
            end
            sequence = [b0(z)  b1(z) b2(z)  b3(z)];
            Total_sequence = [Total_sequence sequence];
        end
        %________Deinterleaving________
        S = max(NBPSC/2,1);
        for j = 0:(NCBPS-1)    
            i = (S)* floor(j/S) + mod((j + floor(16 * j/NCBPS)), S);    
            k(j+1) = 16 * i - (NCBPS-1) * floor(16 * i/NCBPS);  
        end
        ia = ia-1;
        Final_sequence(k+1)= Total_sequence((ia*NCBPS)+1:(ia*NCBPS)+NCBPS);
        Total_final_sequence = [Total_final_sequence Final_sequence];
        ia = ia+1;            
    end
    OFDM_final_output = [OFDM_final_output OFDM_final_output_pilotsremoved];
end
%%
%________Viterbi Decoding________
if strcmp(codingrate,'3/4')
    puncpat = [1,1,1,0,0,1];
    sig_bits_ofdm = vitdec(Total_final_sequence, trellis,35, 'term', 'hard', puncpat);
elseif strcmp(codingrate,'2/3')
    puncpat = [1,1,1,0];
    sig_bits_ofdm = vitdec(Total_final_sequence, trellis, 35, 'term', 'hard', puncpat);
else
    sig_bits_ofdm = vitdec(Total_final_sequence, trellis, 35, 'term', 'hard');
end
sig_bits_ofdm_wo_pad = sig_bits_ofdm(1:(length(sig_bits_ofdm)-Npad));
sig_bits_ofdm_tail_bits = sig_bits_ofdm_wo_pad((length(sig_bits_ofdm_wo_pad)-5):(length(sig_bits_ofdm_wo_pad)));
if isequal(sig_bits_ofdm_tail_bits,[0,0,0,0,0,0])
    fprintf('All the Tail bits before descrambler are matched with zero bits');
else
    fprintf('All the Tail bits before descrambler are not matched with zero bits');
end
%%
%________Descrambler________
descrambler_array = [0,0,0,0,1,1,1,0,1,1,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,1,1];
descrambler_arrayy = [descrambler_array descrambler_array];
for xo = 1:(length(descrambler_arrayy)-6)       
    if isequal(sig_bits_ofdm(1:7),descrambler_arrayy(xo:xo+6))
        descrambler_bits(1:7)=descrambler_arrayy(xo:(xo+6));
        descrambler_index =xo;      
        descrambler_array1 = [descrambler_arrayy(descrambler_index:end) descrambler_arrayy(1:descrambler_index-1)];
        descrambler_array_total = repmat(descrambler_array1,1,ceil(length(sig_bits_ofdm)/120));       
        descrambler_rem_bits = descrambler_array_total(1:length(sig_bits_ofdm));
        descrambler_rem_bits_length = length(descrambler_rem_bits);
        signal_rem_bits = sig_bits_ofdm(1:end);
        signal_rem_bits_length = length(signal_rem_bits);
        xor_sig_descrambler = xor( signal_rem_bits,  descrambler_rem_bits);
        xor_sig_descrambler_first_7_bits= xor_sig_descrambler(1:7);
        xor_sig_descrambler_second_7_bits= xor_sig_descrambler(8:16);
    end
end
%%
%________TYPE/SUBTYPE________
type = fliplr(xor_sig_descrambler(19:20))
subtype = fliplr(xor_sig_descrambler(21:24))
if isequal(type,[0,0])
    display('It is a Management Frame');
    if isequal(subtype,[1,0,0,0])
        display('Subtype is a Beacon');
    elseif isequal(subtype,[1,0,1,0])
        display('Subtype is a Disassociation');
    else
        fprintf('This is not a Supported Subtype of a Management Frame');
        %break;
    end
elseif isequal(type,[0,1])
    display('It is a Control Frame');
    if isequal(subtype,[1,0,1,0])
        display('Subtype is a Block ACK');
    elseif isequal(subtype,[1,0,1,1])
        display('Subtype is an RTS');
    elseif isequal(subtype,[1,1,0,1])
        display('Subtype is an ACK');  
    else
        fprintf('This is not a Supported Subtype of a Control Frame');
        %break;
    end
elseif isequal(type,[1,0])
    display('It is a Data Frame');
    if isequal(subtype,[0,0,0,0])
        display('It is a Data Subtype');
    elseif isequal(subtype,[0,1,0,0])
        display('It is a Null Subtype');
    elseif isequal(subtype,[1,0,0,0])
        display('It is a QoS data Subtype');
    elseif isequal(subtype,[1,1,0,0])
        display('It is a QoS Null Subtype');
    else
        fprintf('This is not a Supported Subtype of a Data Frame');
        %break;
    end
elseif isequal(type,[1,1])
    display('It is a Reserved');
end 
%Finding the destination address
Destination.address_binary = fliplr(xor_sig_descrambler(49:96));
Destination.address_hex= binaryVectorToHex(Destination.address_binary)
%Finding the source address
Source.address_binary = fliplr(xor_sig_descrambler(97:144));
Source.address_hex= binaryVectorToHex(Source.address_binary)
%Finding the BSSID
% bssid = (xor_sig_descrambler(145:192));
% bssid= binaryVectorToHex(bssid, 'LSBFirst')
bssid = reshape(xor_sig_descrambler(17:end-48), 8, 1558)';
bssid = binaryVectorToHex(bssid, 'LSBFirst')'

%%
%Plotting the Constellation before channel compensation
scatterplot(OFDM_fft_before_compensation);
title('Constellation before channel compensation');
xlabel('In-Phase');
ylabel('Quadrature-Phase');
%Plotting the Constellation after channel compensation  
scatterplot(OFDM_channel_compensated_total);
title('Constellation after channel compensation');
xlabel('In-Phase');
ylabel('Quadrature-Phase');
%Plotting the Final constelation after channel and Residual compensation  
scatterplot(OFDM_final_output);
title('Final Constellation');
xlabel('In-Phase');
ylabel('Quadrature-Phase');
fprintf('The final constellation is %s\n',modulation);
%%

