clc;
clearvars;
close all;

%% Wave generation and Transmitter
nht = wlanNonHTConfig('MCS', 7, 'PSDULength', 1558)
% x = randi([0 1], nht.PSDULength*8, 1);
% y = wlanWaveformGenerator(x, nht, 'WindowTransitionTime', 0);

load test_OFDM_data
y = I_OFDM1 + (sqrt(-1))*Q_OFDM1;
f_ind = 15469;


%% The Channel
% dist = 3;
% pathLoss = 10^(-log10(4*pi*dist*(2.4e9/3e8)));
fs = 20e6;
% trms = 2/fs;
% maxDoppShift = 3;
% ch802 = comm.RayleighChannel('SampleRate',fs,'MaximumDopplerShift',maxDoppShift,'PathDelays',trms)
% 
% awgnChan = comm.AWGNChannel('NoiseMethod','Variance','VarianceSource','Input port');
% 
noiseVar = 10^((-228.6 + 10*log10(290) + 10*log10(fs) + 9)/10);
% rxPPDU = awgnChan(ch802(y),noiseVar) * pathLoss;

rxPPDU = y;

%% Receiver
fieldInd = wlanFieldIndices(nht)



numSamples = fieldInd.LSIG(2);

rxLLTF = rxPPDU(fieldInd.LLTF(1):fieldInd.LLTF(2),:);
demodLLTF = wlanLLTFDemodulate(rxLLTF, nht);
chEstLLTF = wlanLLTFChannelEstimate(demodLLTF, nht);

rxLSIG = rxPPDU(fieldInd.LSIG(1):fieldInd.LSIG(2),:);
recLSIG = wlanLSIGRecover(rxLSIG, chEstLLTF, noiseVar, 'CBW20');
rate = recLSIG(1:4)'

cfgRec = wlanRecoveryConfig('EqualizationMethod', 'ZF');

rxPSDU = rxPPDU(fieldInd.NonHTData(1):fieldInd.NonHTData(2),:);
[recPSDU, ~, eqSym] = wlanNonHTDataRecover(rxPSDU,chEstLLTF,noiseVar,nht,cfgRec);

numErr = biterr(x, recPSDU)
























