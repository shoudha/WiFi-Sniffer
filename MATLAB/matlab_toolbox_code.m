clc;
clearvars;
% close all;

load 'Collected Data_Current\wifiData_index_915909_code_3_4_54qam'

f_ind = 909381;
len_OFDM = 94;

%% MATLAB Toolbox

nht = wlanNonHTConfig('MCS', 6, 'PSDULength', len_OFDM)

rxPPDU = Y.';

%% Receiver
fieldInd = wlanFieldIndices(nht)
fieldInd.LSTF = fieldInd.LSTF + f_ind;
fieldInd.LLTF = fieldInd.LLTF + f_ind;
fieldInd.LSIG = fieldInd.LSIG + f_ind;
fieldInd.NonHTData = fieldInd.NonHTData + f_ind

numSamples = fieldInd.LSIG(2);

rxLLTF = rxPPDU(fieldInd.LLTF(1):fieldInd.LLTF(2),:);
demodLLTF = wlanLLTFDemodulate(rxLLTF, nht);
chEstLLTF = wlanLLTFChannelEstimate(demodLLTF, nht);

plot(unwrap(angle(chEstLLTF)))

rxLSIG = rxPPDU(fieldInd.LSIG(1):fieldInd.LSIG(2),:);
recLSIG = wlanLSIGRecover(rxLSIG, chEstLLTF, 1, 'CBW20');
rate = recLSIG(1:4)'

cfgRec = wlanRecoveryConfig('EqualizationMethod', 'ZF');

rxPSDU = rxPPDU(fieldInd.NonHTData(1):fieldInd.NonHTData(2),:);
[recPSDU, ~, eqSym] = wlanNonHTDataRecover(rxPSDU, chEstLLTF, 1, nht, cfgRec);


% rec_bin = logical(reshape(recPSDU, 8, len_OFDM)).';
% rec_hex = binaryVectorToHex(rec_bin, 'LSBfirst');
% rec_hex = cell2mat(rec_hex)';
% rec_hex = rec_hex(:)';
% [rec_hex(65:65+7); rec_hex(73:80); rec_hex(81:88)]

% numErr = biterr(data_est.', recPSDU)

% check = [data_est.' recPSDU (data_est.' ~= recPSDU)];
% min(find((check(:,3) == 1)))
% stem(check(:,3))
% axis tight
