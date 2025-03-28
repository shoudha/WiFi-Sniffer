clc;
clearvars;

table = [   '00000000'; '04C11DB7'; '09823B6E'; '0D4326D9'; '130476DC'; '17C56B6B'; '1A864DB2'; '1E475005';...
            '2608EDB8'; '22C9F00F'; '2F8AD6D6'; '2B4BCB61'; '350C9B64'; '31CD86D3'; '3C8EA00A'; '384FBDBD';...
            '4C11DB70'; '48D0C6C7'; '4593E01E'; '4152FDA9'; '5F15ADAC'; '5BD4B01B'; '569796C2'; '52568B75';...
            '6A1936C8'; '6ED82B7F'; '639B0DA6'; '675A1011'; '791D4014'; '7DDC5DA3'; '709F7B7A'; '745E66CD';...
            '9823B6E0'; '9CE2AB57'; '91A18D8E'; '95609039'; '8B27C03C'; '8FE6DD8B'; '82A5FB52'; '8664E6E5';...
            'BE2B5B58'; 'BAEA46EF'; 'B7A96036'; 'B3687D81'; 'AD2F2D84'; 'A9EE3033'; 'A4AD16EA'; 'A06C0B5D';...
            'D4326D90'; 'D0F37027'; 'DDB056FE'; 'D9714B49'; 'C7361B4C'; 'C3F706FB'; 'CEB42022'; 'CA753D95';...
            'F23A8028'; 'F6FB9D9F'; 'FBB8BB46'; 'FF79A6F1'; 'E13EF6F4'; 'E5FFEB43'; 'E8BCCD9A'; 'EC7DD02D';...
            '34867077'; '30476DC0'; '3D044B19'; '39C556AE'; '278206AB'; '23431B1C'; '2E003DC5'; '2AC12072';...
            '128E9DCF'; '164F8078'; '1B0CA6A1'; '1FCDBB16'; '018AEB13'; '054BF6A4'; '0808D07D'; '0CC9CDCA';...
            '7897AB07'; '7C56B6B0'; '71159069'; '75D48DDE'; '6B93DDDB'; '6F52C06C'; '6211E6B5'; '66D0FB02';...
            '5E9F46BF'; '5A5E5B08'; '571D7DD1'; '53DC6066'; '4D9B3063'; '495A2DD4'; '44190B0D'; '40D816BA';...
            'ACA5C697'; 'A864DB20'; 'A527FDF9'; 'A1E6E04E'; 'BFA1B04B'; 'BB60ADFC'; 'B6238B25'; 'B2E29692';...
            '8AAD2B2F'; '8E6C3698'; '832F1041'; '87EE0DF6'; '99A95DF3'; '9D684044'; '902B669D'; '94EA7B2A';...
            'E0B41DE7'; 'E4750050'; 'E9362689'; 'EDF73B3E'; 'F3B06B3B'; 'F771768C'; 'FA325055'; 'FEF34DE2';...
            'C6BCF05F'; 'C27DEDE8'; 'CF3ECB31'; 'CBFFD686'; 'D5B88683'; 'D1799B34'; 'DC3ABDED'; 'D8FBA05A';...
            '690CE0EE'; '6DCDFD59'; '608EDB80'; '644FC637'; '7A089632'; '7EC98B85'; '738AAD5C'; '774BB0EB';...
            '4F040D56'; '4BC510E1'; '46863638'; '42472B8F'; '5C007B8A'; '58C1663D'; '558240E4'; '51435D53';...
            '251D3B9E'; '21DC2629'; '2C9F00F0'; '285E1D47'; '36194D42'; '32D850F5'; '3F9B762C'; '3B5A6B9B';...
            '0315D626'; '07D4CB91'; '0A97ED48'; '0E56F0FF'; '1011A0FA'; '14D0BD4D'; '19939B94'; '1D528623';...
            'F12F560E'; 'F5EE4BB9'; 'F8AD6D60'; 'FC6C70D7'; 'E22B20D2'; 'E6EA3D65'; 'EBA91BBC'; 'EF68060B';...
            'D727BBB6'; 'D3E6A601'; 'DEA580D8'; 'DA649D6F'; 'C423CD6A'; 'C0E2D0DD'; 'CDA1F604'; 'C960EBB3';...
            'BD3E8D7E'; 'B9FF90C9'; 'B4BCB610'; 'B07DABA7'; 'AE3AFBA2'; 'AAFBE615'; 'A7B8C0CC'; 'A379DD7B';...
            '9B3660C6'; '9FF77D71'; '92B45BA8'; '9675461F'; '8832161A'; '8CF30BAD'; '81B02D74'; '857130C3';...
            '5D8A9099'; '594B8D2E'; '5408ABF7'; '50C9B640'; '4E8EE645'; '4A4FFBF2'; '470CDD2B'; '43CDC09C';...
            '7B827D21'; '7F436096'; '7200464F'; '76C15BF8'; '68860BFD'; '6C47164A'; '61043093'; '65C52D24';...
            '119B4BE9'; '155A565E'; '18197087'; '1CD86D30'; '029F3D35'; '065E2082'; '0B1D065B'; '0FDC1BEC';...
            '3793A651'; '3352BBE6'; '3E119D3F'; '3AD08088'; '2497D08D'; '2056CD3A'; '2D15EBE3'; '29D4F654';...
            'C5A92679'; 'C1683BCE'; 'CC2B1D17'; 'C8EA00A0'; 'D6AD50A5'; 'D26C4D12'; 'DF2F6BCB'; 'DBEE767C';...
            'E3A1CBC1'; 'E760D676'; 'EA23F0AF'; 'EEE2ED18'; 'F0A5BD1D'; 'F464A0AA'; 'F9278673'; 'FDE69BC4';...
            '89B8FD09'; '8D79E0BE'; '803AC667'; '84FBDBD0'; '9ABC8BD5'; '9E7D9662'; '933EB0BB'; '97FFAD0C';...
            'AFB010B1'; 'AB710D06'; 'A6322BDF'; 'A2F33668'; 'BCB4666D'; 'B8757BDA'; 'B5365D03'; 'B1F740B4'];
        
        
        

poly = ([0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1]);       
crcTable = zeros(32, 256);       
for i = 0:255
    
    crc = [de2bi(i, 8, 'left-msb') zeros(1, 24)];
        
    for j = 0:7
                
        pop_out = crc(1);
        
        if crc(1)
            crc = [crc(2:end) 0];
            crc = xor(crc, poly);
        else
            crc = [crc(2:end) 0];
        end
        
    end
    
    crcTable(:, i+1) = crc;
    
end

% crcTable


% data = randi([0 1], 1, 32);
data = [0 0 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 0 1 1 0 1 0 0 0 0 0 0 1 1 1 1];


%Hex data
data_hex = reshape(data, 4, length(data)/4)';
data_hex = binaryVectorToHex(data_hex);
data_hex = cell2mat(data_hex)';
    

len_data = length(data);
no_bytes = ceil(len_data/8);
data = [zeros(1, no_bytes*8 - len_data) data];


data = reshape(data, 8, no_bytes);



crc = ones(1, 32);
for i = 1:no_bytes

    data_byte = [(data(:, i)') zeros(1,24)];

    crc_xor = xor(crc, data_byte);
    pos = crc_xor(1:8);

    pos = bin2dec(num2str(pos))

    crc_shift_8 = [crc(9:end) zeros(1,8)];
    crc = xor(crc_shift_8, fliplr(crcTable(:, pos)'));

end


crc = fliplr(xor(crc, ones(1, length(crc))))


%Hex CRC
crc_reg_hex = reshape(crc, 4, length(crc)/4)';
crc_reg_hex = binaryVectorToHex(crc_reg_hex);
crc_reg_hex = cell2mat(crc_reg_hex)';
    
data_hex
crc_reg_hex  



bits = data(:)';
crc_2 = ~crc32(bits)'


function ret = crc32(bits)
poly = [1 de2bi(hex2dec('04C11DB7'), 32)]';
bits = bits(:);

% Flip first 32 bits
bits(1:32) = 1 - bits(1:32);
% Add 32 zeros at the back
bits = [bits; zeros(32,1)];

% Initialize remainder to 0
rem = zeros(32,1);
% Main compution loop for the CRC32
for i = 1:length(bits)
    rem = [rem; bits(i)]; %#ok<AGROW>
    if rem(1) == 1
        rem = mod(rem + poly, 2);
    end
    rem = rem(2:33);
end

% Flip the remainder before returning it
ret = 1 - rem;
end

