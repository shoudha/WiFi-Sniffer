function varargout = prac_gui(varargin)
% PRAC_GUI MATLAB code for prac_gui.fig
%      PRAC_GUI, by itself, creates a new PRAC_GUI or raises the existing
%      singleton*.
%
%      H = PRAC_GUI returns the handle to a new PRAC_GUI or the handle to
%      the existing singleton*.
%
%      PRAC_GUI('CALLBACK',hObject,~,handles,...) calls the local
%      function named CALLBACK in PRAC_GUI.M with the given input arguments.
%
%      PRAC_GUI('Property','Value',...) creates a new PRAC_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before prac_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to prac_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
% Edit the above text to modify the response to help prac_gui
% Last Modified by GUIDE v2.5 07-Dec-2018 14:55:37
% Begin initialization code - DO NOT EDIT
clc
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @prac_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @prac_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before prac_gui is made visible.
function prac_gui_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to prac_gui (see VARARGIN)

% Choose default command line output for prac_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes prac_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = prac_gui_OutputFcn(~, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(~, ~, ~)
% hObject    handle to axes1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1

% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(~, ~, ~)
% hObject    handle to axes2 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2

% --- Executes during object creation, after setting all properties.
function axes3_CreateFcn(~, ~, ~)
% hObject    handle to axes3 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes3

% --- Executes during object creation, after setting all properties.
function axes7_CreateFcn(~, ~, ~)
% hObject    handle to axes7 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes7

% --- Executes during object creation, after setting all properties.
function text3_CreateFcn(~, ~, ~)
% hObject    handle to text3 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

function edit1_Callback(~, ~, ~)
% hObject    handle to edit1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double

% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, ~, ~)
% hObject    handle to edit1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function text4_CreateFcn(~, ~, ~)
% hObject    handle to text4 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text6_CreateFcn(~, ~, ~)
% hObject    handle to text6 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text8_CreateFcn(~, ~, ~)
% hObject    handle to text8 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text9_CreateFcn(~, ~, ~)
% hObject    handle to text9 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes on button press in togglebutton1.
function togglebutton1_Callback(~, ~, ~)
% hObject    handle to togglebutton1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton1

% --- Executes during object creation, after setting all properties.
function text13_CreateFcn(~, ~, ~)
% hObject    handle to text13 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text14_CreateFcn(~, ~, ~)
% hObject    handle to text14 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text15_CreateFcn(~, ~, ~)
% hObject    handle to text15 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text16_CreateFcn(~, ~, ~)
% hObject    handle to text16 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text36_CreateFcn(~, ~, ~)
% hObject    handle to text36 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text37_CreateFcn(~, ~, ~)
% hObject    handle to text37 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text47_CreateFcn(~, ~, ~)
% hObject    handle to text47 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text49_CreateFcn(~, ~, ~)
% hObject    handle to text49 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text50_CreateFcn(~, ~, ~)
% hObject    handle to text50 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text46_CreateFcn(~, ~, ~)
% hObject    handle to text46 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text45_CreateFcn(~, ~, ~)
% hObject    handle to text45 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text54_CreateFcn(~, ~, ~)
% hObject    handle to text54 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called




%% --------------------- LOAD VARIABLE ---------------------------- %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% --- Executes on button press in pushbutton1.
function pushbutton10_Callback(hObject, ~, handles)
% hObject    handle to pushbutton1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[variable_file, variable_path] = uigetfile('MultiSelect', 'on');

if size(variable_file, 2) == 2
    
    variable_to_load_1 = strcat(variable_path, variable_file{1});
    variable_to_load_2 = strcat(variable_path, variable_file{2});
    load(variable_to_load_1)
    load(variable_to_load_2)
    Y = I_OFDM1 + (sqrt(-1))*Q_OFDM1;
    set(handles.text54, 'String', strcat(variable_file{1}, ", ", variable_file{2}));
    
else
    
    variable_to_load = strcat(variable_path, variable_file);
    load(variable_to_load)
    set(handles.text54, 'String', variable_file);
    
end

handles = guidata(hObject);
handles.Y = Y;
guidata(hObject, handles);


%% --------------------- START DECODING ---------------------------- %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% --- Executes on button press in pushbutton3.
function pushbutton8_Callback(hObject, ~, handles)
% hObject    handle to pushbutton3 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = guidata(hObject);

Y = handles.Y;

if size(Y, 2) ~= 1
    Y = Y.';
end

%% Parameters
%Lookup table
rate_tb = [1101; 1111; 0101; 0111; 1001; 1011; 0001; 0011];
data_rate_tb = [6, 9, 12, 18, 24, 36, 48, 54]';
modulation = [{'BPSK'}, {'BPSK'}, {'QPSK'}, {'QPSK'}, {'16-QAM'}, {'16-QAM'}, {'64-QAM'}, {'64-QAM'}]';
coding_rates = [1/2, 3/4, 1/2, 3/4, 1/2, 3/4, 2/3, 3/4]';
coding_rates_char = {'1/2', '3/4', '1/2', '3/4', '1/2', '3/4', '2/3', '3/4'};
nbpsc_tb = [1 1 2 2 4 4 6 6]';
ncbps_tb = [48, 48, 96, 96, 192, 192, 288, 288]';
ndbps_tb = ncbps_tb.*coding_rates;
normalize_modulation = [1 1 2 2 10 10 42 42];

wifi_tb = table(rate_tb,...
                data_rate_tb,...
                modulation,...
                coding_rates,...
                nbpsc_tb,...
                ncbps_tb,...
                ndbps_tb);
wifi_tb.Properties.VariableNames = {'Rate',...
                                    'Data_Rate',...
                                    'Modulation',...
                                    'Coding_Rate',...
                                    'NBPSC',...
                                    'NCBPS',...
                                    'NDBPS'};
%Receiver Parameters
Fs = 20e6;
Ts = 1/Fs;
Mt_new = 160;

%Frame lengths in time
LT_len_t = 3.2e-6;
GI_len_t = .8e-6;

%Frame lengths in sample
LT_len = round(LT_len_t*Fs);
GI_len = round(GI_len_t*Fs);
len_ofdm = 80;

%Pilot Polarity
pilot_polarity = [  1 1 1 1,...
                    -1 -1 -1 1,...
                    -1 -1 -1 -1,...
                    1 1 -1 1,...
                    -1 -1 1 1,...
                    -1 1 1 -1,...
                    1 1 1 1,...
                    1 1 -1 1,...
                    1 1 -1 1,...
                    1 -1 -1 1,...
                    1 1 -1 1,...
                    -1 -1 -1 1,...
                    -1 1 -1 -1,...
                    1 -1 -1 1,...
                    1 1 1 1,...
                    -1 -1 1 1,...
                    -1 -1 1 -1,...
                    1 -1 1 1,...
                    -1 -1 -1 1,...
                    1 -1 -1 -1,...
                    -1 1 -1 -1,...
                    1 -1 1 1,...
                    1 1 -1 1,...
                    -1 1 -1 1,...
                    -1 -1 -1 -1,...
                    -1 1 -1 1,...
                    1 -1 1 -1,...
                    1 1 1 -1,...
                    -1 1 -1 -1,...
                    -1 1 1 1,...
                    -1 -1 -1 -1,...
                    -1 -1 -1]';
pilot_wifi = [1 1 1 -1]';

%Long Preamble frame
LT_frame_org = [1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 0,...
                1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1,...
                1 1 1 1];
LT_frame_re = [LT_frame_org(27:end) LT_frame_org(1:26)]';
LT_frame_re(1) = [];

%% CRC-32 check on '123456789'
poly = [1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 1 0 1 1 1];
len_poly = length(poly);

data_hex_no_crc = '313233343536373839';

msg = hexToBinaryVector(data_hex_no_crc);

%Processing Message for CRC calculation
data = [zeros(1, ceil(length(msg)/8)*8-length(msg)) msg];
data = fliplr(reshape(data, 8, length(data)/8)');
data = data';
data = data(:)';
data = [data zeros(1, len_poly - 1)];

%XORing first 32 bits with 0xFFFFFFFF
data = [xor(data(1:32), ones(1, 32)) data(33:end)];

%Binary division
xor_with_poly = bitxor(data(1:len_poly), poly);
data(1:len_poly) = [];

while(1)
    
    cur_data_len = length(data);
    num_zeros_xor = find(xor_with_poly == 1, 1) - 1;
    
    if ~sum(xor_with_poly)
        num_zeros_xor = length(xor_with_poly);
    end
    
    xor_with_poly(1:num_zeros_xor) = [];
    
    len_xor = length(xor_with_poly);
    diff_poly_xor = len_poly - len_xor;
    
    if diff_poly_xor > cur_data_len
        
        xor_with_poly_2 = [xor_with_poly data];
        xor_with_poly = xor_with_poly_2;
        clear xor_with_poly_2    
        break
        
    else
                
        append_xor = data(1:diff_poly_xor);
        new_xor = [xor_with_poly append_xor];

    end
    
    xor_with_poly = bitxor(poly, new_xor);
    data(1:diff_poly_xor) = [];
    
end

CRC = xor_with_poly;
CRC = [zeros(1, len_poly - length(CRC) - 1) CRC];

%XORing CRC with 0xFFFFFFFF and reflecting
CRC = fliplr(xor(CRC, ones(1, 32)));
CRC_check_hex_calc = binaryVectorToHex(CRC);


%%------- RECEIVER PROCESS START - SIGNAL DETECTION AND TRUNCATION ---------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Normalizing the signal
Y = Y./max(abs(Y));
t_y = ((1:length(Y)) - 1)*Ts;

%% Determinig signal positions
ro = 0.9;
p = zeros(size(Y));
len_p = length(p);
p(1) = abs(Y(1))^2;
for i = 2:len_p
    
    p(i) = (1 - ro)*abs(Y(i))^2 + ro*p(i - 1);
    
end

%Selected threshold
thr = 0.003;
margin = thr*ones(size(p));
m = p - margin;
m = -(0.5)*[sign(m(1:end-1)) - sign(m(2:end)); 0];

sig_start = find(m == 1);
sig_end = find(m == -1);

if ~isempty(sig_end) && ~isempty(sig_start) && sig_end(1) < sig_start(1)
    sig_end(1) = [];
end

if ~isempty(sig_end) && ~isempty(sig_start) && sig_end(end) < sig_start(end)
    sig_start(end) = [];
end

% Siscard signals smaller than 400 samples or 400*Ts seconds
sig_duration = sig_end - sig_start;
sig_start(sig_duration < 400) = [];
sig_end(sig_duration < 400) = [];
num_sig = length(sig_start);

%%------------------- INTIALIZE STRUCTURE -------------------%%

signal(num_sig).data_rate_output = [];


%%------------------- MAIN FOR LOOP GOING THROUGH ALL THE DETECTED SIGNALS -------------------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


valid_OFDM_sig = 0;

if ~isempty(sig_end) && ~isempty(sig_start)

    for sig_ind = 1:num_sig
        
        if sig_start(sig_ind) > 10
            start_limit = sig_start(sig_ind) - 10;
        else
            start_limit = sig_ind;
        end

        end_limit = sig_end(sig_ind);

        Y_temp = Y(start_limit:end_limit);

        %Frame Synchronization
        len_y_temp = length(Y_temp);
        ac_bound = len_y_temp - 160 - 1;
        comp_sig = zeros(1, ac_bound);
        comp_sig_norm = comp_sig;

        for k = 1:ac_bound

            frame = Y_temp(k:k + 160 - 1);

            f1 = frame(1:144);
            f2 = frame(17:160);

            comp_sig(k) = sum((f1).*conj(f2));
            comp_sig_norm(k) = abs(comp_sig(k)).^2/(160*sum(abs(frame).^2));

        end
        
        [~, f_ind_temp] = max(comp_sig_norm);

        f_ind = start_limit + f_ind_temp - 1;
        t_ind = f_ind*Ts*1e3;

        %CFO estimation
        f_ind_LT = f_ind + Mt_new + 2*GI_len;

        LT_frame_1 = Y(f_ind_LT:f_ind_LT + LT_len - 1);
        LT_frame_2 = Y(f_ind_LT + LT_len:f_ind_LT + 2*LT_len - 1);

        Y_long = sum(LT_frame_1.*conj(LT_frame_2));
        f_offset = angle(Y_long)/(pi*2*LT_len*Ts);

        %CFO compensation of only SIGNAL frame - DATA FRAME CFO Compensation is done
        %later
        r_sig_SP = Y(f_ind_LT:f_ind_LT + 207);

        f_offset_angle = exp(sqrt(-1)*(f_ind_LT:f_ind_LT + 207)...
                            *2*pi*Ts*f_offset).';
        r_sig_CFO = r_sig_SP.*f_offset_angle;

        LT_frame_1 = r_sig_CFO(1:64);
        LT_frame_2 = r_sig_CFO(65:128);

        %Channel estimation from LT frame
        LT_frame_1_52 = fft(LT_frame_1);
        LT_frame_1_52([1, 28:38]) = [];

        LT_frame_2_52 = fft(LT_frame_2);
        LT_frame_2_52([1, 28:38]) = [];

        %Estimating channel coefficients
        channel_coeff_LT_1 = LT_frame_1_52./LT_frame_re;
        channel_coeff_LT_2 = LT_frame_2_52./LT_frame_re;
        channel_coeff_LT = 0.5*(channel_coeff_LT_1 + channel_coeff_LT_2);

        %% EXTRACTING SIGNAL BLOCK
        ERP = r_sig_CFO(145:208);
        ERP_freq = fft(ERP);

        %Removing Null subcarriers
        ERP_freq([1, 28:38]) = [];

        %% RF AND PHASE OFFSET COMPENSATION TO SIGNAL BLOCK
        %RF compensation
        ERP_freq_RF = ERP_freq./channel_coeff_LT;

        %Phase offset compensation
        ERP_pilot = [1 1 1 -1]';
        ERP_freq_pilot = ERP_freq_RF([32 46 7 21]);
        sig_phase = sum(ERP_freq_pilot./ERP_pilot)/4;
        ERP_freq_RF_po = ERP_freq_RF.*conj(sig_phase);
        ERP_freq_RF_po = ERP_freq_RF_po./abs(sig_phase);

        %% Rearranging the signal from [1-52] to [-26-26]
        ERP_freq_RF_po = [ERP_freq_RF_po(27:end).' ERP_freq_RF_po(1:26).'];

        %% Demodulation of BPSK symbols to bit - SIGNAL BLOCK
        ERP_th = real(ERP_freq_RF_po);
        ERP_th(ERP_th > 0) = 1;
        ERP_th(ERP_th < 0) = 0;

        %% DE-INTERLEAVER - SIGNAL BLOCK
        %Removing the pilots
        ERP_th([6 20 33 47]) = [];

        %Deinterleaver
        %Parameters
        Ndatasc = 48;
        ncbps = Ndatasc*1;
        ncpc = 1;

        s = max(ncpc/2, 1);
        j = 0:ncbps - 1;

        %First de-permutation of de-interleaver
        i = s*floor(j/s) + mod(j + floor(16*j/ncbps), s);

        %second de-permutation of de-interleaver
        k = 16*i - (ncbps - 1)*floor(16*i/ncbps);

        ERP_dei = zeros(1, length(k));
        ERP_dei(k + 1) = ERP_th(j + 1);

        %% VITERBI DECODER - SIGNAL BLOCK
        trellis = poly2trellis(7, [133 171]);
        decoded = vitdec(ERP_dei, trellis, 1, 'term', 'hard');

        %% Signal Field Information
        %Parity check
        parity_even = ~mod(sum(decoded(1:end - 6)), 2);

        rate_wifi = decoded(1:4);
        len_OFDM_bin = decoded(17:-1:6);
        len_OFDM = bin2dec(num2str(len_OFDM_bin));

        %% Select Parameters from Look-up Table
        rate_wifi_tb = str2double(strrep(num2str(rate_wifi), ' ', ''));
        tb_ind = find(rate_tb == rate_wifi_tb);

        data_rate = data_rate_tb(tb_ind);
        ndbps = ndbps_tb(tb_ind);
        n_bits_per_sym = nbpsc_tb(tb_ind);
        coding_rate = coding_rates(tb_ind);
        coding_rate_char = coding_rates_char(tb_ind);
        modulation_scheme = modulation(tb_ind);

        %% Calculating total length of Data and extracting the signal
        no_OFDM_symbols = ceil((16 + 8*len_OFDM + 6)/ndbps);

        %% IF NO OFDM SYMBOLS FOUND, CONTINUE TO THE NEXT SIGNAL
        if isempty(no_OFDM_symbols)
            continue
        end

        %% Extracting the OFDM symbols
        data_ind = f_ind_LT + 128 + 80;

        if length(Y) > data_ind + no_OFDM_symbols*len_ofdm - 1
            val_OFDM = Y(data_ind:data_ind + no_OFDM_symbols*len_ofdm - 1);
        else
            continue
        end

        %CFO correction of DATA symbol
        f_OFDM_offset_angle = exp(sqrt(-1)*(data_ind:data_ind + no_OFDM_symbols*len_ofdm - 1)...
                            *2*pi*Ts*f_offset).';
        val_OFDM = val_OFDM.*f_OFDM_offset_angle;
        val_OFDM = reshape(val_OFDM, len_ofdm, no_OFDM_symbols);

        %% Constellation of the DATA OFDM blocks
        %Removing CP
        val_OFDM(1:16, :) = [];

        %Taking FFT
        OFDM_freq = fft(val_OFDM);

        %Removing Null subcarriers
        OFDM_freq([1, 28:38],:) = [];

        %RF compensation
        OFDM_RF = OFDM_freq./channel_coeff_LT;

        %Phase offset compensation
        pilot_wifi_rep = repmat(pilot_wifi, 1, no_OFDM_symbols);

        pilot_polarity_ext = repmat(pilot_polarity, ceil(no_OFDM_symbols/127), 1);

        pilot_polarity_rep = repmat(pilot_polarity_ext(2:no_OFDM_symbols + 1).', 4, 1);
        pilot_OFDM = pilot_wifi_rep.*pilot_polarity_rep;
        OFDM_pilot = OFDM_RF([32 46 7 21],:);
        OFDM_phase = sum(OFDM_pilot./pilot_OFDM)/4;
        OFDM_RF_po = OFDM_RF.*conj(OFDM_phase);
        OFDM_RF_po = OFDM_RF_po./abs(OFDM_phase);

        %% Rearranging the data from [1-52] to [-26-26]
        OFDM_RF_po = [OFDM_RF_po(27:end,:); OFDM_RF_po(1:26,:)];

        %Removing the pilots
        OFDM_RF_po([6 20 33 47], :) = [];

        %Normalize
        OFDM = OFDM_RF_po.*sqrt(normalize_modulation(tb_ind));

        %% Demodulation of Symbols to bits - DATA BLOCK
        data = zeros(n_bits_per_sym, 48*no_OFDM_symbols);
        
        %Quantize the symbols
        if (tb_ind == 1) || (tb_ind == 2)

            OFDM_real = real(OFDM);
            data(OFDM_real > 0) = 1;
            data(OFDM_real < 0) = 0;

        else
            
            %Demodulate using encode table
            if (tb_ind == 3) || (tb_ind == 4)

                modulation_lookup_tb = [0; 1]';
                levels = -1:2:1;

            elseif (tb_ind == 5) || (tb_ind == 6)

                modulation_lookup_tb = [0,0; 0,1; 1,1; 1,0]';
                levels = -3:2:3;

            elseif (tb_ind == 7) || (tb_ind == 8)

                modulation_lookup_tb = [0,0,0; 0,0,1; 0,1,1; 0,1,0; 1,1,0; 1,1,1; 1,0,1; 1,0,0]';
                levels = -7:2:7;

            end

            OFDM_freq_re = real(OFDM(:));
            OFDM_freq_im = imag(OFDM(:));

            OFDM_re_diff = abs(OFDM_freq_re - levels);
            OFDM_im_diff = abs(OFDM_freq_im - levels);

            [~, OFDM_re_min_ind] = min(OFDM_re_diff, [], 2);
            [~, OFDM_im_min_ind] = min(OFDM_im_diff, [], 2);

            OFDM_re_q = levels(OFDM_re_min_ind);
            OFDM_im_q = levels(OFDM_im_min_ind);

            levels = repmat(levels.', 1, length(OFDM_re_q));

            OFDM_re_q_mod = repmat(OFDM_re_q, size(levels, 1), 1);
            OFDM_im_q_mod = repmat(OFDM_im_q, size(levels, 1), 1);

            re_ind = find(OFDM_re_q_mod == levels);
            im_ind = find(OFDM_im_q_mod == levels);

            [re_ind, ~] = ind2sub(size(OFDM_re_q_mod), re_ind);
            [im_ind, ~] = ind2sub(size(OFDM_im_q_mod), im_ind);

            data(1:size(data, 1)/2, :) = modulation_lookup_tb(:, re_ind);
            data((size(data, 1)/2) + 1:end, :) = modulation_lookup_tb(:, im_ind);

            data = data(:);

        end

        %% Deinterleaving - DATA BLOCK
        data = reshape(data, Ndatasc*n_bits_per_sym, no_OFDM_symbols);
        data_dei = zeros(size(data));

        mod_typ = n_bits_per_sym;
        ncbps = Ndatasc*mod_typ;
        ncpc = n_bits_per_sym;

        k = 0:ncbps - 1;
        s = ceil(ncpc/2);

        %First de-permutation of de-interleaver
        i = s*floor(k/s) + mod(k + floor(16*k/ncbps), s);

        %second de-permutation of de-interleaver
        j = 16*i - (ncbps - 1)*floor(16*i/ncbps);

        data_dei(j + 1,:) = data(k + 1,:);

        %% Puncturing and Viterbi decoding - DATA BLOCK
        decoder_input = data_dei(:);

        %Decode
        trellis = poly2trellis(7, [133 171]);

        switch coding_rate
            case 1/2
                decoder_output = vitdec(decoder_input, trellis, 35, 'trunc', 'hard');
            case 2/3
                puncpat = [1,1,1,0,1,1,1,0,1,1,1,0];
                decoder_output = vitdec(decoder_input, trellis, 35, 'trunc', 'hard', puncpat);
            case 3/4
                puncpat = [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1];
                decoder_output = vitdec(decoder_input, trellis, 35, 'trunc', 'hard', puncpat);
        end

        %% Descrambler - DATA BLOCK
        decoder_output = decoder_output(:)';

        init = (decoder_output(1:7));
        seq = zeros(1, 127);

        for i = 1:127

            init_4 = init(4);
            init_7 = init(1);

            init_4_7_xor = xor(init_7, init_4);

            seq(i) = init_4_7_xor;

            init(1:end-1) = init(2:end);
            init(7) = init_4_7_xor;

        end

        N = length(decoder_output) - 7;
        n = floor(N/127);
        n_extra_bits = N - n*127;

        seq_repeated = [init repmat(seq, 1, n) seq(1:n_extra_bits)];

        data_est = bitxor(decoder_output, seq_repeated);

        %% CRC Check - DATA BLOCK
        n_data = length(data_est);
        n_pad = n_data - (16 + 8*len_OFDM + 6);

        %Pad bits
        data_est(end - n_pad + 1:end) = [];

        %Tail bits
        data_est(end - 5:end) = [];

        %Service bits
        data_est(1:16) = [];

        %Data bit in HEX form
        data_bin = logical(reshape(data_est, 8, len_OFDM)).';
        data_hex = binaryVectorToHex(data_bin, 'LSBfirst');

        data_hex = cell2mat(data_hex)';
        data_hex = data_hex(:)';
        data_hex_no_crc = data_hex(1:end - 8);

        msg = hexToBinaryVector(data_hex_no_crc);

        %Processing Message for CRC calculation
        data = [zeros(1, ceil(length(msg)/8)*8-length(msg)) msg];
        data = fliplr(reshape(data, 8, length(data)/8)');
        data = data';
        data = data(:)';

        data_2 = [data zeros(1, len_poly - 1)];
        data = data_2;
        clear data_2

        %XORing first 32 bits with 0xFFFFFFFF
        data = [xor(data(1:32), ones(1, 32)) data(33:end)];

        %Binary division
        xor_with_poly = bitxor(data(1:len_poly), poly);
        data(1:len_poly) = [];

        while(1)

            cur_data_len = length(data);
            num_zeros_xor = find(xor_with_poly == 1, 1) - 1;

            if ~sum(xor_with_poly)
                num_zeros_xor = length(xor_with_poly);
            end

            xor_with_poly(1:num_zeros_xor) = [];

            len_xor = length(xor_with_poly);
            diff_poly_xor = len_poly - len_xor;

            if diff_poly_xor > cur_data_len

                xor_with_poly_2 = [xor_with_poly data];
                xor_with_poly = xor_with_poly_2;
                clear xor_with_poly_2
                break

            else

                append_xor = data(1:diff_poly_xor);
                new_xor = [xor_with_poly append_xor];

            end

            xor_with_poly = bitxor(poly, new_xor);
            data(1:diff_poly_xor) = [];

        end

        CRC = xor_with_poly;
        CRC_2 = [zeros(1, len_poly - length(CRC) - 1) CRC];
        CRC = CRC_2;
        clear CRC_2

        %XORing CRC with 0xFFFFFFFF and reflecting
        CRC = fliplr(xor(CRC, ones(1, 32)));
        CRC_hex_calc = binaryVectorToHex(CRC);

        %CRC extraction from FCS
        CRC_rec = rot90(reshape(data_est(end-31:end), 4, 8)', 2);
        CRC_rec = binaryVectorToHex(CRC_rec);
        CRC_rec = cell2mat(CRC_rec)';

        if isequal(CRC_hex_calc, CRC_rec)
            CRC_match = 'Yes';
        else
            CRC_match = 'No';
        end

        %% IF CRC DOESN'T MATCH, CONTINUE WITH THE NEXT SIGNAL %%
        if ~strcmp(CRC_match, 'Yes')
            continue
        end

        %% Extract data field information - Frame control, Duration, SA, DA, BSSID, SSID, Basic and Supported Rates  
        %Frame Type
        frame_type = fliplr(data_est(3:4));

        %Frame Sub-Type
        frame_sub_type = fliplr(data_est(5:8));
        frame_sub_type_char = 'Subtype not defined';

        if isequal(frame_type, [0 0])
            frame_type_char = 'Management Frame';

            if isequal(frame_sub_type, [1 0 0 0])
                frame_sub_type_char = 'Beacon Frame';
            elseif isequal(frame_sub_type, [1 0 1 0])
                frame_sub_type_char = 'Disassociation Frame';
            end    

        elseif isequal(frame_type, [0 1])
            frame_type_char = 'Control Frame';

            if isequal(frame_sub_type, [1 0 1 0])
                frame_sub_type_char = 'Block ACK';
            elseif isequal(frame_sub_type, [1 0 1 1])
                frame_sub_type_char = 'RTS Frame';
            elseif isequal(frame_sub_type, [1 1 0 1])
                frame_sub_type_char = 'ACK Frame';
            end    
            
        elseif isequal(frame_type, [1 0])
            frame_type_char = 'Data Frame';

            if isequal(frame_sub_type, [0 0 0 0])
                frame_sub_type_char = 'Data Frame';
            elseif isequal(frame_sub_type, [0 1 0 0])
                frame_sub_type_char = 'Null (no data)';
            elseif isequal(frame_sub_type, [1 0 0 0])
                frame_sub_type_char = 'QoS Data Frame';
            elseif isequal(frame_sub_type, [1 1 0 0])
                frame_sub_type_char = 'QoS Null (no data)';
            end    
        end

        %To DS Flag
        to_ds = data_est(9);
        to_ds_c = mat2str(to_ds);

        %From DS Flag
        from_ds = data_est(10);
        from_ds_c = mat2str(from_ds);

        %Duration
        duration = hex2dec(binaryVectorToHex(fliplr(data_est(17:32))));

        %Address
        data_hex_add = reshape(data_hex, 2, length(data_hex)/2)';
        div = " : ";
        DA = "";
        SA = "";
        BSSID = "";
        if isequal(frame_type, [0 1]) && isequal(frame_sub_type, [1 1 0 1])

            DA = data_hex_add(5:10, :);
            DA = DA';
            DA = DA(:)';

            DA = strcat((DA(1:2)), div,...
                        (DA(3:4)), div,...
                        (DA(5:6)), div,...
                        (DA(7:8)), div,...
                        (DA(9:10)), div,...
                        (DA(11:12)));
            BSSID = "";
            SA = "";

        elseif isequal(frame_type, [0 1]) && isequal(frame_sub_type, [1 0 1 1])

            DA = data_hex_add(5:10, :);
            DA = DA';
            DA = DA(:)';
            DA = strcat((DA(1:2)), div,...
                        (DA(3:4)), div,...
                        (DA(5:6)), div,...
                        (DA(7:8)), div,...
                        (DA(9:10)), div,...
                        (DA(11:12)));

            SA = data_hex_add(11:16, :);
            SA = SA';
            SA = SA(:)';
            SA = strcat((SA(1:2)), div,...
                        (SA(3:4)), div,...
                        (SA(5:6)), div,...
                        (SA(7:8)), div,...
                        (SA(9:10)), div,...
                        (SA(11:12)));        

            BSSID = "";

        elseif     (isequal(frame_type, [0 0]) && isequal(frame_sub_type, [1 0 0 0]))...
                || (isequal(frame_type, [0 0]) && isequal(frame_sub_type, [1 0 1 0]))...
                || (isequal(frame_type, [1 0]) && isequal(frame_sub_type, [1 0 0 0]))...
                || (isequal(frame_type, [1 0]) && isequal(frame_sub_type, [0 0 0 0]))

            switch [to_ds_c, from_ds_c]
                case ['0', '0']
                    DA = data_hex_add(5:10, :);
                    SA = data_hex_add(11:16, :);
                    BSSID = data_hex_add(17:22, :);            
                case ['0', '1']
                    DA = data_hex_add(5:10, :);
                    BSSID = data_hex_add(11:16, :);
                    SA = data_hex_add(17:22, :);
                case ['1', '0']
                    BSSID = data_hex_add(5:10, :);
                    SA = data_hex_add(11:16, :);
                    DA = data_hex_add(17:22, :);
            end

            DA = DA';
            DA = DA(:)';
            DA = strcat((DA(1:2)), div,...
                        (DA(3:4)), div,...
                        (DA(5:6)), div,...
                        (DA(7:8)), div,...
                        (DA(9:10)), div,...
                        (DA(11:12)));
                    
            SA = SA';
            SA = SA(:)';
            SA = strcat((SA(1:2)), div,...
                        (SA(3:4)), div,...
                        (SA(5:6)), div,...
                        (SA(7:8)), div,...
                        (SA(9:10)), div,...
                        (SA(11:12)));
                    
            BSSID = BSSID';
            BSSID = BSSID(:)';
            BSSID = strcat( (BSSID(1:2)), div,...
                            (BSSID(3:4)), div,...
                            (BSSID(5:6)), div,...
                            (BSSID(7:8)), div,...
                            (BSSID(9:10)), div,...
                            (BSSID(11:12)));
        end

        %SSID and Supported Rate information element extraction
        ssid = "";
        sup_rate_char = "";
        basic_rate_char = "";
        if length(data_hex) > 77

            id_element = data_hex(73:74);
            id_element = hex2dec(id_element);

            if id_element == 0

                ssid_len = hex2dec(data_hex(75:76));

                if ssid_len > 32
                    continue
                end

                ssid = data_hex(77:77 + 2*ssid_len - 1);
                ssid = reshape(ssid, 2, ssid_len)';
                ssid = char(hex2dec(ssid))';

                %Basic and Supported Rates
                if length(data_hex) > 77 + 2*ssid_len

                    id_element = data_hex(77 + 2*ssid_len:77 + 2*ssid_len + 1);
                    id_element = hex2dec(id_element);

                    if id_element == 1

                        sup_rate_len = hex2dec(data_hex(77 + 2*ssid_len + 2:77 + 2*ssid_len + 3));
                        sup_rate = data_hex(77 + 2*ssid_len + 4:77 + 2*ssid_len + 4 + 2*sup_rate_len - 1);

                        sup_rate = reshape(sup_rate, 2, sup_rate_len)';
                        sup_rate = dec2hex(sort(hex2dec(sup_rate)));
                        sup_rate = hexToBinaryVector(sup_rate);

                        n_basic = sum(sup_rate(:, 1));
                        n_sup = sup_rate_len - n_basic;

                        c_sup = 0;
                        c_bas = 0;

                        for i = 1:sup_rate_len
                            sup_rate_row = sup_rate(i, :);

                            if ~sup_rate_row(1)

                                c_sup = c_sup + 1;
                                sup_rate_row = sup_rate_row(find(sup_rate_row, 1)+1:end);
                                sup_rate_row = num2str(round(.5*bi2de(sup_rate_row, 'left-msb')));
                                sup_rate_char = strcat(sup_rate_char, sup_rate_row);

                                if c_sup ~= n_sup
                                    sup_rate_char = strcat(sup_rate_char, ", ");
                                else
                                    sup_rate_char = strcat(sup_rate_char, " Mbps");
                                end
                            else

                                c_bas = c_bas + 1;
                                basic_rate_row = sup_rate_row(find(sup_rate_row, 1)+1:end);
                                basic_rate_row = num2str(round(.5*bi2de(basic_rate_row, 'left-msb')));
                                basic_rate_char = strcat(basic_rate_char, basic_rate_row);
                                
                                if c_bas ~= n_basic
                                    basic_rate_char = strcat(basic_rate_char, ", ");
                                else
                                    basic_rate_char = strcat(basic_rate_char, " Mbps");
                                end
                            end
                        end   
                    end
                end
            end
        end

        %Hex dump processing for display
        data_hex_output = reshape(data_hex, 2, len_OFDM)';
        num_pad_groups = ceil(len_OFDM/16)*16 - len_OFDM;
        data_hex_output_2 = [data_hex_output; repmat('  ', num_pad_groups, 1)];
        data_hex_output = data_hex_output_2;
        clear data_hex_output_2

        data_hex_output = data_hex_output';
        data_hex_output = data_hex_output(:)';
        data_hex_output = reshape(data_hex_output, 16, length(data_hex_output)/16);
        data_hex_output_2 = [data_hex_output; repmat(' ', 2, size(data_hex_output, 2))];
        data_hex_output = data_hex_output_2;
        data_hex_output = data_hex_output(:)';
        data_hex_output = reshape(data_hex_output, 2, length(data_hex_output)/2)';
        clear data_hex_output_2

        new_len_OFDM = length(data_hex_output);
        new_num_pad_groups = ceil(new_len_OFDM/18)*18 - new_len_OFDM;
        data_hex_output_2 = [data_hex_output; repmat('  ', new_num_pad_groups, 1)];
        data_hex_output = data_hex_output_2;
        clear data_hex_output_2

        data_hex_output = mat2cell(data_hex_output, ones(1, size(data_hex_output, 1)));
        data_hex_output = reshape(data_hex_output, 18, numel(data_hex_output)/18)';

        %% COUNT THE CURRENT SIGNAL AS A VALID OFDM SIGNAL %%
        valid_OFDM_sig = valid_OFDM_sig + 1;

        %% Saving current packet information in 'signal' structure
        signal(valid_OFDM_sig).data_rate_output = strcat(num2str(data_rate), " Mbps");
        if parity_even
            signal(valid_OFDM_sig).parity_check = 'Even';
        else
            signal(valid_OFDM_sig).parity_check = 'Odd';
        end
        signal(valid_OFDM_sig).no_octets_output = strcat(num2str(len_OFDM));
        signal(valid_OFDM_sig).no_sym_output = strcat(num2str(no_OFDM_symbols));
        signal(valid_OFDM_sig).coding_rate_char = strcat(cell2mat(coding_rate_char));
        signal(valid_OFDM_sig).mod_scheme_char = strcat(cell2mat(modulation_scheme));
        signal(valid_OFDM_sig).ncbps_char = mat2str(ncbps);
        signal(valid_OFDM_sig).ndbps_char = mat2str(ndbps);
        signal(valid_OFDM_sig).frame_type_char = frame_type_char;
        signal(valid_OFDM_sig).frame_sub_type_char = frame_sub_type_char;
        signal(valid_OFDM_sig).CRC_match = CRC_match;
        signal(valid_OFDM_sig).CRC_check_hex_calc = strcat('0x', CRC_check_hex_calc);
        signal(valid_OFDM_sig).to_ds = to_ds;
        signal(valid_OFDM_sig).from_ds = from_ds;
        signal(valid_OFDM_sig).BSSID = BSSID;
        signal(valid_OFDM_sig).DA = DA;
        signal(valid_OFDM_sig).SA = SA;
        signal(valid_OFDM_sig).duration = strcat(num2str(duration), " ", char(181), "s");
        signal(valid_OFDM_sig).t_ind = strcat(num2str(round(t_ind, 2)), " ms");
        signal(valid_OFDM_sig).data_hex_output = data_hex_output;
        signal(valid_OFDM_sig).ssid = ssid;
        signal(valid_OFDM_sig).sup_rate_char = sup_rate_char;
        signal(valid_OFDM_sig).basic_rate_char = basic_rate_char;
        %Plots
        signal(valid_OFDM_sig).OFDM = OFDM;
        signal(valid_OFDM_sig).corr_sig = comp_sig_norm;
        signal(valid_OFDM_sig).start_limit = start_limit;
        signal(valid_OFDM_sig).ac_bound = ac_bound;
        signal(valid_OFDM_sig).Y = real(Y);
        signal(valid_OFDM_sig).tY = t_y;
        Y_trunc = Y;
        Y_trunc(1:start_limit - 1) = NaN;
        Y_trunc(end_limit + 1:end) = NaN;
        signal(valid_OFDM_sig).Y_trunc = real(Y_trunc);
        signal(valid_OFDM_sig).ERP_freq_RF_po = ERP_freq_RF_po;
        signal(valid_OFDM_sig).channel_coeff_LT = unwrap(angle(channel_coeff_LT));
        signal(valid_OFDM_sig).CRC_check_hex_calc = CRC_check_hex_calc;

    end

    for i = 1:valid_OFDM_sig
        signal(i).valid_OFDM_sig = valid_OFDM_sig;
    end
    
    %Truncate the last empty entries in the structure
    signal(valid_OFDM_sig + 1:end) = [];

    %% GUI Functions
    ind = 1;
    if valid_OFDM_sig == 0

        signal(ind).Y = real(Y);
        signal(ind).tY = t_y;
        signal(ind).valid_OFDM_sig = valid_OFDM_sig;

        handles = guidata(hObject);
        handles.signal = signal;
        handles.ind = ind;
        guidata(hObject, handles);
        no_WiFi_signal(hObject);

    else

        handles = guidata(hObject);
        handles.signal = signal;
        handles.ind = ind;
        guidata(hObject, handles);
        display_info(hObject);

    end
else
    
    ind = 1;
    signal(ind).Y = real(Y);
    signal(ind).tY = t_y;
    signal(ind).valid_OFDM_sig = valid_OFDM_sig;
    
    handles = guidata(hObject);
    handles.signal = signal;
    handles.ind = ind;
    guidata(hObject, handles);
    no_WiFi_signal(hObject);
    
end

% --- Display values and plots in GUI
function display_info(hObject)
% signal   structure with information fields
% ind      index of the desired packet
handles = guidata(hObject);
signal = handles.signal;
ind = handles.ind;

%Data Constellation Plot
axes(handles.axes1)
plot(signal(ind).OFDM, 'x')
xlabel('In-phase amplitude')
ylabel('Quadrature amplitude')
grid on

%Correlation Signal Plot
axes(handles.axes2)
corr_time = signal(ind).start_limit:signal(ind).start_limit + signal(ind).ac_bound - 1;
corr_time = corr_time.*(1/20e6);
corr_time = corr_time*1e3;
plot(corr_time, signal(ind).corr_sig, 'LineWidth', 2)
xlabel('Time (ms)')
ylabel('Amplitude')
grid on
axis tight

%Received Signal and Current Packet Plot
axes(handles.axes3)
plot(signal(ind).tY, signal(ind).Y)
hold on 
plot(signal(ind).tY, signal(ind).Y_trunc)
hold off
xlabel('Time (ms)')
ylabel('Amplitude')
grid on

%Signal Field constellation plot
axes(handles.axes4)
plot(signal(ind).ERP_freq_RF_po, '.')
xlim([-1.5 1.5])
ylim([-.25 .25])
xlabel('In-phase amplitude')
ylabel('Quadrature amplitude')
grid on

%Estimated channel plot
axes(handles.axes7)
stem(signal(ind).channel_coeff_LT, '.')
xlabel('Sample (n)')
ylabel('Phase angle')
grid on
axis tight

%signal Field Information - Text display
set(handles.text6, 'String', signal(ind).data_rate_output);
set(handles.text49, 'String', signal(ind).parity_check);
set(handles.text8, 'String', signal(ind).no_octets_output);
set(handles.text9, 'String', signal(ind).no_sym_output);
set(handles.text13, 'String', signal(ind).coding_rate_char);
set(handles.text14, 'String', signal(ind).mod_scheme_char);
set(handles.text15, 'String', signal(ind).ncbps_char);
set(handles.text16, 'String', signal(ind).ndbps_char);
set(handles.text36, 'String', signal(ind).frame_type_char);
set(handles.text37, 'String', signal(ind).frame_sub_type_char);
set(handles.text47, 'String', signal(ind).CRC_match);
set(handles.text50, 'String', strcat('0x', signal(ind).CRC_check_hex_calc));
set(handles.text45, 'String', '32');
set(handles.text46, 'String', '0x04C11DB7');
set(handles.text59, 'String', signal(ind).to_ds);
set(handles.text60, 'String', signal(ind).from_ds);
set(handles.text64, 'String', signal(ind).duration);
set(handles.text65, 'String', signal(ind).BSSID);
set(handles.text66, 'String', signal(ind).DA);
set(handles.text67, 'String', signal(ind).SA);
set(handles.text78, 'String', signal(ind).t_ind);
set(handles.uitable3, 'Data', signal(ind).data_hex_output);
set(handles.uitable3, 'ColumnName',{' '});
set(handles.uitable3, 'ColumnWidth', {18});
set(handles.text90, 'String', signal(ind).ssid);
set(handles.text93, 'String', signal(ind).sup_rate_char);
set(handles.text84, 'String', signal(ind).basic_rate_char);
set(handles.text96, 'String', strcat(num2str(signal(ind).valid_OFDM_sig), '.'));
set(handles.text97, 'String', strcat(num2str(ind), " / ", num2str(signal(ind).valid_OFDM_sig)));

handles = guidata(hObject);
handles.signal = signal;
handles.ind = ind;
guidata(hObject, handles);

% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles = guidata(hObject);
signal = handles.signal;
ind = handles.ind;

if ind > 1
    
    ind = ind - 1;
    handles = guidata(hObject);
    handles.signal = signal;
    handles.ind = ind;
    guidata(hObject, handles);
    display_info(hObject)

end

% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles = guidata(hObject);
signal = handles.signal;
ind = handles.ind;

if ind ~= signal(ind).valid_OFDM_sig && signal(ind).valid_OFDM_sig ~= 0
    
    ind = ind + 1;
    handles = guidata(hObject);
    handles.signal = signal;
    handles.ind = ind;
    guidata(hObject, handles);
    display_info(hObject)

end

% --- Executes on button press in pushbutton11.
function no_WiFi_signal(hObject)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA) 
handles = guidata(hObject);
signal = handles.signal;
axes(handles.axes3)
plot(signal(1).tY, signal(1).Y)
xlabel('Time (ms)')
ylabel('Amplitude')
grid on
set(handles.text96, 'String', 0);

% --- Executes on button press in pushbutton4.
function pushbutton9_Callback(hObject, ~, handles)
% hObject    handle to pushbutton4 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clc;
cla(handles.axes1);
cla(handles.axes2);
cla(handles.axes3);
cla(handles.axes4);
cla(handles.axes7);

clear_string = [];
set(handles.text6,  'String', clear_string);
set(handles.text8,  'String', clear_string);
set(handles.text9,  'String', clear_string);
set(handles.text13, 'String', clear_string);
set(handles.text14, 'String', clear_string);
set(handles.text15, 'String', clear_string);
set(handles.text16, 'String', clear_string);
set(handles.text36, 'String', clear_string);
set(handles.text37, 'String', clear_string);
set(handles.text47, 'String', clear_string);
set(handles.text50, 'String', clear_string);
set(handles.text45, 'String', clear_string);
set(handles.text46, 'String', clear_string);
set(handles.text49, 'String', clear_string);
set(handles.text59, 'String', clear_string);
set(handles.text60, 'String', clear_string);
set(handles.text64, 'String', clear_string);
set(handles.text65, 'String', clear_string);
set(handles.text66, 'String', clear_string);
set(handles.text67, 'String', clear_string);
set(handles.text78, 'String', clear_string);
set(handles.text84, 'String', clear_string);
set(handles.text90, 'String', clear_string);
set(handles.text93, 'String', clear_string);
set(handles.text96, 'String', clear_string);
set(handles.text97, 'String', clear_string);
set(handles.uitable3, 'Data', {''});

% --- Executes during object creation, after setting all properties.
function text56_CreateFcn(hObject, ~, handles)
% hObject    handle to text56 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text59_CreateFcn(hObject, ~, handles)
% hObject    handle to text59 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text58_CreateFcn(hObject, ~, handles)
% hObject    handle to text58 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text60_CreateFcn(hObject, ~, handles)
% hObject    handle to text60 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text64_CreateFcn(hObject, ~, handles)
% hObject    handle to text64 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text65_CreateFcn(hObject, ~, handles)
% hObject    handle to text65 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text66_CreateFcn(hObject, ~, handles)
% hObject    handle to text66 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text67_CreateFcn(hObject, ~, handles)
% hObject    handle to text67 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text78_CreateFcn(hObject, ~, handles)
% hObject    handle to text78 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, ~, handles)
% hObject    handle to pushbutton6 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes when figure1 is resized.
function figure1_SizeChangedFcn(hObject, ~, handles)
% hObject    handle to figure1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes on scroll wheel click while the figure is in focus.
function figure1_WindowScrollWheelFcn(hObject, ~, handles)
% hObject    handle to figure1 (see GCBO)
% ~  structure with the following fields (see MATLAB.UI.FIGURE)
%	VerticalScrollCount: signed integer indicating direction and number of clicks
%	VerticalScrollAmount: number of lines scrolled for each click
% handles    structure with handles and user data (see GUIDATA)

% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, ~, handles)
% hObject    handle to listbox1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1

% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, ~, handles)
% hObject    handle to listbox1 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, ~, handles)
% hObject    handle to pushbutton7 (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes when entered data in editable cell(s) in uitable3.
function uitable3_CellEditCallback(hObject, ~, handles)
% hObject    handle to uitable3 (see GCBO)
% ~  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) edited
%	PreviousData: previous data for the cell(s) edited
%	EditData: string(s) entered by the user
%	NewData: EditData or its converted form set on the Data property. Empty if Data was not changed
%	Error: error string when failed to convert EditData to appropriate value for Data
% handles    structure with handles and user data (see GUIDATA)

% --- Executes during object creation, after setting all properties.
function text90_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text90 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text91_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text91 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text83_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text83 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text84_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text84 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text93_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text93 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text96_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text96 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes during object creation, after setting all properties.
function text97_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text97 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
