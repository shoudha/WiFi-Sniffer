function varargout = wifi_gui(varargin)
%WIFI_GUI MATLAB code file for wifi_gui.fig
%      WIFI_GUI, by itself, creates a new WIFI_GUI or raises the existing
%      singleton*.
%
%      H = WIFI_GUI returns the handle to a new WIFI_GUI or the handle to
%      the existing singleton*.
%
%      WIFI_GUI('Property','Value',...) creates a new WIFI_GUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to wifi_gui_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      WIFI_GUI('CALLBACK') and WIFI_GUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in WIFI_GUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help wifi_gui

% Last Modified by GUIDE v2.5 07-Dec-2018 18:37:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @wifi_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @wifi_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
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


% --- Executes just before wifi_gui is made visible.
function wifi_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for wifi_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes wifi_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = wifi_gui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
