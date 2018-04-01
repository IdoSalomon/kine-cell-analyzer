function DebugMode(hObject, eventdata,showimg)

debug=get(Pcn.mode, 'Value');

assignin('base', 'debug', debug);