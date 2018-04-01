function precondition_img(hObject,eventdata,Pcn)

hfig = gcf;
Fig.precond= axes('Parent',hfig, 'Units', 'Normalized', 'Position', [.06 .06 .4 .4]);

debug=get(Pcn.debug, 'Value');
mode=get(Pcn.mode, 'string');

img = evalin('base', 'orgimg');
optparas=evalin('base','optparas');
kernparas=evalin('base','kernparas');
debug=evalin('base','debug');

% Sparse representation of phase contrast image  
 imgproc=precondition_sparse_respresent(img,optparas,kernparas,debug);


axes(Fig.precond)

imshow(imgproc)

assignin('base', 'imgproc', imgproc);








