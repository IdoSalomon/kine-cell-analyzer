function precondition_img(hObject,eventdata,Pcn)

hfig = gcf;
Fig.precond= axes('Parent',hfig, 'Units', 'Normalized', 'Position', [.06 .06 .4 .4]);

debug=get(Pcn.debug, 'Value');
modeVal=get(Pcn.mode, 'Value');

if (modeVal==1)
    mode='sparse_respresent';
else 
    mode='linear_model';
end

img = evalin('base', 'orgimg');
optparas=evalin('base','optparas');
kernparas=evalin('base','kernparas');

% Sparse representation of phase contrast image  
 imgproc=precondition_phase_contrast(img,optparas,kernparas,mode,debug);


axes(Fig.precond)

imshow(imgproc)

assignin('base', 'imgproc', imgproc);








