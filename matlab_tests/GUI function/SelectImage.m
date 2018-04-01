function SelectImage(hObject,eventdata,handles)

hfig = gcf;

[filename,pathname] = uigetfile('*.*','Select image file');
if ~ischar(filename); return; end


%%%%%%%%%%%%%%%%%%%Draw Image%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load  Image file
longfilename = strcat(pathname,filename);
img = imread(longfilename);

if ndims(img) == 3
    im=rgb2gray(img);
else
    im=img;
end

im=im2double(im);

Fig.org = axes('Parent',hfig, 'Units', 'Normalized', 'Position', [.06 .55 .4 .4]);

imshow(im)

assignin('base', 'orgimg', im);

