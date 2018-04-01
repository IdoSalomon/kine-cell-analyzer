function BinaryImage()

hfig = gcf;
img = evalin('base', 'imgproc');

% Convert to Binary Image 
precond_img=max(img,[],3);
segResult=im2bw(precond_img,graythresh(precond_img));

Fig.binimg= axes('Parent',hfig, 'Units', 'Normalized', 'Position', [.52 .06 .4 .4]);
imshow(segResult)

assignin('base', 'segResult',segResult);