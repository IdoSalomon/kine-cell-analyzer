orig = imread('small.png');
I = imread('res2.tif');

[BW,maskedRGBImage] = createMask(I);

imshow(maskedRGBImage);

I = rgb2gray(maskedRGBImage);

[~, threshold] = edge(I, 'sobel');
fudgeFactor = .52;
BWs = edge(I,'sobel', threshold * fudgeFactor);
figure, imshow(BWs), title('binary gradient mask');

CC = bwconncomp(BWs, conndef(ndims(BWs), 'maximal'));
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
BWfinal = ismember(L, find([S.Area] >= 15));
figure, imshow(BWfinal), title('segmented image 2');

se90 = strel('line', 5, 90);
se0 = strel('line', 5, 0);
BWsdil = imdilate(BWfinal, [se90 se0]);
figure, imshow(BWsdil), title('dilated gradient mask');




BWoutline = bwperim(BWsdil);
Segout = orig; 
Segout(BWoutline) = 255; 
figure, imshow(Segout), title('outlined original image');

% 
% BWoutline = bwperim(BWfinal);
% Segout = I; 
% Segout(BWoutline) = 255; 
% figure, imshow(Segout), title('outlined original image');