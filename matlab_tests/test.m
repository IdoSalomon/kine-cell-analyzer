I = imread('test.png');
figure, imshow(I), title('original image');
text(size(I,2),size(I,1)+15, ...
    'Image courtesy of Alan Partin', ...
    'FontSize',7,'HorizontalAlignment','right');
text(size(I,2),size(I,1)+25, ....
    'Johns Hopkins University', ...
    'FontSize',7,'HorizontalAlignment','right');

[~, threshold] = edge(I, 'sobel');
fudgeFactor = .9;
BWs = edge(I,'sobel', threshold * fudgeFactor);
figure, imshow(BWs), title('binary gradient mask');

se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
figure, imshow(BWsdil), title('dilated gradient mask');

% BWsdil = imread('res.tif');



BWdfill = imfill(BWsdil, 'holes');
figure, imshow(BWdfill);
title('binary image with filled holes');



%%BWnobord = imclearborder(BWdfill, 4);
%figure, imshow(BWnobord), title('cleared border image');



seD = strel('diamond',1);
BWfinal = imerode(BWdfill,seD);
BWfinal = imerode(BWfinal,seD);
figure, imshow(BWfinal), title('segmented image');

CC = bwconncomp(BWfinal, conndef(ndims(BWfinal), 'maximal'));
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
BWfinal = ismember(L, find([S.Area] >= 50));
figure, imshow(BWfinal), title('segmented image 2');

BWoutline = bwperim(BWfinal);
Segout = I; 
Segout(BWoutline) = 255; 
figure, imshow(Segout), title('outlined original image');