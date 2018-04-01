I = imread('res.tif');

se = strel('ball',5, 5);

dilatedI = imdilate(I,se);

imshowpair(I,dilatedI,'montage')