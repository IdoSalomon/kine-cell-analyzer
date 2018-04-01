%%%%%%% Demo of sparse representation for phase contrast image
%     kernparas: kernel parameter structure {'R', 'W', 'radius','zetap','dicsize'} 
%        -- R: Outer radius of phase ring; 
%        -- W: width of phase ring; 
%        -- zetap: amplitude attenuation factors caused by phase ring.
%        -- R, W and zetap are provided by microscope manufacturers
%        -- radius: radius of kernel
%        -- dicsize: size of dictionary 
%%--------------------------------------------------------------------------------------------%
%      optparas: optimization parameter structure {'w_smooth_spatio','w_sparsity','epsilon','gamma','m_scale','maxiter','tol'}
%        --w_smooth_spatio: weight of the spatial smoothness term
% 	     --w_sparsity: weight of the sparsity term
% 	     --epsilon: used in smooth term: (epsilon+exp)/(epsilon+1)
% 	     --gamma: used in re-weighting. 1/(f+gamma): [1/(maxf+gamma), 1/gamma]
% 	     --m_scale: %downsize image
% 	     --maxiter: the max iteration in optimization 
%        --tol: tolerance in the optimization
%        --sel: maximum number of selected basis 
%%--------------------------------------------------------------------------------------------%
%     mode: algorithm select  
%        --'linear_model': an earlier algorithm published on Medical Image Analysis (2012), 
%                          which restore dark cells in phase contrast images
%        --'sparse _respresent': a recent algorithm published on MICCAI2012, which restore phase contrast images
%                                with sprarse representation model
%%--------------------------------------------------------------------------------------------%
%        debug: debug model     
%
%%------------------------------------------Output--------------------------------------------------%
%
%       precd_img: preconditioning result of phase contrast image 

%%%%Reference:
% [1] Hang Su, Zhaozheng Yin, Takeo Kanade, Seungil Huh: Phase Contrast Image Restoration via Dictionary Representation of Diffraction Patterns.
% the 15th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2012: 615-622

% [2] Zhaozheng Yin, Takeo Kanade, Mei Chen: Understanding the phase
% contrast optics to restore artifact-free microscopy images for segmentation. Medical Image Analysis 16(5): 1047-1062 (2012)

% [3] http://www.celltracking.ri.cmu.edu/

% Composed by Zhaozheng Yin, Hang Su and modified by Seung-il Huh 
% Robotics Institute, Carnegie Mellon University  
% ------------------------------------------------------------------------------------------------------------
% If you have any suggestions, questions, and bug reports etc, please feel free
% to contact Hang Su at (suhangss@gmail.com)

% Copyright (C) Zhaozheng Yin, Hang Su
% All rights reserved.

%%-------------------------------------Recommended Parameter--------------------------------------------------------------%%
% For the human-stem-cells (HSC), provided by Pitt University
% kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
% optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',20,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',eps);
%  
% For Bovine aortic endothelial cell (BAEC)
% kernparas = struct('R',4000,'W',800,'radius',5,'zetap',0.8,'dicsize',20);
% optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',10);
%% -------------------------------------------------------------------------------------------------------------------------%%

clear all
close all
clc
%% Read a phase contrast image:
addpath('./func')
phc_img=imread('small.png');
if ndims(phc_img) == 3
   grayimg=rgb2gray(phc_img);
else
   grayimg=phc_img;
end
%Convert to double precision 
img=im2double(grayimg);

%% Parameter Setup 
optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',eps);
% Kernel Parameter Setup
kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
% Algorithm mode
mode='sparse_respresent';
debug=0;
  
%% Precondition of phase contrast image  
precd_img=precondition_phase_contrast(img,optparas,kernparas,mode,debug);

%% Convert to Binary Image 

if strcmp(mode,'sparse_respresent')
    procimg=max(precd_img,[],3);
elseif strcmp(mode,'linear_model')
    procimg=normalize(precd_img);
end
segResult=im2bw(procimg,graythresh(procimg));

%% Demonstrate Result
if (optparas.sel>3)
    demonResult=precd_img(:,:,1:3);
elseif (optparas.sel<3)
    [nrow,ncol,ndep]=size(precd_img,3);
    demonResult= precd_img;
    demonRsult(:,:,ndep+1:3) = zeros(nrow,ncol);
else
    demonResult= precd_img;
end
%display:
imwrite(segResult,'res.tif')
imwrite(demonResult,'res2.tif')
subplot(1,3, 1), imshow(img, []), title('Original Image');
subplot(1,3, 2), imshow(demonResult,[]), title('Preconditioning of Phase Contrast Image');
subplot(1,3, 3), imshow(segResult), title('Cell Mask');


















