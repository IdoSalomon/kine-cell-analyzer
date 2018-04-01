function imgproc=precondition_sparse_respresent(img,optparas,kernparas,debug)

%Input:
%       img: phase contrast microscopy image
%%%%%%%
%        kernparas: kernel parameter structure {'R', 'W', 'radius','dicsize'} 
%        kernparas.R: Outer radius of phase ring; 
%        kernparas.W: width of phase ring; 
%        kernparas.zetap: amplitude attenuation factors caused by phase ring. 
%        R, W and zetap are provided by microscope manufacturers
%        
%        kernpara.radius: radius of kernel
%        kernpara.dicsize': size of dictionary 
%%%%%%%
%        optparas: optimization parameter structure {'w_smooth_spatio','w_sparsity','epsilon','gamma','m_scale','maxiter','tol'}
%        optparas.w_smooth_spatio: weight of the spatial smoothness term
% 	     optparas.w_sparsity: weight of the sparsity term
% 	     optparas.epsilon: used in smooth term: (epsilon+exp)/(epsilon+1)
% 	     optparas.gamma: used in re-weighting. 1/(f+gamma): [1/(maxf+gamma), 1/gamma]
% 	     optparas.m_scale: %downsize image
% 	     optparas.maxiter: the max iteration in optimization 
%        optparas.tol: tolerance in the optimization
%        optparas.sel: maximum number of selected basis 
%%%%%%%%%
%        debug: debug model          
%Output:
%       imgproc: preconditioning result of phase contrast image 

%%%%Reference:
% [1] Hang Su, Zhaozheng Yin, Takeo Kanade, Seungil Huh: Phase Contrast Image Restoration via Dictionary Representation of Diffraction Patterns. the 15th 
%     International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2012: 615-622
%
% [2] Zhaozheng Yin, Takeo Kanade, Mei Chen: Understanding the phase
% contrast optics to restore artifact-free microscopy images for segmentation. Medical Image Analysis 16(5): 1047-1062 (2012)

% [3] http://www.celltracking.ri.cmu.edu/

%Composed by Hang Su on 09/25/2012 
% Robotics Institute, Carnegie Mellon University  
%If you have any suggestions, questions, and bug reports etc please feel free
%to contact Hang Su at (suhangss@gmail.com)

% Copyright (C) Hang Su
% All rights reserved.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Example %%%%%%%%%%%%%%%%%%%%%%%%%%%
%Examples:
%Read a phase contrast image:
%  phc_img=imread('demo.tif');
% if isrgb(phc_img)
%    grayimg=rgb2gray(phc_img);
%  else
%    grayimg=phc_img;
% end
% %Convert to double precision 
%  img=im2double(grayimg);
% 
% % Optimization Parameter Setup 
%   optparas=struct('w_smooth_spatio',0.3,'w_sparsity',0.15,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',6,'tol',100);
% % Kernel Parameter Setup
%   kernparas=struct('R',4,'W',0.8,'radius',2,'dicsize',20);
%   debug=0;
%   
%  % Sparse representation of phase contrast image  
%  imgproc=precondition_sparse_respresent(cellimg,optparas,kernparas,debug);
% 
% 
% % Convert to Binary Image 
% segResult=im2bw(precond_img,graythresh(precond_img));
% %display:
% subplot(3, 1, 1), imshow(img, []), title('Original Image');
% subplot(3, 1, 2), imshow(precond_img,[]), title('Preconditioning Result');
% subplot(3, 1, 3), imshow(segResult), title('Binary Phase Constrast Image');
%%----------------------------------------------------------------------%%

%% %% Default parameters 
default_optparas=struct('w_smooth_spatio',0.3,'w_sparsity',0.15,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',6,'tol',100);
default_kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
%% Check Input Arguement 
if(~exist('kernparas','var')),
    kernparas=default_kernparas;
end
if(~exist('optparas','var')),
    optparas=default_optparas;
end
if(~exist('debug','var'))
    debug=0;
end
if nargin<1
    error('No phase contrast image input');
end

%% Initialization 
if optparas.m_scale>1
        [nrows,ncols]=size(img);
        scale_img=img(1:optparas.m_scale:nrows,1:optparas.m_scale:ncols);
        img=scale_img;
end

[nrow,ncol]=size(img);

%% Remove constant items in phase contrast image by by a second-order polynomial surface
fprintf('Background removal\n');
img=BackgroundRemoval(img,debug);

if debug
    
    figure('Name','Background')
    subplot(1,2,1)
    imshow(img)
    title('Original Image')
    subplot(1,2,2)
    imshow(img)
    title('Background Removal')
end

%% Dictionary construction and basis selection
M=kernparas.dicsize;
K=optparas.sel;     
%selbasis=[1,6,3];
%% Sparse Representation
rimg=img;
for k=1:K 
      % Selected basis calculation 
      fprintf('Select the best basis\n');
      selbasis(k)=BasisBestSelection(rimg,kernparas,M,debug);
      fprintf('%sth basis generation\n',num2str(k));
      kernel=getKernel(kernparas,selbasis(k)/M*2*pi,0);
      basis=imgfun(kernel,nrow,ncol);
      
      %Calculate the coefficient of the correspong basis with multiplicative updating method 
      fprintf('Calculate coefficient of the %s th basis\n',num2str(k));
      resimg(:,:,k)=reshape(PhaseContrastSegParam(basis, rimg, optparas,debug),nrow,ncol); 
      
      fprintf('Residual error update\n');
      rimg=rimg-reshape(basis*resimg(1+(k-1)*numel(rimg):k*numel(rimg))',[nrow,ncol]);
      %normalization 
      imgproc(:,:,k)=normalize(resimg(:,:,k));
      if debug
         figure(3)
         subplot(1,2,1)
         imshow(rimg)
         title('Residual Image')
         subplot(1,2,2)
         imshow(imgproc(:,:,k))
         title(['Retoration Result for ' num2str(k) 'th basis']);
      end 
     if (norm(resimg(:,:,k))/norm(resimg(:,:,1))<0.01)
         break
     end
end
ndep=size(imgproc,3);
imgproc(:,:,ndep+1:K)=0;
