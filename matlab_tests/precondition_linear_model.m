function precond_img= precondition_linear_model(img,optparas,kernparas,debug)

%Input:
%       img: phase contrast microscopy image
%%%%%%%
%        kernparas: kernel parameter structure {'R', 'W', 'radius'} 
%       'kernparas.R': Outer radius of phase ring; 'kernparas.W': width of
%        phase ring; R and W are provided by microscope manufacturers
%       'kernpara.radius: radius of kernel
%%%%%%%
%        optparas: optimization parameter structure {'w_smooth_spatio','w_sparsity','epsilon','gamma','m_scale','maxiter','tol'}
%        optparas.w_smooth_spatio: weight of the spatial smoothness term
% 	     optparas.w_sparsity: weight of the sparsity term
% 	     optparas.epsilon: used in smooth term: (epsilon+exp)/(epsilon+1)
% 	     optparas.gamma: used in re-weighting. 1/(f+gamma): [1/(maxf+gamma), 1/gamma]
% 	     optparas.m_scale: %downsize image
% 	     optparas.maxiter: the max iteration in optimization 
%        optparas.tol: tolerance in the optimization
%%%%%%%%%
%        debug: debug model          
%Output:
%       precond_img: preconditioning result of phase contrast image 

%Reference:
% [1] Z. Yin, K. Li, T. Kanade and M. Chen, "Understanding the Optics to
% Aid Microscopy Image Segmentation," the 13th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2010

% [2] Zhaozheng Yin, Takeo Kanade, Mei Chen: Understanding the phase
% contrast optics to restore artifact-free microscopy images for segmentation. Medical Image Analysis 16(5): 1047-1062 (2012)

% [3] http://www.celltracking.ri.cmu.edu/

%Composed by Zhaozheng Yin, and modified by Hang Su on 09/15/2012, 
% Robotics Institute, Carnegie Mellon University  
%If you have any suggestions, questions, and bug reports etc please feel free
%to contact Hang Su at (suhangss@gmail.com)

% Copyright (C) Zhaozheng Yin
% All rights reserved.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Example %%%%%%%%%%%%%%%%%%%%%%%%%%%
%Examples:
% %Read a phase contrast image:
%  phc_img=imread('B23P17 LatB 1nM-1_chan00_220.tif');
% %Convert to double precision 
%  img=im2double(phc_img);

% % Optimization Parameter Setup 
%   optparas=struct('w_smooth_spatio',1,'w_sparsity',0.5,'epsilon',100,'gamma',3,'m_scale',1,'maxiter',100,'tol',1);
% % Kernel Parameter Setup
%   kernparas=struct('R',4000,'W',800,'radius',5);
%   debug=0;
% % precond_img=PhaseContrastSegParam(img,optparas,kernparas,debug);
   
% % Convert to Binary Image 
%   segResult=im2bw(precond_img,graythresh(precond_img));
% %display:
% subplot(3, 1, 1), imshow(img, []), title('Original Image');
% subplot(3, 1, 2), imshow(precond_img,[]), title('Preconditioning Result');
% subplot(3, 1, 3), imshow(segResult), title('');


%% Default parameters
default_kernparas=struct('R',4000,'W',800,'radius',5);
default_optparas= struct('w_smooth_spatio',1,'w_sparsity',0.5,'epsilon',100,'gamma',3,'m_scale',1,'maxiter',100,'tol',1);

%% Check Input
if(~exist('kernparas','var')),
    kernparas=default_kernparas;
end
if(~exist('optparas','var')),
    optparas=default_optparas;
end
if(~exist('debug','var')),
    debug=0;
end

%% 
	%-------------------------------------------------------------------------------------
	% initialize
	%----------------------------------------------------------------------
	% Optimization parameter setting
	w_smooth_spatio = optparas.w_smooth_spatio; % weight of the spatial smoothness term
	w_sparsity = optparas.w_sparsity;           % weight of the sparsity term
	epsilon = optparas.epsilon;                 % used in smooth term: (epsilon+exp)/(epsilon+1)
	gamma = optparas.gamma;                     % used in re-weighting. 1/(f+gamma): [1/(maxf+gamma), 1/gamma]
	m_scale = optparas.m_scale;                 % downsize image
	maxiter = optparas.maxiter; tol = optparas.tol;                       % the max iteration and tolerance in the optimization

    % Kernel parameter setting 
    R=kernparas.R;
    W=kernparas.W;
    radius=kernparas.radius;

	%crop/downsize an image or not
    if m_scale>1
        [nrows,ncols]=size(img);
        scale_img=img(1:m_scale:nrows,1:m_scale:ncols);
        img=scale_img;
    end
    
    %% Remove constant items in phase contrast image by a second-order polynomial surface
	im =BackgroundRemoval(img,debug);
    if debug
        figure(2); subplot(2,2,1); 
        imshow(img,[]);
        title('Background Removal'); drawnow; 
    end
    
    %% Get the microscope imaging model
    fprintf('get the microscope imaging model\n');
 %%---------------------------  Phase contrast kernel construction  ---------------------------
	[nrows, ncols] = size(im);
	N = nrows*ncols;
    H = getPhaseConstKernel(nrows, ncols, R, W, radius); 
	HH = H'*H;
   
    fprintf('get the spatial smooth term\n');
 %%---------------------------Get the spatial smooth term  ---------------------------
	inds = reshape(1:N, nrows, ncols); %inds = (xx-1)*nrows+yy;
	HorVerLinks = [[Mat2Vec(inds(:,1:ncols-1)), Mat2Vec(inds(:,2:ncols))];...
	    [Mat2Vec(inds(1:nrows-1,:)), Mat2Vec(inds(2:nrows,:))]];
	DiagLinks = [[Mat2Vec(inds(1:nrows-1,1:ncols-1)), Mat2Vec(inds(2:nrows,2:ncols))];...
	    [Mat2Vec(inds(1:nrows-1,2:ncols)), Mat2Vec(inds(2:nrows,1:ncols-1))]]; 
    HorVerlinkpot = (im(HorVerLinks(:,1))-im(HorVerLinks(:,2))).^2; %grayscale image    
    HorVerlinkpot = (epsilon+exp(-HorVerlinkpot/mean(HorVerlinkpot)))/(epsilon+1); 
    Diaglinkpot = (im(DiagLinks(:,1))-im(DiagLinks(:,2))).^2; %grayscale image    
    Diaglinkpot = 0.707*(epsilon+exp(-Diaglinkpot/mean(Diaglinkpot)))/(epsilon+1); 
    W = sparse([HorVerLinks(:,1); HorVerLinks(:,2); DiagLinks(:,1); DiagLinks(:,2)],...
        [HorVerLinks(:,2); HorVerLinks(:,1); DiagLinks(:,2); DiagLinks(:,1)],...
        [HorVerlinkpot; HorVerlinkpot; Diaglinkpot; Diaglinkpot], N, N);    
    L = spdiags(sum(W,2),0,N,N)-W;
    
 %%-------------------------  For sparse regulation term ------------------
 	%kernel for edge detection (mag)
    fprintf('get the sparse ites\n');
	sigma = 2.5; GaussHwd = 8;
	x = -GaussHwd:GaussHwd; 
	GAUSS = exp(-0.5*x.^2/sigma^2);
	GAUSS = GAUSS/sum(GAUSS);
	dGAUSS = -x.*GAUSS/sigma^2;
	kernelx = GAUSS'*dGAUSS;
	kernely = kernelx';
	%disk filter for saliency detection
	diskfilter = fspecial('disk', 10);
    
    [xx yy] = meshgrid(1:ncols, 1:nrows);
	xx = xx(:); yy = yy(:);
	%for low pass filtering
	lowFreqRadius = 0.3;
	lowFreq = sqrt((xx-ncols/2).^2+(yy-nrows/2).^2)<lowFreqRadius;
    %for high pass filtering
    myFFT = fft2(im);
    myPhase = angle(myFFT);
    myAmplitude = abs(myFFT);
    myHighFreq = fftshift(myAmplitude);    
    myHighFreq(lowFreq)=0;
    saliencyMap = imfilter(abs(ifft2(ifftshift(myHighFreq).*exp(1i*myPhase))), diskfilter,'same');    
    W0 = (epsilon+exp(-saliencyMap(:)/mean(saliencyMap(:))))/(epsilon+1);
    
%%------------------------- Get the prior -------------------------
fprintf('get the prior based on phase contrast distribution\n')
   nImBin = 31; nMagBin = 31; nfBin = 31; 
   load(sprintf('ImMagFRange%.1f%d%d%d%d.mat',sigma,GaussHwd,nImBin,nMagBin,nfBin),...
	    'maxim','minim','maxmag','minmag','maxf','minf');
   load(sprintf('PriorAndCfd%.1f%d%d%d%d.mat',sigma,GaussHwd,nImBin,nMagBin,nfBin),'prior'); % load prior of phase contrast distribution 
    dx = imfilter(im,kernelx,'same');  % x direction
    dy = imfilter(im,kernely,'same');  % y direction
    mag = sqrt(dx.^2+dy.^2);    

    %get bin index
    mag(mag>maxmag) = maxmag; mag(mag<minmag) = minmag;    
    im(im>maxim) = maxim; im(im<minim) = minim;
    iIm = round(nImBin*(im(:)-minim)/(maxim-minim))+1;
    iMag = round(nMagBin*(mag(:)-minmag)/(maxmag-minmag))+1; 
    %look up the prior
    prior_f = reshape(prior((iMag-1)*(nImBin+1)+iIm),[nrows, ncols]);    
    prior_f(prior_f<=0) = 0.00001; %avoid nonpositive initial point  

	clear xx yy inds sigma GaussHwd GAUSS dGAUSS;
    if debug
        figure(2); subplot(2,2,2); 
        imshow(reshape(prior_f,[nrows,ncols]),[]);
        title('prior f'); drawnow;
    end


%% solve the optimization problem
%%------------------------  Deconvolution Items	-------------------------
fprintf('Optimization process\n');
    A = HH + w_smooth_spatio*L;
	btmp = -H'*im(:);
    Ap=(abs(A)+A)/2;   % positive elements of A
    An = (abs(A)-A)/2; % negative elements of A   

	f = prior_f(:); f(f==0) = 0.000001;
	W = W0;
    err = zeros(maxiter,1);
    
%%------------------------  Optimization process -------------------------
	for iter = 1:maxiter        
		b = btmp + w_sparsity*W; 
		tmp = Ap*f;       
		newf = 0.5*f.*(-b+sqrt(b.^2+4*tmp.*(An*f)))./(tmp+eps);    
		W = W0./(newf+gamma);
		err(iter) = sum(abs(f-newf));
		if err(iter) < epsilon
			break;
        end
		f = newf;       
		if debug
			figure(2); 
            subplot(2,2,3);   
			imshow(reshape(f,[nrows,ncols]),[]);
			title(['iter=' num2str(iter) '; err=' num2str(err(iter),'%010.2f')]);
			drawnow;
        end
    end
    
    precond_img=reshape(f,nrows,ncols);
    
    if debug
        figure(2); 
        subplot(2,2,3);   
		imshow(reshape(normalize(f),[nrows,ncols]),[]);
		title('Precondition Result');
		drawnow;
    end
fprintf('Completed!\n');
clear HH H f
end




