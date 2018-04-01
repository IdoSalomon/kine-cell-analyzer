%%%%%%% GUI Demo of precondition for phase contrast image
%
%%----------------------------------Parameter----------------------------------------------------------%
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
%        --'linear_model': an earlier algorithm published on MICCAI2010, which restore dark cells in phase contrast images
%        --'sparse _respresent': a recent algorithm published on MICCAI2012, which restore phase contrast images with sprarse representation model
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
% kernparas = struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
% optparas  = struct('w_smooth_spatio',0.3,'w_sparsity',0.15,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',100);
%  
% For Bovine aortic endothelial cell (BAEC)
% kernparas = struct('R',4000,'W',800,'radius',5,'zetap',0.8,'dicsize',20);
% optparas  = struct('w_smooth_spatio',1,'w_sparsity',0.5,'sel',3,'epsilon',100,'gamma',3,'m_scale',1,'maxiter',100,'tol',1);

%% 
clear all
close all
clc
addpath('./GUI function');
addpath('./func');
%% Pannel Layerout 
p = get(0, 'ScreenSize'); % screen width: p(3); screen height: p(4)	
%Create Dialog Window
figw = p(3)*1/2;
figh = 2*p(4)/3;
hfig = figure('Visible','on','Name','Precondition', ...
				'Toolbar', 'figure',...
				'MenuBar', 'none',...
				'Resize', 'On',...
				'Position',[(p(3)-figw)/2,(p(4)-figh)/2,figw,figh]);
hmenu = uimenu('Label','File');
uimenu(hmenu,'label','Open...','callback','SelectImage');
uimenu(hmenu,'label','Exit','callback','closereq','separator','on');

Fig.org = axes('Parent',hfig, 'Units', 'Normalized', 'Position', [.06 .55 .4 .4]);
title('Original Image')

Fig.precond= axes('Parent',hfig, 'Units', 'Normalized', 'Position', [.06 .06 .4 .4]);
title('Precondition Result')

Fig.binimg= axes('Parent',hfig, 'Units', 'Normalized', 'Position', [.52 .06 .4 .4]);
title('Cell Mask')

%% Parameter Setup Pannel
                
phPcn= uipanel('Parent',hfig,'Title','Preconditioning Control Pannel','Units','Normalized','Position',[.52 .52 .45 .45]);
                
Pcn.kern = uicontrol('Parent',phPcn, 'Style','pushbutton', 'Units', 'Normalized','Position',[.05 .55 .9 .12], ...
                    'String','Optimization Paramter','Callback', 'SetOptParameter');
Pcn.para = uicontrol('Parent',phPcn, 'Style','pushbutton', 'Units', 'Normalized','Position',[.05 .40 .9 .12], ...
                    'String','Kernel Parameter','Callback', 'SetKernelParameter');         
Pcn.bin = uicontrol('Parent',phPcn, 'Style','pushbutton', 'Units', 'Normalized','Position',[.05 .70 .9 .12], ...
                   'String','Cell Mask','Callback', 'BinaryImage');
               
Pcn.debug = uicontrol('Parent',phPcn,'Style','checkbox',...
'String','debug',...
'Value',0,'Units','Normalize','Position',[.05 .02 .9 .15]);

uicontrol('Parent',phPcn, 'Style', 'text', 'String', 'Algotithm Selection', 'HorizontalAlignment', 'center',...
					'Units','Normalized','Position', [0.05 0.20 0.3 0.12], 'BackgroundColor', [.8,.8,.8]);

Pcn.mode=uicontrol('Parent',phPcn, 'Units','Normalized','Position', [0.45 0.1 0.5 0.2],...
    'style','popup','fontunits','normalized','fontsize',0.3,'FontName','Times New Roman','string',{'sparse respresent','linear model'},...
    'value',1,'Tag','mode');

Pcn.prec = uicontrol('Parent',phPcn, 'Style','pushbutton', 'Units', 'Normalized','Position',[.05 .85 .9 .12], ...
                    'String','Precondition','Callback', {@precondition_img,Pcn});      
%% Defaualt Parameter
% Optimization Parameter Setup 
 optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',1);
% Kernel Parameter Setup
 kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
 debug=0;
 mode='sparse _respresent';

 assignin('base','optparas',optparas);
 assignin('base','kernparas',kernparas);
 assignin('base','debug',debug);
 assignin('base','mode',mode);
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  