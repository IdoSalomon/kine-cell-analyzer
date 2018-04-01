
clear all
close all
clc

addpath('seq_training');
imgname='./newTifs/';
%Control parameter
ConPara=struct('fstd',121,'fend',221,'fint',10);
%Table parameter 
TblPara=struct('sigma',2.5,'GaussHwd',8,'nImBin',31,'nMagBin',31,'nfBin',31);
%% Parameter Setup 
optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',eps);
% Kernel Parameter Setup
kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);

%% Lookup table construction
[prior,ImParas]=phLookUpTable(imgname,ConPara,TblPara,optparas,kernparas,selbasis);

maxim=ImParas.maxim;
minim=ImParas.minim;
maxmag=ImParas.maxmag;
minmag=ImParas.minmag;
maxf=ImParas.maxf;
minf=ImParas.minf;

% save prior of phase contrast distribution 
save(sprintf('ImMagFRange%.1f%d%d%d%d.mat',TblPara.sigma,TblPara.GaussHwd,TblPara.nImBin,...
    TblPara.nMagBin,TblPara.nfBin), 'maxim','minim','maxmag','minmag','maxf','minf');
save(sprintf('PriorAndCfd%.1f%d%d%d%d.mat',TblPara.sigma,...
    TblPara.GaussHwd,TblPara.nImBin,TblPara.nMagBin,TblPara.nfBin),'prior'); 







