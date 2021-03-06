function [phTbl,ImParas]=phLookUpTable(img_name,ConPara,TblPara,optparas,kernparas,selbasis)

fstd=ConPara.fstd;
fend=ConPara.fend;
fint=ConPara.fint;

sigma =TblPara.sigma; 
GaussHwd=TblPara.GaussHwd;
nImBin=TblPara.nImBin+1;
nMagBin=TblPara.nMagBin+1;
nfBin=TblPara.nfBin+1;

ImParas=struct('maxim',0,'minim',0,'maxmag',0,'minmag',0,'maxf',0,'minf',0);
PhTbl=zeros(nImBin,nMagBin);
maxim=0; minim=0;
maxmag=0; minmag=0;
maxf=0; minf=0;


imname=sprintf('%s%04d.tif',img_name,fstd);
img=imread(imname);
nrow=size(img,1);
ncol=size(img,2);

M=optparas.dicsize;
selBasis=BasisBestSelection(img,kernparas,M,debug);

kernel=getKernel(kernparas,selbasis,0);
basis=imgfun(kernel,nrow,ncol);
j=1;

for i=fstd:fint:fend
    imname=sprintf('%s%04d.tif',img_name,i);
    img=imread(imname);
    
    if isrgb(img)
        img=im2double(rgb2gray(img));
    else
        img=im2double(img);
    end
    % remove background
    im=BackgroundRemoval(img,0);
    [nrow,ncol]=size(im);
    
    %Gradient Calculation 
    x = -GaussHwd:GaussHwd; 
	GAUSS = exp(-0.5*x.^2/sigma^2);
	GAUSS = GAUSS/sum(GAUSS);
	dGAUSS = -x.*GAUSS/sigma^2;
	kernelx = GAUSS'*dGAUSS;
	kernely = kernelx';
    dx = imfilter(im,kernelx,'same');  % x direction
    dy = imfilter(im,kernely,'same');  % y direction
    mag = sqrt(dx.^2+dy.^2);    
    
    %Range Calculation
    cminim=min(im(:)); cmaxim=max(im(:));
    cminmag=min(mag(:)); cmaxmag=max(mag(:));
    if cminim<minim
        minim=cminim;
    end
    if cmaxim>maxim
        maxim=cmaxim;
    end
    if cminmag<minmag;
        minmag=cminmag;
    end
    if cmaxmag>maxmag
        maxmag=cmaxmag;
    end
    %get bin index
    mag(mag>maxmag) = maxmag; mag(mag<minmag) = minmag;    
    im(im>maxim) = maxim; im(im<minim) = minim;
    iIm = round(nImBin*(im(:)-minim)/(maxim-minim))+1;
    iMag = round(nMagBin*(mag(:)-minmag)/(maxmag-minmag))+1; 
  
    f= PhaseContrastTabel(basis,im, optparas);
    cminf=min(f(:)); cmaxf=max(f(:));
    if cmaxf>maxf
        maxf=cmaxf;
    end
    if cminf<minf
        minf=cminf;
    end
    
    ifmag = round(nfBin*(f(:)-minf)/(maxf-minf))+1;
    
    for m=1:nImBin
        for n=1:nMagBin
            corID=intersect(find(iIm==m),find(iMag==n));
            if ~isempty(corID)
             Tbl(m,n,j)=max(f(corID));
            end
        end
    end
    j=j+1;
    
end

phTbl=max(Tbl,[],3);

ImParas=struct('maxim',maxim,'minim',minim,'maxmag',maxmag,'minmag',minmag,'maxf',maxf,'minf',minf);






