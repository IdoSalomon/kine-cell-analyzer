function selBasis=BasisBestSelection(img,kernelparas,M,debug)

if nargin < 4, debug = 0; end

%% Inner production calculation for phase contrast image and 
[nrow,ncol]=size(img);

for m=1:M
   if debug
       fprintf('Calculate inner production with %sth basis\n', num2str(m));
   end
   ang=2*pi/M*m;
   kernel=getKernel(kernelparas,ang,0);
   basis=imgfun(kernel,nrow,ncol);

   resfeature=basis*img(:); %inner product of kernel and phase contrast image 
   resfeature=reshape(resfeature,nrow,ncol);

   resfeature(resfeature<0)=0;
   inner_norm(m)=norm(resfeature(:)); %norm of inner production 
 
 %   discrim(m)=sum(bwimg(:));
    if debug
        figure(10)
        imshow(normalize(resfeature))
        title(['Inner production of basis with phase retardation ' num2str(m) '\times 2\pi/' num2str(M) ' and original image'] )
    end
end
[val,pos]=max(inner_norm);
selBasis=pos; 