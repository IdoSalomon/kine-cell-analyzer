function [H, kernel] = getPhaseConstKernel(nrows, ncols, R, W, radius)

N = nrows*ncols;

if nargin<5
    radius = 3;
end

%% The corresponding point spread kernel function for the negative phase contrast 
diameter = 2*radius + 1;
[xx,yy] = meshgrid(-radius:radius,-radius:radius);
rr = sqrt(xx.^2 + yy.^2);

kernel1 = pi*R^2*somb(2*R*rr);     
kernel2 = pi*(R-W)^2*somb(2*(R-W)*rr);    
kernel = kernel1 - kernel2;
kernel = -kernel/norm(kernel);  
kernel(radius+1,radius+1) = kernel(radius+1,radius+1) + 1;
kernel = -kernel(:);

    
%% Build the sparse imaging H matrix
nzidx = abs(kernel) > 0.01; %very important to save memory and speed up

inds = reshape(1:N, nrows, ncols);
inds_pad = padarray(inds,[radius radius],'symmetric'); %deal with the boundary

row_inds = repmat(1:N, sum(nzidx), 1);
col_inds = im2col(inds_pad, [diameter,diameter], 'sliding'); %slide col and then row
col_inds = col_inds(repmat(nzidx, [1,N]));
vals = repmat(kernel(nzidx), N, 1);
H = sparse(row_inds(:), col_inds(:), vals, N, N); 

