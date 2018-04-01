function precond_img= PhaseContrastTabel(basis,im, optparas,debug)

    
	%-------------------------------------------------------------------------------------
	% initialize
	%----------------------------------------------------------------------
    if nargin<4
        debug=0;
    end
	% Optimization parameter setting
	w_smooth_spatio = optparas.w_smooth_spatio; % weight of the spatial smoothness term
	w_sparsity = optparas.w_sparsity;           % weight of the sparsity term
	epsilon = optparas.epsilon;                 % used in smooth term: (epsilon+exp)/(epsilon+1)
	gamma = optparas.gamma;                     % used in re-weighting. 1/(f+gamma): [1/(maxf+gamma), 1/gamma]
	m_scale = optparas.m_scale;                 % downsize image
	maxiter =optparas.maxiter; tol = optparas.tol;                       % the max iteration and tolerance in the optimization
	maxval = 1;                                 % 48 C2C12; resolution bit per pixel


	  %% Get the microscope imaging model
    
 %%---------------------------  Phase contrast kernel construction  ---------------------------
	[nrows, ncols] = size(im);
	N = nrows*ncols;
    H = basis; 
	HH = H'*H;
   
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
    
 

%% solve the optimization problem
%%------------------------  Deconvolution Items	-------------------------
    A = HH + w_smooth_spatio*L;
	btmp = -H'*im(:);
    Ap=(abs(A)+A)/2;   % positive elements of A
    An = (abs(A)-A)/2; % negative elements of A   
     W0=ones(N,1);
     W=W0;
     f=ones(N,1)*0.01;
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
		title('Coeffient of the basis');
		drawnow;
    end

clear HH H f
end

