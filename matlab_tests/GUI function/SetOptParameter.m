function SetOptParameter()

phopt = figure('units','pixels','position', [50 100 300 400],...
   'tag','GUI', 'name','Optimization Pameter Setup',...
   'menubar','none','numbertitle','off');

uicontrol(phopt, 'Style', 'text', 'String', 'Selected Basis', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.85,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfSel = uicontrol(phopt, 'Style', 'edit', ...
					'String', '3', 'Callback', @UpdateParameter, ...
					'Units','Normalize','Position', [0.55,.88,.4,.08]);
uicontrol(phopt, 'Style', 'text', 'String', 'spatial smooth ', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.73,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfSmooth = uicontrol(phopt, 'Style', 'edit', ...
					'String', '1', 'Callback', @UpdateParameter, ...
					'Units','Normalize','Position', [0.55,.76,.4,.08]);
uicontrol('Parent', phopt, 'Style', 'text', 'String', 'sparsity', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.61,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfSparsity = uicontrol('Parent', phopt, 'Style', 'edit', ...
					'String', '0.4', 'Callback', @UpdateParameter, ...
					'Units','Normalize','Position', [0.55,.64,.4,.08]);
uicontrol('Parent',phopt, 'Style', 'text', 'String', 'epsilon ', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.49,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfepsilon = uicontrol('Parent',phopt, 'Style', 'edit', ...
					'String', '3', 'Callback', @UpdateParameter, ...
					'Units','Normalize','Position', [0.55,.52,.4,.08]);
uicontrol('Parent',phopt, 'Style', 'text', 'String', 'gamma', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.37,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfgamma= uicontrol('Parent',phopt, 'Style', 'edit', ...
					'Units','Normalize','String', '3', 'Callback', @UpdateParameter, ...
					'Position', [0.55,.40,.4,.08]);
uicontrol(phopt, 'Style', 'text', 'String', 'm_scale', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.25,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfScale= uicontrol(phopt, 'Style', 'edit', ...
					'String', '1', 'Callback', @UpdateParameter, ...
					'Units','Normalize','Position', [0.55,.28,.4,.08]);
uicontrol(phopt, 'Style', 'text', 'String', 'max iteration', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.13,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfMaxiter = uicontrol(phopt, 'Style', 'edit', ...
					'String', '100', 'Callback', @UpdateParameter, ...
					'Units','Normalize','Position', [0.55,.16,.4,.08]);    
uicontrol(phopt, 'Style', 'text', 'String', 'tolerance', 'HorizontalAlignment', 'center',...
					'Units','Normalize','Position', [0.05,.01,.4,.1], 'BackgroundColor', [.8,.8,.8]);
psfTol = uicontrol(phopt, 'Style', 'edit', ...
					'String', '1', 'Callback', @UpdateParameter, ...
					'Units','Normalize','Position', [0.55,.04,.4,.08]);        
                
optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',1);        
 assignin('base', 'optparas', optparas);              
                
                
                
   function  UpdateParameter(hObject,eventdata)
     optparas.sel= str2num(get(psfSel, 'String'));
     optparas.w_smooth_spatio = str2num(get(psfSmooth, 'String'));
	 optparas.w_sparsity = str2num(get(psfSparsity, 'String'));
	 optparas.epsilon = str2num(get(psfepsilon, 'String'));
     optparas.gamma= str2num(get(psfgamma, 'String'));
     optparas.m_scale = str2num(get(psfScale, 'String'));
     optparas.maxiter = str2num(get(psfMaxiter, 'String'));
     optparas.tol = str2num(get(psfTol, 'String'));
     
     assignin('base', 'optparas', optparas);
   end       
                         
                
end