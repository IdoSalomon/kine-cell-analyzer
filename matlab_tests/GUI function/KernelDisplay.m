function KernelDisplay

   
    set(0, 'Units', 'pixels');
    p = get(0, 'ScreenSize'); % screen width: p(3); screen height: p(4)
	
    % Create Dialog Window
	figw = p(3)*0.5;
	figh = p(4)*0.86;
	fh = figure('Visible','on','Name','Cell Tracking Processor', ...
				'Toolbar', 'figure',...
				'MenuBar', 'none',...
				'Resize', 'off',...
				'Position',[(p(3)-figw)/2,(p(4)-figh)/2,figw,figh]);

  
    img = [];
    imgcell=zeros(256,256);
    imgphase=zeros(256,256);
	% display original image
	img.org = axes('Parent', fh, 'Units', 'pixels', 'Position', [figw/15 5/9*figh 1*figw/3 1*figh/3]);
    title('Cell Demostrate ');
    img.kernel = axes('Parent', fh, 'Units', 'pixels', 'Position', [4*figw/9 5/9*figh 1*figw/3 1*figh/3]);
    title('Kernel Demotrate');
    img.phase = axes('Parent', fh, 'Units', 'pixels', 'Position', [figw/15 1/9*figh 1*figw/3 1*figh/3]);
    title('Phase Contrast Image');
    drawnow;
    
    uicontrol('Parent',fh, 'Style', 'text', 'String', 'Retarded Degree', 'HorizontalAlignment', 'center',...
					'Position', [4*figw/9 26/54*figh 1*figw/8 1*figh/30], 'BackgroundColor', [.9,.9,.9]);        
            
   hslider = uicontrol('Parent',fh, ...
   'Callback',@kernel_call_back, ...
   'Max',360, ...
   'Min',0, ...
   'Position',[52*figw/90 26/54*figh 1*figw/6 1*figh/30], ...
   'String','test', ...
   'Style','slider', ...
   'Value',90, ...
   'Tag','Label1');
          
   uicontrol('Parent',fh, 'Style', 'text', 'String', '360', 'HorizontalAlignment', 'center',...
					'Position', [3*figw/4 26/54*figh 1*figw/20 1*figh/30], 'BackgroundColor', [.9,.9,.9],'Tag','Label1');       
   
    uicontrol('Parent',fh, 'Style', 'text', 'String', 'Cell Intensity', 'HorizontalAlignment', 'center',...
					'Position', [1*figw/27 26/54*figh 1*figw/8 1*figh/30], 'BackgroundColor', [.9,.9,.9]);        
            
   hslider = uicontrol('Parent',fh, ...
   'Callback',@cell_call_back, ...
   'Max',256, ...
   'Min',0, ...
   'Position',[1*figw/6 26/54*figh 1*figw/6 1*figh/30], ...
   'String','test', ...
   'Style','slider', ...
   'Value',90);
          
   uicontrol('Parent',fh, 'Style', 'text', 'String', '90', 'HorizontalAlignment', 'center',...
					'Position', [32*figw/90 26/54*figh 1*figw/20 1*figh/30], 'BackgroundColor', [.9,.9,.9],'Tag','intensity');  
    
    % Generate the Cell Demostrate
function cell_call_back(hObject,eventdata)
       value = get(gcbo,'Value');

% Place the value into the text field
      Hndl = findobj(gcbf,'Tag','intensity');
      str = sprintf('%.2f',value);
      set (Hndl,'String',str);
      r=75;  
      imgcell=drawCell(value,r);
      axes(img.org);
      set(img.org, 'hittest', 'off');
      imshow(imgcell);
      image(imgcell);
 end
       
   
 function kernel_call_back(hObject,eventdata)
     kernel_paras=evalin('base','kernparas');

     % Place the value into the text field
     degree = get(gcbo,'Value');
     Hndl = findobj(gcbf,'Tag','Label1');
     str = sprintf('%.2f',degree);
     set (Hndl,'String',str);
     
     radius=kernel_paras.radius;  
     [xx,yy] = meshgrid(-radius:radius,-radius:radius);
     degree=degree/360*2*pi;
     kernel=getKernel(kernel_paras,degree,0);
    
     h=figure('name','kernel demonstrate');
     colormap('jet');
     surf(xx,yy,kernel,'FaceColor','interp','EdgeColor','none','FaceLighting','phong');
     F=getframe(gca);
     close(h)
     
     axes(img.kernel);
     set(img.kernel, 'hittest', 'off');
     image(F.cdata);
     
     
     nrows=256;
     ncols=256;
     basis=imgfun(kernel,nrows,ncols);
     imgphase=reshape(basis*imgcell(:),nrows,ncols);
     imgphase=normalize(imgphase);
     axes(img.phase);
     set(img.phase, 'hittest', 'off');
     imshow(imgphase)
     
     clear basis kernel
      
 end    
end