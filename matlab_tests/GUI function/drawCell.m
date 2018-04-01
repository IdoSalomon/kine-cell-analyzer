function imgcell=drawCell(intensity,R)

imgcell=100*ones(256,256);

x=1:256;
y=1:256;
[xx,yy]=meshgrid(x,y);
r=sqrt((xx-128).^2+(yy-128).^2);
imgcell(r<R)=intensity;

end