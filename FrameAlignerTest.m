im = imread('cameraman.tif');
im = single(im);

realign_params.type = 3;

n_x = size(im,2);
n_frames = 2;

image_scan_params.line_duration = 1;
image_scan_params.interline_duration = 3;
image_scan_params.n_x = n_x;
image_scan_params.n_y = n_x;
a = FrameAlignerMex(realign_params, image_scan_params, n_frames);

ref = rand(n_x,n_x,'single');
im1 = circshift(im,5);

FrameAlignerMex(a,'SetReference',0,im);
FrameAlignerMex(a,'AddFrame',1,im1);

im2 = FrameAlignerMex(a,'GetRealignedFrame',1);

imagesc(imfuse(im,im2))
daspect([1 1 1])