%------------------2.1--------------------%
% a. Input Image
image = imread('D:\NTU Class\CE4003\mrt-train.jpg');

%Check for RGB or Gray_scale
whos image

%Convert to grayscale
P = rgb2gray(image);
whos P

% b. Show Image
imshow(P);

% c. Check minimum and maximum intensity
min(P(:)), max(P(:))

% d. Contrast Stretch to 0-255 from 13-204
subOperation=imsubtract(P,13);
P2 = immultiply(subOperation, 255/191);

%Check for Min, Max -> 0,255
min(P2(:)), max(P2(:))

% e. Show Stretched Image
imshow(P2,[]);


%------------------2.2--------------------%
% a. Show Image Intensity Historygram of Gray Image
imhist(P,10)
imhist(P, 256)

% b. Do Histogram Equalization for Gray Image
P3=histeq(P,255);
imhist(P3, 10)
imhist(P3, 256)

% c. Re-running histogram equalization on P3
P4=histeq(P3,255);
imhist(P4, 10)
imhist(P4, 256)


%------------------2.3---------------------%
% a. Generate filters
% (i) Y and X-dimensions are 5 and σ = 1.0
sig1=1.0;
% (ii) Y and X-dimensions are 5 and σ = 2.0
sig2=2.0;

dimension=5;
range=-floor(dimension/2):floor(dimension/2);
[X, Y]=meshgrid(range,range);

filter1=h(X,Y,sig1);
filter2=h(X,Y,sig2);

%Normalizing filters
filter1=filter1/sum(filter1(:));
filter2=filter2/sum(filter2(:));

%Display filters
mesh(filter1)
mesh(filter2)


% b. View ntu-gn.jpg
image_gaussian_noise=imread('D:\NTU Class\CE4003\ntu-gn.jpg');
imshow(image_gaussian_noise);
imhist(image_gaussian_noise);

% c. Linear filtering
P5_1=uint8(conv2(image_gaussian_noise, filter1));
imshow(P5_1);
P5_2=uint8(conv2(image_gaussian_noise, filter2));
imshow(P5_2);
 
% d. View image with pseckle noise
image_speckle_noise=imread('D:\NTU Class\CE4003\ntu-sp.jpg');
imshow(image_speckle_noise);

%e. Repeat step con image_speckle_noise
P6_1=uint8(conv2(image_speckle_noise, filter1));
imshow(P6_1);
P6_2=uint8(conv2(image_speckle_noise, filter2));
imshow(P6_2);


%------------------2.4---------------------%
% Median filtering of image with gaussian noise
filter_3_3= [3 3];
P7_1=uint8(medfilt2(image_gaussian_noise, filter_3_3));
imshow(P7_1);
filter_5_5= [5 5];
P7_2=uint8(medfilt2(image_gaussian_noise, filter_5_5));
imshow(P7_2);

% Median filtering of image with speckle noise
P8_1=uint8(medfilt2(image_speckle_noise, filter_3_3));
imshow(P8_1);
P8_2=uint8(medfilt2(image_speckle_noise, filter_5_5));
imshow(P8_2);


%------------------2.5---------------------%
% a. Display pck-int.jpg
Pck_Int=imread('D:\NTU Class\CE4003\pck-int.jpg');
imshow(Pck_Int);

% b. Obtain the Fourier transform F of the image using fft2, and subsequently compute the power spectrum S
F=fft2(Pck_Int);
% S=abs(F).^2 / length(Pck_Int);
imagesc(fftshift(real(F.^0.1)));
colormap('default');

% c. Redisplay the power spectrum without fftshift
imagesc(real(F.^0.1));
colormap('default');
%[x,y]=ginput(2);

% d. Set to zero the 5x5 neighbourhood elements at locations corresponding to the above peaks in the Fourier transform F
F_peaks_zero=F;
dim=floor(5/2);

F_peaks_zero(240-dim:240+dim,9-dim:9+dim)=0;
F_peaks_zero(17-dim:17+dim,248-dim:248+dim)=0;
imagesc(real(F_peaks_zero.^0.1));
colormap('default');
 
% e. Compute the inverse Fourier transform using ifft2 and display the resultant image
F_peaks_zero_inversefft=uint8(ifft2(F_peaks_zero));
imshow(F_peaks_zero_inversefft);

% Improvement
F_peaks_zero2=F;
F_peaks_zero2(240-dim:240+dim,9-dim:9+dim)=0;
F_peaks_zero2(17-dim:17+dim,248-dim:248+dim)=0;

%Extending the zero to edge
F_peaks_zero2(:,9)=0;
F_peaks_zero2(241,:)=0;
F_peaks_zero2(:,249)=0;
F_peaks_zero2(17,:)=0;

imagesc(real(F_peaks_zero2.^0.1))
F_peaks_zero2_inversefft = uint8(ifft2(F_peaks_zero2));
imshow(real(F_peaks_zero2_inversefft))

F_peaks_zero2_minVal=double(min(real(F_peaks_zero2_inversefft(:))));
F_peaks_zero2_maxVal=double(max(real(F_peaks_zero2_inversefft(:))));
subt=imsubtract(real(F_peaks_zero2), F_peaks_zero2_minVal);
Final_F_peaks_zero2=immultiply(subt, 255/(F_peaks_zero2_maxVal-F_peaks_zero2_minVal));


% f. Primate caged
primate=imread('D:\NTU Class\CE4003\primate-caged.jpg');
whos primate
primate=rgb2gray(primate);
imshow(primate);

primate_fourier=fft2(primate);
imagesc(fftshift(real(primate_fourier.^0.1)));
colormap('default');

primate_fourier_peak_zero=primate_fourier;
imagesc(real(primate_fourier_peak_zero.^0.1));
% [x,y]=ginput(4);
% (y,x)=(21,247), (11, 252), (237, 9), (246, 6)
y1=21; x1=247;
y2=11; x2=252;
y3=237; x3=9;
y4=246; x4=6;

primate_fourier_peak_zero(x1-3:x1+3,y1-3:y1+3) = 0;
primate_fourier_peak_zero(x2-3:x2+3,y2-3:y2+3) = 0;
primate_fourier_peak_zero(x3-3:x3+3,y3-3:y3+3) = 0;
primate_fourier_peak_zero(x4-3:x4+3,y4-3:y4+3) = 0;
imagesc(real(primate_fourier_peak_zero.^0.1));

primate_inverse_fourier_peak_zero=uint8(ifft2(primate_fourier_peak_zero));
imshow(real(primate_inverse_fourier_peak_zero));


%------------------2.6---------------------%
% a. Display book.jpg
P=imread('D:\NTU Class\CE4003\book.jpg');
whos P
imshow(P);

% b. Find out the location of 4 corners of the book and store in x and y

[X, Y] = ginput(4);
ximvec = [0 210 210 0];
yimvec = [0 0 297 297];

% c. Projective transformations on image
v = [ximvec(1); yimvec(1); ximvec(2); yimvec(2); ximvec(3); yimvec(3); ximvec(4); yimvec(4)];
A = [
[X(1), Y(1), 1, 0, 0, 0, -ximvec(1) * X(1), -ximvec(1) * Y(1)];
[0, 0, 0, X(1), Y(1), 1, -yimvec(1) * X(1), -yimvec(1) * Y(1)];
[X(2), Y(2), 1, 0, 0, 0, -ximvec(2) * X(2), -ximvec(2) * Y(2)];
[0, 0, 0, X(2), Y(2), 1, -yimvec(2) * X(2), -yimvec(2) * Y(2)];
[X(3), Y(3), 1, 0, 0, 0, -ximvec(3) * X(3), -ximvec(3) * Y(3)];
[0, 0, 0, X(3), Y(3), 1, -yimvec(3) * X(3), -yimvec(3) * Y(3)];
[X(4), Y(4), 1, 0, 0, 0, -ximvec(4) * X(4), -ximvec(4) * Y(4)];
[0, 0, 0, X(4), Y(4), 1, -yimvec(4) * X(4), -yimvec(4) * Y(4)];
];

u = A \ v;
disp(u);
U = reshape([u;1], 3, 3)';
disp(U);
w = U * [X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));
disp(w);

% d. Warping the image
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

% e. Display image
imshow(P2);

% f. Identify the pink area
%Validating that it ias a RGB image
whos(p2)
%Extracting the image layers
red=P2(:,:,1);
green=P2(:,:,2);
blue=P2(:,:,3);

% thresholding the output pixel range
output=red>200 & red<225 & green<150 & green>0 & blue<160 & blue>0;
imshow(output);

% enhancing the image
output2=imfill(output,'holes');
output3=bwmorph(output2,'dilate',3);
output3=imfill(output3,'holes');
imshow(output3);

imBoth=imoverlay(P2,out3,'black');
imshow(imBoth);

% 2.2 a Normalizing Filter function
function h = h(x,y,sigma)
    h=(1/(2*pi*power(sigma,2)))*(exp(-(power(x,2)+power(y,2))/(2*power(sigma,2))));
end