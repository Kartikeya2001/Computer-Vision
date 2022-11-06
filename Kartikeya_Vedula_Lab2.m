%------------------3.1--------------------%
% a. Input Image
Pc=imread('D:\NTU Class\CE4003 Computer Vision\Lab2 Edges, Hough Lines, and Disparity\macritchie.jpg');

%Check for RGB or Gray_scale
whos Pc

%Convert to grayscale
P_original=Pc;
Pc = rgb2gray(Pc);

% Display the grayscale image
imshow(Pc,[]);

% b. Create 3x3 horizontal and vertical sobel mask
horiSobel = [-1 -2 -1;
              0 0 0;
              1 2 1];
vertiSobel = [-1 0 1;
              -2 0 2;
              -1 0 1];

% Convolution image with sobel mask created above
image = conv2(double(Pc),double(horiSobel));
imagehori =  abs(image)/4;

imageverti= conv2(double(Pc),double(vertiSobel));
imageverti = abs(imageverti)/4;

% Display horizontal filter
figure;
imshow(imagehori, []);
title('Horizontal Sobel Filter');
% Display vertical filter
figure;
imshow(imageverti, []);
title('Vertical Sobel Filter');

% diagonal edges detection
diagSobel = [ 0 1 2;
             -1 0 1;
             -2 -1 0];
imagediagonal= conv2(double(Pc),double(diagSobel));
imagediagonal = abs(imagediagonal)/4;

% Display horizontal filter
figure;
imshow(imagediagonal, []);
title('Detect Diagonal edges');

% c. Squaring horizontal and vertical edge images and adding them
imgcombedge = sqrt(imagehori.^2+imageverti.^2);
figure;
imshow(imgcombedge, []);
title('Combined Sobel Filter');

% d. Trying different threshold values and displaing the resultant binary image
% threshold(t) = 100
threshold100 = imgcombedge>100;
figure;
imshow(threshold100, []);
title('t=100');

% threshold(t) = 75
threshold75 = imgcombedge>75;
figure;
imshow(threshold75, []);
title('t=75');

% threshold(t) = 50
threshold50 = imgcombedge>50;
figure;
imshow(threshold50, []);
title('t=50');

% e. Recomputing the edge image using the more advanced Canny edge
% detection algorithm
% i. Trying different sigma from 1.0 to 5.0
canny1 = edge(Pc, 'canny', [0.04 0.1], 1.0);
canny2_5 = edge(Pc, 'canny', [0.04 0.1], 2.5);
canny5 = edge(Pc, 'canny', [0.04 0.1], 5.0);
% showing results
figure;
imshow(canny1, []);
title('sigma = 1');
figure;
imshow(canny2_5, []);
title('sigma = 2.5');
figure;
imshow(canny5, []);
title('sigma = 5');

% ii. Trying raising and lowering the value of tl, keeping sigma at 3
canny3_1 = edge(Pc, 'canny', [0.01 0.1], 3.0);
canny3_2 = edge(Pc, 'canny', [0.05 0.1], 3.0);
canny3_3 = edge(Pc, 'canny', [0.09 0.1], 3.0);
% showing results
figure;
imshow(canny3_1, []);
title('tl = 0.01');
figure;
imshow(canny3_2, []);
title('tl = 0.05');
figure;
imshow(canny3_3, []);
title('tl = 0.09');


%------------------3.2--------------------%
% a. Reuse the edge image computed via the Canny algorithm with sigma=1.0
cannyRe = edge(Pc, 'canny', [0.04 0.1], 1.0);
figure;
imshow(cannyRe, []);
title('sigma = 1');
theta=0:179;

% b. Radon transform on binary image is equivalent to the Hough transform
[H, xp] = radon(cannyRe);
figure; imagesc(uint8(H));title('Image in Hough Space');
xlabel('Theta');
ylabel('Rho');
colormap("hot");
colorbar;

% c. Getting location of max pixel intensity in the Hough image
[max_value,idx]=max(H(:));
[x_max,y_max]=ind2sub(size(H),idx);
%Find line theta and radius with strongest edge support
theta=theta(y_max);
radius=xp(x_max);
fprintf('radius: %d\n', radius);
fprintf('theta: %d\n', theta);

% d. Convert theta, radius to Ax+by=C and find C
[A, B] = pol2cart(theta*pi/180, radius);
B = -B;
% Original intercept=radius*csc(theta*pi/180)=radius/sin(theta*pi/180), and
% since y-axis has changed direction, C1=-radius/sin(theta*pi/180)
C1=-(radius/sin(theta*pi/180));
% C is the new intercept, and from the graph it can be seen that C=L/2+ the
% y original value of the intersection of line and new y-axis
[L,W]=size(cannyRe);
x0=-W/2;
y0=(-A/B)*x0+C1;
C = y0+(L/2);
fprintf('A = %d\n', A);
fprintf('B = %d\n', B);
fprintf('C = %d\n', C);

% e. Compute yl and yr values and corresponding xl, xr
xl = 0;
yl=(-A/B)*xl+C;
% Width of the image can be obtained from the whos command used in 3.1a
xr = 358-1;
yr=(-A/B)*xr+C;
fprintf('yl = %d\n', yl);
fprintf('yr = %d\n', yr);

% f. Display the original 'macritchie.jpg by superimposing the line
figure;
imshow(P_original,[]);
line([xl xr], [yl yr], 'Color', 'red');


%------------------3.3--------------------%
% b. Load the image
corridorl=imread('D:\NTU Class\CE4003 Computer Vision\Lab2 Edges, Hough Lines, and Disparity\corridorl.jpg');
corridorr=imread('D:\NTU Class\CE4003 Computer Vision\Lab2 Edges, Hough Lines, and Disparity\corridorr.jpg');
corridor_disp=imread('D:\NTU Class\CE4003 Computer Vision\Lab2 Edges, Hough Lines, and Disparity\corridor_disp.jpg');
corridorl=rgb2gray(corridorl);
corridorr=rgb2gray(corridorr);
% c. running the dis_map function
dispmap=disparitymap(corridorl,corridorr,11,11);
imshow(-dispmap,[-15 15]);

% d. Re-run algotihm on the real images of ‘triclops-i2l.jpg’ and triclops-i2r.jpg’
triclopsi2l=imread('D:\NTU Class\CE4003 Computer Vision\Lab2 Edges, Hough Lines, and Disparity\triclopsi2l.jpg');
triclopsi2R=imread('D:\NTU Class\CE4003 Computer Vision\Lab2 Edges, Hough Lines, and Disparity\triclopsi2R.jpg');
triclopsid=imread('D:\NTU Class\CE4003 Computer Vision\Lab2 Edges, Hough Lines, and Disparity\triclopsid.jpg');
triclopsi2l=rgb2gray(triclopsi2l);
triclopsi2R=rgb2gray(triclopsi2R);
dispmap=disparitymap(triclopsi2l, triclopsi2R, 11, 11);
imshow(-dispmap,[-15 15]);

% a. Write the disparity map algorithm
%dis_map function
function dispmap = disparitymap(Pl, Pr, m, n)
    [x,y]= size(Pl);
    xs=ceil((m+1)/2);
    ys=ceil((n+1)/2);
    dispmap=ones(x,y);
    Pln=zeros(x+xs,y+ys+150);
    Prn=zeros(x+xs,y+ys+150);
    Pln(1:x,15+(ys-1):y+14+(ys-1))=Pl;
    Prn(1:x,15+(ys-1):y+14+(ys-1))=Pr;
    for i=xs:x
         for j=ys+15+(ys-1):y+15+(ys-1)
             template=Pln(i-(xs-1):i+(xs-1),j-(ys-1):j+(ys-1));
             I=Prn(i-(xs-1):i+(xs-1),j-15-(ys-1):j+15+(ys-1));
             TD=double(template);
             TD=rot90(rot90(TD));
             Isquare=double(I).*(double(I));
             F=ones(m,n);
             Ss=conv2(Isquare,F,'same')-2*conv2(double(I),TD,'same');
             [~,sw]=size(Ss);
             Ssd=Ss(xs,ys:sw-(ys-1));
             [~,I]=min(Ssd);
             dispmap(i,j-15-(ys-1))=I-15;
         end
     end
end