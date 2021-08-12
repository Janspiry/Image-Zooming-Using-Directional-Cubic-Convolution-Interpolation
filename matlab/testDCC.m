
function testDCC
%
% Copyright (c) May 28, 2009. Dengwen Zhou. All rights reserved.
% Department of Computer Science & Technology
% North China Electric Power University(NCEPU)
%
% Last time modified: Oct. 11, 2012
%

close all
clear all
clc
% Set the zooming level
level = 2;
%Verbose
Point_Num = 51;
Img_Num = 2;
% Hyper-parameters
k = 5; 
T = 1.15;

MSE_SUM = 0;
SNR_SUM = 0;
PSNR_SUM = 0;
IMG_CNT = 0;
for idx=0:Img_Num
    name = num2str(idx,'%05d');
    type = '.jpg';
    ifname = ['..\data\hr\' name type];
    ORIG_RGB = imread(ifname);
    SR_RGB = ORIG_RGB;
    for channel=1:3
        ORIG = ORIG_RGB(:,:,channel);
        % Downsample the original image
        ORIG_LR = ORIG(1:2^level:end-1,1:2^level:end-1);
        ORIG_LR = im2double(ORIG_LR);
        % Do the interpolation 
        tic;
        OUTgo = ORIG_LR;
        for s = 1:level 
            OUT = DCC(OUTgo,k,T);
            OUTgo = OUT;
        end
        toc
        OUT = im2uint8(OUT);
        SR_RGB(:,:, channel)=OUT;
    end
    OUT = SR_RGB;
    % SR_RGB = reshape(SR_RGB,256,256,[]);
    % Save the interpolated image
    output_dir = ['..\data\sr\'];
    if isdir(output_dir) == 0
        mkdir(output_dir);
    end
    ifname = [output_dir '\' name '_dcc_matlab.png'];
    imwrite(SR_RGB,ifname);

    % Compute error
    OUT = imread(ifname);
    b = 12;
    [MSE, SNR, PSNR] = Calc_MSE_SNR(ORIG_RGB,OUT,b);
    MSE_SUM = MSE_SUM+MSE;
    SNR_SUM = SNR_SUM+SNR;
    PSNR_SUM = PSNR_SUM+PSNR;
    IMG_CNT = IMG_CNT+1;
end
disp(['MSE = ', num2str(MSE_SUM/IMG_CNT)]);
disp(['SNR = ', num2str(SNR_SUM/IMG_CNT)]);
disp(['PSNR = ', num2str(PSNR_SUM/IMG_CNT)]);
end