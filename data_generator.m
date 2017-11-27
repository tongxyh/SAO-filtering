clear;close all;
%% parameters
patch_size = 64;
stride = 25;

%% generate data
data = zeros(patch_size,patch_size,3,1);

count = 0;

for i=1:400
    im_gdt = im2double(imread(['C:\Users\lhj\Desktop\Adaptive compression\Train_gdt\',num2str(i),'.bmp']));
    %im_gdt = imread(['C:\Users\lhj\Desktop\Adaptive compression\koda\',num2str(i),'.bmp']);
    %im_gdt = (double(im_gdt)-128.0)/128.0;
    [H,W,C] = size(im_gdt);
    for x=1:stride:H-patch_size+1
        for y=1:stride:W-patch_size+1
            subim_input = im_gdt(x:x+patch_size-1,y:y+patch_size-1,:);
            count = count+1;
            count
            data(:,:,:,count) = subim_input;
            
        end
    end
       
end
h5create('C:\Users\lhj\Desktop\Adaptive compression\train64.h5','/data',[patch_size patch_size 3 count]);
h5write('C:\Users\lhj\Desktop\Adaptive compression\train64.h5','/data',data);