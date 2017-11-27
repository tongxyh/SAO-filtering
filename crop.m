% org = imread('C:\Users\XiaoYu\Downloads\DIV2K\DIV2K\DIV2K_train_HR\0003.png');
% coded = imread('J:\DIV2K_2040_1352_482_intra_main_JEM7.1_10bit_DFonSAOoff_Q40\26.png');
% org2 = imresize(org,[1352,2040]);
% res = org2 - coded;
% sum(sum(sum(res)))

% Files=dir('C:\Users\XiaoYu\Downloads\DIV2K\DIV2K\DIV2K_train_HR\*.png');
% for k=1:length(Files)
%    a = imread(['C:\Users\XiaoYu\Downloads\DIV2K\DIV2K\DIV2K_train_HR\', Files(k).name]);
%    [h,w,c] = size(a);
%    if ( h == 1356 && w == 2040)
%         imwrite(a,['C:\Users\XiaoYu\Downloads\DIV2K\DIV2K\NEW\',num2str(k),'.png'])
%    end
% end

Files=dir('C:\Users\XiaoYu\Downloads\DIV2K\DIV2K\NEW\*.png');
for k=1:length(Files)
   a = imread(['C:\Users\XiaoYu\Downloads\DIV2K\DIV2K\NEW\', Files(k).name]);
   imwrite(a,['C:\Users\XiaoYu\Downloads\DIV2K\DIV2K\Input\',num2str(k),'.png'])
end