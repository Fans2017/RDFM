clear; close all;
warning off;
load('./data/OO5.mat');

tic
config = struct();

config.img_scale = [1,2,4,8];
config.use_layers = {{'conv1_2'},{'conv2_2'},{'conv3_2'},{'conv4_3'}};
config.use_ker_sz = {[5,5],[5,15],5,15};
config.use_patch_sz = {[5,5],[7,15],9,0};
config.use_search_sz = {[3,3],[3,3],3,0};

net = vgg19;
img0 = I_move;
img1 = I_fix;

[matchedPoints1, matchedPoints2] = hierarchicalMatching(img0, img1, config, net);
toc

% outlier remove
H=FSC(matchedPoints1,matchedPoints2,'affine',2);
Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
inliersIndex=E<2;
matchedPoints1 = matchedPoints1(inliersIndex,:);
matchedPoints2 = matchedPoints2(inliersIndex,:);

figure(), showMatchedFeatures(I_move, I_fix, matchedPoints1(:,1:2), matchedPoints2(:,1:2), 'montage'); 
