clear; close all;
warning off;
load('./data/DN5.mat');

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

feature0_pyramid = constructPyramidfeatureUsingVGG(img0, config, net);
feature1_pyramid = constructPyramidfeatureUsingVGG(img1, config, net);

init_pts0 = [];
init_pts1 = [];

for ii = length(config.img_scale):-1:1
    ker_sz = config.use_ker_sz{ii};
    patch_sz = config.use_patch_sz{ii};
    search_sz = config.use_search_sz{ii};
    
    tmp_pts0 = [];
    tmp_pts1 = [];
    for kk = 1:size(feature0_pyramid{ii},2)
        feat0 = feature0_pyramid{ii}{kk};
        feat1 = feature1_pyramid{ii}{kk};
        [pts0_list, pts1_list] = MatchFrame(feat0, feat1, init_pts0, init_pts1, ker_sz(kk), patch_sz(kk), search_sz(kk));
        tmp_pts0 = cat(1, tmp_pts0, pts0_list);
        tmp_pts1 = cat(1, tmp_pts1, pts1_list);
    end
    
    if isempty(tmp_pts0)
        disp('no pts, failed!');
        break;
    end
    if ii >1
        init_pts0 = (pts0_list-1)*config.img_scale(ii)/config.img_scale(ii-1);
        init_pts1 = (pts1_list-1)*config.img_scale(ii)/config.img_scale(ii-1);
    end
end
toc

% outlier remove
matchedPoints1 = tmp_pts0;
matchedPoints2 = tmp_pts1;
H=FSC(matchedPoints1,matchedPoints2,'affine',2);
Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
inliersIndex=E<2;
matchedPoints1 = matchedPoints1(inliersIndex,:);
matchedPoints2 = matchedPoints2(inliersIndex,:);

figure(), showMatchedFeatures(I_move, I_fix, matchedPoints1(:,1:2), matchedPoints2(:,1:2), 'montage'); 
