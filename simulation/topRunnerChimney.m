%% Siddarth Kaki - Thesis

%% housekeeping
clear all; close all; clc;

%% read in 3D points
pc_file_id = fullfile(pwd, '../data/global_map_chimney/sparse/0/points3D.txt');
fid = fopen(pc_file_id);
data_raw = cell2mat(textscan(fid, [ repmat('%f',[1,4]), '%*[^\n]' ], 'Delimiter', ' ', 'EmptyValue', NaN, 'HeaderLines', 3));
fclose(fid);

mapNumPoints = linecount(pc_file_id) - 3;

posTemp = data_raw(:,2:4);
mapKeypoints = posTemp;

% % generate random descriptors for real keypoints
% mapDescriptors = randi([0,128],[mapNumPoints,128]);

%% read in camera poses

cp_file_id = fullfile(pwd, '../data/global_map_chimney/sparse/0/image_poses.txt');
fid = fopen(cp_file_id);
cp_data_raw_temp = textscan(fid, ['%C', repmat('%f',[1,7]) ], 'Delimiter', ' ', 'EmptyValue', NaN);
cp_data_raw = cell2mat(cp_data_raw_temp(1:end,2:end));
fclose(fid);

cam_pos = cp_data_raw(:,1:3);
cam_rot_quat = cp_data_raw(:,4:7);
cam_rot_dcm = zeros(length(cam_rot_quat),9);

%% zero
riG = [-742015.2849821189, -5462219.4951718654, 3198014.4005017849]';
RIW = Recef2enu(riG);

for i = 1:length(mapKeypoints),
    mapKeypoints(i,:) = (RIW*(mapKeypoints(i,:)' - riG))';
end

for i = 1:length(cam_pos),
    cam_pos(i,:) = (RIW*(cam_pos(i,:)' - riG))';
    cam_rot_dcm(i,:) = reshape(quat2dcm(cam_rot_quat(i,:)), [1 9]);
end

%% filter down to close by points
distanceThreshold = 3;
pointsNew = [];
descriptorsNew = [];
for i = 1:length(mapKeypoints),
    if norm(mapKeypoints(i,:)) < distanceThreshold,
        pointsNew = [pointsNew; mapKeypoints(i,:)];
    else
        mapNumPoints = mapNumPoints - 1;
        %descriptorsNew = [descriptorsNew; mapDescriptors(i,:)];
    end
end

mapKeypoints = pointsNew;
% generate random descriptors for real keypoints
mapDescriptors = randi([0,128],[mapNumPoints,128]);
%mapDescriptors = descriptorsNew;

mapX = mapKeypoints(:,1);
mapY = mapKeypoints(:,2);
mapZ = mapKeypoints(:,3);

%% generate new object keypoints and descriptors
objL = 0.25; % m
mapL = 2.5; % m
cg = [0 -.75 -0.75];%-(mapL-objL) + rand(1,3)*(2*(mapL-objL));
cg = [0 0 0.25];
objNumPoints = 100;

[objKeypoints,objDescriptors] = generateSphere(objL,cg,objNumPoints);

objX = objKeypoints(:,1);
objY = objKeypoints(:,2);
objZ = objKeypoints(:,3);

figure,
subplot(1,2,1)
scatter3(mapX,mapY,mapZ)
axis equal
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
grid on
hold on
title('Prior Map')

subplot(1,2,2)
scatter3(mapX,mapY,mapZ)
axis equal
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
grid on
hold on
scatter3(objX,objY,objZ,'r')
title('Environment with New Object')

%% insert object into environment

% simulate extraneous features
extL = distanceThreshold;%5; % m
extNumPoints = 250;
extKeypoints = -extL + rand(extNumPoints,3)*(2*extL);
extDescriptors = randi([0,128],[extNumPoints,128]);

% construct scene
sceneKeypoints = [mapKeypoints; objKeypoints; extKeypoints];
sceneDescriptors = [mapDescriptors; objDescriptors; extDescriptors];

scnX = sceneKeypoints(:,1);
scnY = sceneKeypoints(:,2);
scnZ = sceneKeypoints(:,3);


%% projection

image_size_x = 3840;
image_size_y = 2160;
focal_length_x = 1691;%4608; %image_size_x / 2;
focal_length_y = 1697;%4608; %image_size_y / 2;
center_x = 1914;
center_y = 1073;
skew = 0;
% setup camera intrinsics with focal length 200, centre 500,500
cam = [focal_length_x,           skew, center_x;
                    0, focal_length_y, center_y;
                    0,              0,        1];
% setup image pixel dimensions
image_size = [image_size_y, image_size_x];

idx = 33 + 1;
camOrientationCorrection = rotationMatrix([1 0 0]',deg2rad(90))*rotationMatrix([0 0 1]',deg2rad(-90));
dcm = camOrientationCorrection*(RIW*reshape(cam_rot_dcm(idx,:),[3 3])')';

position = cam_pos(idx,:)';
position = dcm*(-position);

tform = eye(4);
tform(1:3,1:3) = dcm;%angle2dcm(angles);
tform(1:3,4) = position;

% lense distortion
dist = []; %[0.1,0.005];

[projected,valid] = camProjection(sceneKeypoints, cam, dcm, position);
projected = projected(valid,:);
projected(:,1) = projected(:,1);% + image_size_x/2;
projected(:,2) = projected(:,2);% - image_size_y;
projectedKeypoints = sceneKeypoints(valid,:);
projectedDescriptors = sceneDescriptors(valid,:);

[projectedMap,validMap] = camProjection(mapKeypoints, cam, dcm, position);
projectedMap = projectedMap(validMap,:);
projectedMap(:,1) = projectedMap(:,1);% + image_size_x/2;
projectedMap(:,2) = projectedMap(:,2);% - image_size_y;

%% simulate descriptor noise
mag = 10;
projectedDescriptors = projectedDescriptors + randi([-mag,mag], size(projectedDescriptors));

%% feature matching
ratio = 0.75;
indexPairs = matchFeatures(projectedDescriptors, mapDescriptors,'Method','Approximate','MaxRatio',ratio);

if isempty(indexPairs),
    isolated = zeros(2);
else
    isolated = projected;
    isolated(indexPairs(:,1),:) = [];
end

%% visualise object localisation
figure,
subplot(1,2,1)
scatter(projectedMap(:,1),projectedMap(:,2));%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Map Projection')
xlabel('X (pixels)')
ylabel('Y (pixels)')
xlim([0 image_size_x])
ylim([0 image_size_y])

subplot(1,2,2)
scatter(projected(:,1),projected(:,2));%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Simulated Camera View')
xlabel('X (pixels)')
ylabel('Y (pixels)')
xlim([0 image_size_x])
ylim([0 image_size_y])

figure,
subplot(1,2,1)
scatter(projectedMap(:,1),projectedMap(:,2));%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Map Projection')
xlabel('X (pixels)')
ylabel('Y (pixels)')
xlim([0 image_size_x])
ylim([0 image_size_y])

subplot(1,2,2)
scatter(projected(:,1),projected(:,2));%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Simulated Camera View with Deltas')
xlabel('X (pixels)')
ylabel('Y (pixels)')
hold on
scatter(isolated(:,1),isolated(:,2),'r');
xlim([0 image_size_x])
ylim([0 image_size_y])

%% dbscan clustering
addpath dbscan
epsilon = 0.2;
MinPts = 15;
IDX = DBSCAN(isolated/1000,epsilon,MinPts);
PlotClusterinResult(isolated/1000, IDX);
xlabel('X (megapixels)')
ylabel('Y (megapixels)')
xlim([0,image_size_x/1000]);
ylim([0,image_size_y/1000]);
title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);
rmpath dbscan

%%

% %% visualise the projection
% figure,
% scatter3(points(:,1),points(:,2),points(:,3));%,20,points(:,4:6),'fill')
% axis equal
% title('Original Points')
% xlabel('x (m) east')
% ylabel('y (m) north')
% zlabel('z (m) up')
% hold on
% RCI = reshape(cam_rot_dcm(idx,:),[3 3]);
% posTemp = cam_pos(idx,:)';
% uVecTemp = RIW*RCI'*[1 0 0]';
% quiver3(posTemp(1), posTemp(2), posTemp(3), uVecTemp(1), uVecTemp(2), uVecTemp(3),'r');
% 
% %%
% figure,
% for i = 1:length(cam_pos),
%     RCI = reshape(cam_rot_dcm(i,:),[3 3]);
%     posTemp = cam_pos(i,:)';
%     uVecTemp = RIW*RCI'*[1 0 0]';
%     quiver3(posTemp(1), posTemp(2), posTemp(3), uVecTemp(1), uVecTemp(2), uVecTemp(3));
%     scatter3(posTemp(1), posTemp(2), posTemp(3));
%     hold on
% end
% axis equal
% xlabel('x (m) east')
% ylabel('y (m) north')
% zlabel('z (m) up')

%%
figure,
subplot(1,2,1)
scatter3(mapKeypoints(:,1),mapKeypoints(:,2),mapKeypoints(:,3));%,20,points(:,4:6),'fill')
axis equal
title('Original Points')
xlabel('x (m) east')
ylabel('y (m) north')
zlabel('z (m) up')
hold on
RCI = reshape(cam_rot_dcm(idx,:),[3 3]);
posTemp = cam_pos(idx,:)';
uVecTemp = RIW*RCI'*[1 0 0]';
quiver3(posTemp(1), posTemp(2), posTemp(3), uVecTemp(1), uVecTemp(2), uVecTemp(3),'r');

subplot(1,2,2)
scatter(projected(:,1),projected(:,2));%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Points projected with camera model')
xlim([-image_size_x/2 image_size_x/2])
ylim([-image_size_y/2 image_size_y/2])

xlim([0 image_size_x])
ylim([0 image_size_y])