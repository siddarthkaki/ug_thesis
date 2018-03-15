%% Siddarth Kaki - Thesis

%% housekeeping
clear; close all; clc;

%% generate environment keypoints and descriptors

mapL = 5; % m
mapNumPoints = 100;

mapKeypoints = -mapL + rand(mapNumPoints,3)*(2*mapL);
mapDescriptors = randi([0,128],[mapNumPoints,128]);

%% generate map objects

objL = 1;
objNumPoints = 100;

for i = 1:5,
    cg = -(mapL-objL) + rand(1,3)*(2*(mapL-objL));
    [tempKeypoints,tempDescriptors] = generateSphere(objL,cg,objNumPoints);
    
    mapKeypoints = [mapKeypoints; tempKeypoints];
    mapDescriptors = [mapDescriptors; tempDescriptors];
end

mapX = mapKeypoints(:,1);
mapY = mapKeypoints(:,2);
mapZ = mapKeypoints(:,3);

figure,
subplot(1,2,1)
scatter3(mapX,mapY,mapZ,'b')
axis equal
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
grid on
hold on
title('Prior Map')

%% generate new object keypoints and descriptors

objL = 1; % m
cg = -(mapL-objL) + rand(1,3)*(2*(mapL-objL));
objNumPoints = 100;

[objKeypoints,objDescriptors] = generateSphere(objL,cg,objNumPoints);

objX = objKeypoints(:,1);
objY = objKeypoints(:,2);
objZ = objKeypoints(:,3);

subplot(1,2,2)
scatter3(mapX,mapY,mapZ,'b')
axis equal
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
grid on
hold on
scatter3(objX,objY,objZ,'r')
title('Map with New Object')

%% insert object into environment

% simulate extraneous features
extL = 5; % m
extNumPoints = 50;
extKeypoints = -extL + rand(extNumPoints,3)*(2*extL);
extDescriptors = randi([0,128],[extNumPoints,128]);

% construct scene
sceneKeypoints = [mapKeypoints; objKeypoints; extKeypoints];
sceneDescriptors = [mapDescriptors; objDescriptors; extDescriptors];

scnX = sceneKeypoints(:,1);
scnY = sceneKeypoints(:,2);
scnZ = sceneKeypoints(:,3);

%% camera projection
image_size_x = 1080;
image_size_y = 1440;

focal_length_x = max(image_size_x,image_size_y) / 2;
focal_length_y = max(image_size_x,image_size_y) / 2;

skew = 0;

% setup camera intrinsics with focal length 200, centre 500,500
cam = [focal_length_x,           skew, image_size_x/2;
                    0, focal_length_y, image_size_y/2;
                    0,              0,              1];

% setup image pixel dimensions
image_size = [image_size_y, image_size_x];

% create a transform matrix
angles = deg2rad([5,-5,75]);
angles = deg2rad([0,0,0]);
position = [0,0,5];
% position = [-0.743449, -3.23341, 7.8009];
% position = [-0.540707276541575,-1.443508861309167, 9.377850391746321];
dcm = [-0.3151025836926599, -0.4698286571043252, 0.8246037804386764;
        0.3746034626970745, 0.7367473357271332, 0.5629170534229119;
       -0.8719992015395267, 0.486276049443645, -0.05615154719070636];
tform = eye(4);
tform(1:3,1:3) = angle2dcm(angles(1),angles(2),angles(3));
tform(1:3,4) = position;

% lense distortion
dist = []; %[0.1,0.005];

% project the points into image coordinates
[projected, valid] = project_points(sceneKeypoints, cam, tform, dist, image_size, false);
projected = projected(valid,:);
projectedKeypoints = sceneKeypoints(valid,:);
projectedDescriptors = sceneDescriptors(valid,:);

% projected map

[projectedMap, validMap] = project_points(mapKeypoints, cam, tform, dist, image_size, false);
projectedMap = projectedMap(validMap,:);


%% visualise projection
% figure, subplot(1,2,1)
% scatter3(projectedKeypoints(:,1),projectedKeypoints(:,2),projectedKeypoints(:,3));%,20,points(:,4:6),'fill')
% axis equal
% grid on
% title('Original Points')
% xlabel('X (m)')
% ylabel('Y (m)')
% zlabel('Z (m)')
% 
% subplot(1,2,2)
% scatter(projected(:,1),projected(:,2));%,20,projected(:,3:5),'fill')
% axis equal
% grid on
% title('Points projected with camera model')
% xlabel('X (pixels)')
% ylabel('Y (pixels)')

%% simulate camera pixel noise
% projected = projected + randn(size(projected))*5;

%% simulate descriptor noise
mag = 20;
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

%% dbscan clustering
% addpath dbscan
% epsilon = 0.9;
% MinPts = 50;
% IDX = DBSCAN(isolated,epsilon,MinPts);
% PlotClusterinResult(isolated, IDX);
% title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);
% rmpath dbscan

%% visualise object localisation
figure,
subplot(1,2,1)
scatter(projectedMap(:,1),projectedMap(:,2),'b');%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Map Projection')
xlabel('X (pixels)')
ylabel('Y (pixels)')
xlim([0,image_size_x]);
ylim([0,image_size_y]);

subplot(1,2,2)
scatter(projected(:,1),projected(:,2),'b');%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Camera View')
xlabel('X (pixels)')
ylabel('Y (pixels)')
xlim([0,image_size_x]);
ylim([0,image_size_y]);

figure,
subplot(1,2,1)
scatter(projectedMap(:,1),projectedMap(:,2),'b');%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Map Projection')
xlabel('X (pixels)')
ylabel('Y (pixels)')
xlim([0,image_size_x]);
ylim([0,image_size_y]);

subplot(1,2,2)
scatter(projected(:,1),projected(:,2),'b');%,20,projected(:,3:5),'fill')
axis equal
grid on
title('Camera View with New Object Localised')
xlabel('X (pixels)')
ylabel('Y (pixels)')
hold on
scatter(isolated(:,1),isolated(:,2),'r');
xlim([0,image_size_x]);
ylim([0,image_size_y]);

%% save to txt and db
% 
% pc_file_id = fullfile(pwd, 'data/sparse/0/points3D_sim.txt');
% 
% fid = fopen(pc_file_id,'w');
% fmt = '%5d %5d %5d %5d\n';
% fprintf(fid,fmt, magic(4));
% 
% % open db connection
% db_file_id = fullfile(pwd, 'data/database_sim.db');
% db = com.almworks.sqlite4java.SQLiteConnection(java.io.File(db_file_id));
% db.open;