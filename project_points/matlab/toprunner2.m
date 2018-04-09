%% Siddarth Kaki - Thesis

%% housekeeping
clear all; close all; clc;

%% read in 3D points
pc_file_id = fullfile(pwd, '../../data/global_map_chimney/sparse/0/points3D.txt');
fid = fopen(pc_file_id);
data_raw = cell2mat(textscan(fid, [ repmat('%f',[1,4]), '%*[^\n]' ], 'Delimiter', ' ', 'EmptyValue', NaN, 'HeaderLines', 3));
fclose(fid);

num_points = linecount(pc_file_id) - 3;

posTemp = data_raw(:,2:4);
points = posTemp;

%% read in camera poses

cp_file_id = fullfile(pwd, '../../data/global_map_chimney/sparse/0/image_poses.txt');
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

for i = 1:length(points),
    points(i,:) = (RIW*(points(i,:)' - riG))';
end

for i = 1:length(cam_pos),
    cam_pos(i,:) = (RIW*(cam_pos(i,:)' - riG))';
    cam_rot_dcm(i,:) = reshape(quat2dcm(cam_rot_quat(i,:)), [1 9]);
end

%% filter down to close by points
pointsNew = [];
for i = 1:length(points),
    if norm(points(i,:)) < 1.5,
        pointsNew = [pointsNew; points(i,:)];
    end
end
points = pointsNew;

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

% create a transform matrix
%angles = deg2rad([5,-5,75]);
%angles = deg2rad([0,0,0]);

% dcm = RIW*[ 0.572438692476     , 0.785728043122    , 0.234403879794;
%         0.07942869324599994, 0.23139426555     ,-0.969611986778;
%        -0.8160911832019999 , 0.5736617160539998, 0.07004974347000004]';
   
idx = 33 + 1;
camOrientationCorrection = rotationMatrix([1 0 0]',deg2rad(90))*rotationMatrix([0 0 1]',deg2rad(-90));
dcm = camOrientationCorrection*(RIW*reshape(cam_rot_dcm(idx,:),[3 3])')';
%dcm = eye(3);
%dcm = euler2dcm(deg2rad([0, 0, 0]')); % 3-1-2
position = cam_pos(idx,:)';
position = dcm*(-position);
%position = [0, 0, 5]';

tform = eye(4);
tform(1:3,1:3) = dcm;%angle2dcm(angles);
tform(1:3,4) = position;

% lense distortion
dist = []; %[0.1,0.005];

% project the points into image coordinates
% [projected, valid] = project_points(points, cam, tform, dist, image_size, true);
% projected = projected(valid,:);
projected = camProjection(points, cam, dcm, position);
projected(:,1) = projected(:,1);% + image_size_x/2;
projected(:,2) = projected(:,2) - image_size_y;

%% visualise the projection
figure,
scatter3(points(:,1),points(:,2),points(:,3));%,20,points(:,4:6),'fill')
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


%%
figure,
for i = 1:length(cam_pos),
    RCI = reshape(cam_rot_dcm(i,:),[3 3]);
    posTemp = cam_pos(i,:)';
    uVecTemp = RIW*RCI'*[1 0 0]';
    quiver3(posTemp(1), posTemp(2), posTemp(3), uVecTemp(1), uVecTemp(2), uVecTemp(3));
    scatter3(posTemp(1), posTemp(2), posTemp(3));
    hold on
end
axis equal
xlabel('x (m) east')
ylabel('y (m) north')
zlabel('z (m) up')

%%
figure,
subplot(1,2,1)
scatter3(points(:,1),points(:,2),points(:,3));%,20,points(:,4:6),'fill')
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
ylim([-image_size_y 0])