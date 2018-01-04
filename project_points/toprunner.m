%% Siddarth Kaki - Thesis

%% housekeeping
clear
close all
clc

%% open db connection
db_file_id = '/home/siddarthkaki/workspace/brick_seal/database2.db';
%db_file_id = '/home/siddarthkaki/workspace/projectPoints/database.db';
db = com.almworks.sqlite4java.SQLiteConnection(java.io.File(db_file_id));
db.open;

%% read in 3D points

pc_file_id = '~/workspace/brick_seal/sparse/0/points3D.txt';

num_points = linecount(pc_file_id) - 3;

pos = zeros(num_points, 3);

mean_descriptors = zeros(num_points,128);

tic
% loop though points in 3D point cloud
for i = 1:50:num_points,
    curr_line = import_points3D_line(pc_file_id, i); % extract specific line
    
    curr_id = curr_line(1); % POINT3D_ID
    curr_pos = curr_line(2:4); % X,Y,Z
    pos(i,1:3) = curr_pos;
    
    num_tracks = (length(curr_line) - 8)/2; % number of images with this point
    
    curr_descriptors = zeros(num_tracks,128);
    
    % loop through each image with this point
    for j = 1:num_tracks,
        curr_image_id = curr_line(7+2*j); % IMAGE_ID
        curr_point2d_idx = curr_line(8+2*j); % POINT2D_IDX
                
        %st = db.prepare(['SELECT * FROM cameras'])
        st = db.prepare(['SELECT * FROM descriptors WHERE image_id=' num2str(curr_image_id)]);
        while st.step,
            % returning the data type from the desired column
            curr_num_descriptors = st.columnInt(1); % get num of descriptors from column 1
            curr_descriptors_blob = st.columnBlob(3); % get descriptors blob from column 3
            
            curr_start = (curr_point2d_idx-1)*128+1;
            curr_end = curr_start+127;
            curr_descriptor = curr_descriptors_blob(curr_start:curr_end);
            
            curr_descriptors(j,:) = curr_descriptor;
        end
    end
    
    mean_descriptors(i,:) = mean(curr_descriptors,1);
    
end
toc

points = pos;

save('points.mat', 'points')
%load('points.mat')

%% projection

image_size_x = 1000;
image_size_y = 1000;

focal_length_x = 500; %image_size_x / 2;
focal_length_y = 500; %image_size_y / 2;

skew = 0;

% setup camera intrinsics with focal length 200, centre 500,500
cam = [focal_length_x,           skew, image_size_x/2;
                    0, focal_length_y, image_size_y/2;
                    0,              0,              1];

% setup image pixel dimensions
image_size = [image_size_y, image_size_x];

% create a transform matrix
angles = deg2rad([5,-5,75]);
position = [-1,-4,5];
tform = eye(4);
%tform(1:3,1:3) = angle2dcm(angles);
tform(1:3,4) = position;

% lense distortion
dist = []; %[0.1,0.005];

% project the points into image coordinates
[projected, valid] = projectPoints(points, cam, tform, dist, image_size, true);
projected = projected(valid,:);

% visualise the projection
subplot(1,2,1)
scatter3(points(:,1),points(:,2),points(:,3));%,20,points(:,4:6),'fill')
axis equal
title('Original Points')
xlabel('x (m)')
ylabel('y (m)')
zlabel('z (m)')

subplot(1,2,2)
scatter(projected(:,1),projected(:,2));%,20,projected(:,3:5),'fill')
axis equal
title('Points projected with camera model')

%% dispose st connection
% disposed of used up statement container
st.dispose
st.isDisposed
 
%% dispose db connection
db.dispose;
db.isDisposed