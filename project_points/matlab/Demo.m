%Quick Program to demo the use of projectPoints

%% generate a set of 3d points
z = peaks;
x = repmat(1:size(z,1),size(z,1),1);
y = x';
c = z - min(z(:));
c = c./max(c(:));
c = round(255*c) + 1;
cmap = colormap(jet(256));
c = cmap(c,:);

points = [x(:),y(:),z(:),c];

%% setup

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
position = [-25,-25,500];
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
scatter3(points(:,1),points(:,2),points(:,3),20,points(:,4:6),'fill')
axis equal
title('Original Points')

subplot(1,2,2)
scatter(projected(:,1),projected(:,2),20,projected(:,3:5),'fill')
axis equal
title('Points projected with camera model')