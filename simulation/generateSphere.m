function [objKeypoints,objDescriptors] = generateSphere(radius,cg,points)
%GENERATESPHERE Summary of this function goes here
%   Detailed explanation goes here

objL = radius; % m
objNumPoints = points;

objKeypoints = zeros(objNumPoints,3);

for i = 1:objNumPoints,
    objKeypoints(i,:) = (objL*euler2dcm(randn(3,1)*2*pi)*[1,0,0]')' + cg;
end

objDescriptors = randi([0,128],[objNumPoints,128]);
end

