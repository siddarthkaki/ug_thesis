function [xMat, valid] = camProjection(XMat, KMat, RCI, tVec)
%CAMPROJECTION Summary of this function goes here
%   Detailed explanation goes here

    image_size_x = 3840;
    image_size_y = 2160;
       
    numPoints = size(XMat,1);

    xMat = zeros(numPoints,2);
    valid = false(numPoints,1);
    
    PMat = KMat*[RCI, tVec];
    
    for i = 1:numPoints,
        
        XHomoVec = [XMat(i,:), 1]';
        xHomoVec = PMat*XHomoVec;
        
        x = xHomoVec(1) / xHomoVec(3);
        y = xHomoVec(2) / xHomoVec(3);
        
        xMat(i,:) = [x, y];
        %xMat = [xMat; x, y];
        
        if x >=0 && x <= image_size_x && y >= 0 && y <= image_size_y,
            valid(i) = true;
        end
    end

end

