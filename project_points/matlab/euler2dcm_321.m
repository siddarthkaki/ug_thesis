function [DCM] = euler2dcm_321(e)
% euler2dcm_321 : Converts a 3-2-1 euler-angle rotation representation of
%                 how frame B is rotated away from frame A and creates a 
%                 direction cosine matrix that takes a vector from frame A and 
%                 expresses it in frame B.
%
% INPUTS
%
% e ---------- 3x1 vector of euler angles in radians representing rotations 
%              about the x, y, and z axes of frame B away from frame A: 
%              e = [Rx, Ry, Rz]
%
% OUTPUTS
%
% DCM ---------- Direction cosine matrix
%
%+------------------------------------------------------------------------------+
% References:
%
%
% Author: Tucker Haydon
%+==============================================================================+ 
% DCM is the inverse (transpose) of the rotation matrix. 
DCM = euler2rotation_321(e)';
end