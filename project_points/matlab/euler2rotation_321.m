function [R] = euler2rotation_321(e)
% euler2rotation_321 : Converts a 3-2-1 euler-angle rotation representation
%                      to a rotation matrix.
%
% INPUTS
%
% e ---------- 3x1 vector of euler angles in radians representing rotations 
%              about the x, y, and z axes: e = [Rx, Ry, Rz]
%
% OUTPUTS
%
% R ---------- Rotation matrix
%
%+------------------------------------------------------------------------------+
% References:
%
%
% Author: Tucker Haydon
%+==============================================================================+ 
unitx = [1, 0, 0]';
unity = [0, 1, 0]';
unitz = [0, 0, 1]';

Rx = rotationMatrix(unitx, e(1));
Ry = rotationMatrix(unity, e(2));
Rz = rotationMatrix(unitz, e(3));

% 3-2-1 rotation
R = Rz * Ry * Rx;
end