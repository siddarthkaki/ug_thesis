function points3D = import_points3D_line(filename, linenum)
%% init variable(s)
lineind = linenum + 3;

%% open text file
fid = fopen(filename,'r');

%% read specified 3D point from file
dataArray = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',lineind-1);
dataArray_num = str2num(cell2mat(dataArray{1}));

%% close text file
fclose(fid);

%% create output variable
points3D = dataArray_num;

end