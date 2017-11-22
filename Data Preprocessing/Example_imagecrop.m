% Example:
% Getting all faces of the wikipedia dataset with default margin:
 load('wiki/wiki.mat');
 for i=1:length(wiki.dob)
    fprintf('Cropping face %d/%d\n',i,length(wiki.dob));
    orig_img=imread(['wiki/' wiki.full_path{i}]);
    cropped_face{i}=extractSubImage(orig_img,wiki.face_location{i});
 end