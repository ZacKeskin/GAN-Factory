function [new_img,crop,obj_location]=extractSubImage(img,box,crop_margin)
% extractSubImage - extracts the face from an image with a margin
% if the margin goes beyond the image, the last pixel will be continued
%
% Syntax:  [new_img,crop,obj_location]=extractSubImage(img,box,crop_margin)
%
% Inputs:
%    img - image (i.e. load with imread)
%    box - location of face (i.e. img(box(2):box(4),box(1):box(3),:))
%    crop_margin - margin around face as a fraction of the width, height
%                  [left above right below], default is [0.4 0.4 0.4 0.4]
%
% Outputs:
%    new_img - cropped face with margin
%    crop - coordinates in original image of the cropped region
%    obj_location - coordinates in cropped image of the cropped region
%
% Example:
% Getting all faces of the wikipedia dataset with default margin:
% load('wiki/wiki.mat');
% for i=1:length(wiki.dob)
%    fprintf('Cropping face %d/%d\n',i,length(wiki.dob));
%    orig_img=imread(['wiki/' wiki.full_path{i}]);
%    cropped_face{i}=extractSubImage(orig_img,wiki.face_location{i});
% end
%
% Author: Rasmus Rothe
% Address: Sternwartstrasse 7, 8092 Zurich, Switzerland
% email: rrothe@vision.ee.ethz.ch
% Website: http://www.vision.ee.ethz.ch
% Jan 2016; Last revision: 16-Jan-2016

% set default margin
if nargin==2
    crop_margin=[0.4 0.4 0.4 0.4];
end

% size of face
orig_size(1)=box(4)-box(2)+1;
orig_size(2)=box(3)-box(1)+1;

% add margin
full_crop(1)=round(box(1)-crop_margin(1).*orig_size(2));
full_crop(2)=round(box(2)-crop_margin(2).*orig_size(1));
full_crop(3)=round(box(3)+crop_margin(3).*orig_size(2));
full_crop(4)=round(box(4)+crop_margin(4).*orig_size(1));

% size of face with margin
new_size(1)=full_crop(4)-full_crop(2)+1;
new_size(2)=full_crop(3)-full_crop(1)+1;

% ensure that the region cropped from the original image with margin
% doesn't go beyond the image size
crop(1)=max(full_crop(1),1);
crop(2)=max(full_crop(2),1);
crop(3)=min(full_crop(3),size(img,2));
crop(4)=min(full_crop(4),size(img,1));

% size of the actual region being cropped from the original image
crop_size(1)=crop(4)-crop(2)+1;
crop_size(2)=crop(3)-crop(1)+1;

% create new image
new_img=single(zeros(new_size(1),new_size(2),size(img,3)));

% coordinates of region taken out of the original image in the new image
new_location(1)=crop(1)-full_crop(1)+1;
new_location(2)=crop(2)-full_crop(2)+1;
new_location(3)=crop(1)-full_crop(1)+crop_size(2);
new_location(4)=crop(2)-full_crop(2)+crop_size(1);

% coordinates of the face in the new image
obj_location(1)=new_location(1)+box(1)-crop(1)+1;
obj_location(2)=new_location(2)+box(2)-crop(2)+1;
obj_location(3)=new_location(3)+box(3)-crop(3)+1;
obj_location(4)=new_location(4)+box(4)-crop(4)+1;

% do the crop
new_img(new_location(2):new_location(4),new_location(1):new_location(3),:)=...
    img(crop(2):crop(4),crop(1):crop(3),:);


% if margin goes beyond the size of the image, repeat last row of pixels
if new_location(2)>1
    new_img(1:new_location(2)-1,:,:)=repmat(new_img(new_location(2),:,:),[new_location(2)-1,1,]);
end
if new_location(4)<size(new_img,1)
    new_img(new_location(4)+1:end,:,:)=repmat(new_img(new_location(4),:,:),[size(new_img,1)-new_location(4),1,1]);
end
if new_location(1)>1
    new_img(:,1:new_location(1)-1,:)=repmat(new_img(:,new_location(1),:),[1,new_location(1)-1,1]);
end
if new_location(3)<size(new_img,2)
    new_img(:,new_location(3)+1:end,:)=repmat(new_img(:,new_location(3),:),[1,size(new_img,2)-new_location(3),1]);
end

end
