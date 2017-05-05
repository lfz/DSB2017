clear
clc
close all
lungwindow = [-1900,1100];
lumTrans = @(x) uint8((x-lungwindow(1))/(diff(lungwindow))*256);

path = 'E:\Kaggle.Data\stage1';
cases  = dir(path);
cases = {cases.name};
cases = cases(3:end);
    header = {'id',	'coordx1','coordx1','coordx1','diameter'};

labelfile = 'label_job2.csv';
if ~ exist(labelfile)
    
    initial_label = header;
    for i = 1:length(cases)
        initial_label = [initial_label;{cases{i},'x','x','x','x'}];
    end
    cell2csv(labelfile,initial_label)
    label_tabel = initial_label;
else
    label_tabel = csv2cell(labelfile,'fromfile');
end

label_tabel = label_tabel(2:end,:);
fullnamelist = label_tabel(:,1);
uniqueNameList = unique(fullnamelist, 'stable');
% uniqueNameList = fullnamelist(ismember(fullnamelist, uniqueNameList));
annos = label_tabel(:,2:end);
for i = 1:size(fullnamelist)
    if annos{i,1}=='x'
        lineid = i;
        name = fullnamelist{i};
        id = find(strcmp(uniqueNameList,name));
        break
    end
end


for id = id:length(uniqueNameList)
    name = uniqueNameList{id};
    disp(name)
    found = 0;
    folder = [path,'/',name];
%     info = dicom_folder_info(folder);
    im = dicomfolder(folder);
    imint8 = lumTrans(im);
    rgbim = repmat(imint8,[1,1,1,3]);
    
    h1 = figure(1);
    imshow3D(rgbim)

    while 1
        
        in = input('add_square(a), add_diameter(b), delete_last(d), or next(n):','s');
        if in =='n'
            if found==0
                label_tabel(lineid,:) = {name,0,0,0,0};
                lineid = lineid+1;
            end
            break
        elseif in =='d'
            found = found-1;
            lineid = lineid-1;
            if found ==0
                label_tabel(lineid,:) = {name,'x','x','x','x'};
                
            elseif found>0
                label_tabel(lineid,:) = [];
            else
                disp('invalid delete')
            end
            if found>=0
                rgbim=rgbim_back;
                imshow3D(rgbim)
            end
            figure(1);
        elseif strcmp(in,'a')||strcmp(in,'b')
            if found==0
                label_tabel(lineid,:) = [];
            end
            if strcmp(in,'a')
                anno = label_rect(im);
            elseif strcmp(in,'b')
                anno = label_line();
            end
            found=found+1;
            label_tabel= [label_tabel(1:lineid-1,:);{name,anno(1),anno(2),anno(3),anno(4)};label_tabel(lineid:end,:)];
            lineid = lineid+1;
            rgbim_back = rgbim;
            rgbim = drawRect(rgbim,anno,1);
                imshow3D(rgbim)
        else
            figure(1);
            continue
        end
%         disp(label_tabel(max([1, (lineid - 4)]):lineid,:))
    end
    fulltable = [header;label_tabel];
    cell2csv(labelfile,fulltable)
end

function [anno] = label_line()
h_obj=imline;
pos = getPosition(h_obj);
center = mean(pos,1);
diameter = sqrt(sum(diff(pos).^2));

h = gcf;
strtmp = strsplit(h.Children(8).String,' ');
id_layer = str2num(strtmp{2});

anno = [center,id_layer,diameter];
h_obj.delete()
end

function [anno] = label_rect(im)
h = gcf;
h_rect=imrect;
label_pos = round(getPosition (h_rect));
mask = createMask(h_rect);
strtmp = strsplit(h.Children(8).String,' ');
id_layer = str2num(strtmp{2});
im_layer = squeeze( im(:,:,id_layer));
patch = im_layer(label_pos(2):label_pos(2)+label_pos(4),label_pos(1):label_pos(1)+label_pos(3));
bw = patch>-600;
se = strel('disk',round(label_pos(3)/12));
bw2 = imopen(bw,se);
re = regionprops(bw2,'PixelIdxList','area','centroid');
if isempty(re)
    disp('wrong place')
    h_rect.delete()
    anno = label_rect(im);
    return
end
areas = [re.Area];
[bigarea,id_re] = max(areas);
bw3 = bw2-bw2;
bw3(re(id_re).PixelIdxList)=1;
h2 = figure(2);
imshow(bw3)
pause(1)
h2.delete();

diameter = (bigarea/pi).^0.5*2;
centroid = re(id_re).Centroid+label_pos(1:2);
anno = [centroid,id_layer,diameter];
h_rect.delete()
end

function rgbim = drawRect(rgbim,tmpannos,channel)
n_annos = size(tmpannos,1);
newim = squeeze(rgbim(:,:,:,channel));
for i_annos = 1:n_annos
    coord = tmpannos(i_annos,1:3);
    diameter = tmpannos(i_annos,4);
    layer = round(coord(3));
    zspan = 2;
    newimtmp = newim(:,:,layer-zspan:layer+zspan);
    if diameter > 40
        coeff = 1.5;
    else
        coeff= 2;
    end
    newimtmp = drawRectangleOnImg(round([coord(1:2)-diameter*coeff/2,diameter*coeff,diameter*coeff]),newimtmp);
    newim(:,:,layer-zspan:layer+zspan) = newimtmp;
end
rgbim(:,:,:,channel)=newim;
end
function rgbI = drawRectangleOnImg (box,rgbI)
x = box(2); y = box(1); w = box(4); h = box(3);
rgbI(x:x+w,y,:)   = 255;
rgbI(x:x+w,y+h,:) = 255;
rgbI(x,y:y+h,:)   = 255;
rgbI(x+w,y:y+h,:) = 255;
end
% dicom_folder_info