%%
clear;
clc;

DB_root_path = 'E:/DB/MPIIGaze.tar/MPIIGaze/Data/Normalized/'; %p00, p01.. 폴더가 위치한 디렉토리
DB_dirs = dir(DB_root_path);
DB_dirs_name = {DB_dirs(3:end).name};           % '.', '.." 제외. 중괄호가 필요
DB_dirs_num = size(DB_dirs, 1)-2;               % '.', '.." 제외

%각 p 디렉터리 안에는 day01, ... 디텍터리 존재.

    %gaze 기록을 위함
    maxlabel = zeros(DB_dirs_num, 2);
    minlabel = zeros(DB_dirs_num, 2);
    maxlabel_deg = zeros(DB_dirs_num, 2);
    minlabel_deg = zeros(DB_dirs_num, 2);

%각 pxx 디렉터리 내의 DB 접근
for num_dir=1:DB_dirs_num
    filepath = ['E:/DB/MPIIGaze.tar/MPIIGaze/Data/Normalized/' char(DB_dirs_name(num_dir)) '/'];
    
    path = dir(filepath);
    path = path(3:end);         %(1), (2)에 있는 '.'와 '..'를 빼려는 목적
    files = {path.name};       %디렉터리명들을 가져온다.
    total_num = size(files, 2);              
    %각 디렉토리 안의 dayXX.mat들을 불러들인다.
    
    fprintf(1, 'File Path Ready!\n');
    
    
    %********************************************************************************
    %하나의 디렉터리에서 하나의 Data 생성

    % 하나의 mat 파일 (ex: day01.mat)는 left, right의 왼쪽/오른쪽 눈동자 데이터로 구성
    % 각 눈동자는 gaze/image/pose 데이터로 구성

    %하나의 배열로 저장할 것이므로, 왼쪽 눈과 오른쪽 눈을 차례로 저장
    Data=[];
    Data.data = zeros(60,36,1, total_num*2);        % eye image (행 36, 열 60)
    Data.label = zeros(2, total_num*2);             % label과 headpose는 gaze와 pose을 2차원으로 변형하여 저장
    Data.headpose = zeros(2, total_num*2);
    Data.confidence = zeros(1, total_num*2);

    index = 0;
    
    

    for num_f=1:length(files)

        %filepath는 경로, files는 dayxx.mat
        readname = [filepath, files{num_f}];
        temp = load(readname);

        %한 mat 안에 들어있는 이미지 파일의 개수
        num_data = length(temp.filenames(:,1));     
        
        
        for num_i=1:num_data
            % for left (왼쪽 눈동자)
            index = index+1;
            img = temp.data.left.image(num_i, :,:);
            img = reshape(img, 36,60);              %눈을 36*60으로 만든다. imshow로도 확인가능
            img = imresize(img, [62, 62]);              %눈을 36*60으로 만든다. imshow로도 확인가능
            img = double(img);  
            img = (img / 127.5) -1;
            Data.data(:, :, 1, index) = img'; % filp the image


            %Gaze Direction (3차원)
            Lable_left = temp.data.left.gaze(num_i, :)';
            theta = asin((-1)*Lable_left(2));
            phi = atan2((-1)*Lable_left(1), (-1)*Lable_left(3));
            %Gaze Direction을 2차원화하여 Data에 넣는다.
            Data.label(:,index) = [theta; phi];

            %Head Pose (3차원)
            headpose = temp.data.left.pose(num_i, :);
            M = rodrigues(headpose);
            Zv = M(:,3);
            theta = asin(Zv(2));
            phi = atan2(Zv(1), Zv(3));
            %Head Pose를 2차원화하여 Data에 넣는다.
            Data.headpose(:,index) = [theta;phi];         

            % for right (오른쪽 눈동자)
            index = index+1;
            img = temp.data.right.image(num_i, :,:);
            img = reshape(img, 36,60);
            img = imresize(img, [62, 62]);              %눈을 36*60으로 만든다. imshow로도 확인가능
            img = double(img);  
            img = (img / 127.5) -1;
            Data.data(:, :, 1, index) = flip(img, 2)'; % filp the image

            %Gaze Direction (3차원)
            Lable_right = temp.data.right.gaze(num_i,:)';
            theta = asin((-1)*Lable_right(2));
            phi = atan2((-1)*Lable_right(1), (-1)*Lable_right(3));
            %Gaze Direction을 2차원화하여 Data에 넣는다.
            Data.label(:,index) = [theta; (-1)*phi];% flip the direction

            %Head Pose (3차원)
            headpose = temp.data.right.pose(num_i, :); 
            M = rodrigues(headpose);
            Zv = M(:,3);
            theta = asin(Zv(2));
            phi = atan2(Zv(1), Zv(3));
            %Head Pose를 2차원화하여 Data에 넣는다.
            Data.headpose(:,index) = [theta; (-1)*phi]; % flip the direction
        end
        fprintf(1, '%d / %d !\n', num_f, length(files)); 
    end
    
    %Data화 
    Data.data = Data.data/255; %normalize
    Data.data = single(Data.data); % must be single data, because caffe want float type
    Data.label = single(Data.label);
    Data.headpose = single(Data.headpose);

    savename = ['wild_', int2str(num_dir-1), '.h5'];

    %h5 파일로 저장
     hdf5write(savename,'/data', Data.data, '/label',[Data.label; Data.headpose]); 
    fprintf('done\n');

 end

