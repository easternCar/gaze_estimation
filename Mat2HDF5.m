%%
clear;
clc;

DB_root_path = 'E:/DB/MPIIGaze.tar/MPIIGaze/Data/Normalized/'; %p00, p01.. ������ ��ġ�� ���丮
DB_dirs = dir(DB_root_path);
DB_dirs_name = {DB_dirs(3:end).name};           % '.', '.." ����. �߰�ȣ�� �ʿ�
DB_dirs_num = size(DB_dirs, 1)-2;               % '.', '.." ����

%�� p ���͸� �ȿ��� day01, ... �����͸� ����.

    %gaze ����� ����
    maxlabel = zeros(DB_dirs_num, 2);
    minlabel = zeros(DB_dirs_num, 2);
    maxlabel_deg = zeros(DB_dirs_num, 2);
    minlabel_deg = zeros(DB_dirs_num, 2);

%�� pxx ���͸� ���� DB ����
for num_dir=1:DB_dirs_num
    filepath = ['E:/DB/MPIIGaze.tar/MPIIGaze/Data/Normalized/' char(DB_dirs_name(num_dir)) '/'];
    
    path = dir(filepath);
    path = path(3:end);         %(1), (2)�� �ִ� '.'�� '..'�� ������ ����
    files = {path.name};       %���͸������ �����´�.
    total_num = size(files, 2);              
    %�� ���丮 ���� dayXX.mat���� �ҷ����δ�.
    
    fprintf(1, 'File Path Ready!\n');
    
    
    %********************************************************************************
    %�ϳ��� ���͸����� �ϳ��� Data ����

    % �ϳ��� mat ���� (ex: day01.mat)�� left, right�� ����/������ ������ �����ͷ� ����
    % �� �����ڴ� gaze/image/pose �����ͷ� ����

    %�ϳ��� �迭�� ������ ���̹Ƿ�, ���� ���� ������ ���� ���ʷ� ����
    Data=[];
    Data.data = zeros(60,36,1, total_num*2);        % eye image (�� 36, �� 60)
    Data.label = zeros(2, total_num*2);             % label�� headpose�� gaze�� pose�� 2�������� �����Ͽ� ����
    Data.headpose = zeros(2, total_num*2);
    Data.confidence = zeros(1, total_num*2);

    index = 0;
    
    

    for num_f=1:length(files)

        %filepath�� ���, files�� dayxx.mat
        readname = [filepath, files{num_f}];
        temp = load(readname);

        %�� mat �ȿ� ����ִ� �̹��� ������ ����
        num_data = length(temp.filenames(:,1));     
        
        
        for num_i=1:num_data
            % for left (���� ������)
            index = index+1;
            img = temp.data.left.image(num_i, :,:);
            img = reshape(img, 36,60);              %���� 36*60���� �����. imshow�ε� Ȯ�ΰ���
            img = imresize(img, [62, 62]);              %���� 36*60���� �����. imshow�ε� Ȯ�ΰ���
            img = double(img);  
            img = (img / 127.5) -1;
            Data.data(:, :, 1, index) = img'; % filp the image


            %Gaze Direction (3����)
            Lable_left = temp.data.left.gaze(num_i, :)';
            theta = asin((-1)*Lable_left(2));
            phi = atan2((-1)*Lable_left(1), (-1)*Lable_left(3));
            %Gaze Direction�� 2����ȭ�Ͽ� Data�� �ִ´�.
            Data.label(:,index) = [theta; phi];

            %Head Pose (3����)
            headpose = temp.data.left.pose(num_i, :);
            M = rodrigues(headpose);
            Zv = M(:,3);
            theta = asin(Zv(2));
            phi = atan2(Zv(1), Zv(3));
            %Head Pose�� 2����ȭ�Ͽ� Data�� �ִ´�.
            Data.headpose(:,index) = [theta;phi];         

            % for right (������ ������)
            index = index+1;
            img = temp.data.right.image(num_i, :,:);
            img = reshape(img, 36,60);
            img = imresize(img, [62, 62]);              %���� 36*60���� �����. imshow�ε� Ȯ�ΰ���
            img = double(img);  
            img = (img / 127.5) -1;
            Data.data(:, :, 1, index) = flip(img, 2)'; % filp the image

            %Gaze Direction (3����)
            Lable_right = temp.data.right.gaze(num_i,:)';
            theta = asin((-1)*Lable_right(2));
            phi = atan2((-1)*Lable_right(1), (-1)*Lable_right(3));
            %Gaze Direction�� 2����ȭ�Ͽ� Data�� �ִ´�.
            Data.label(:,index) = [theta; (-1)*phi];% flip the direction

            %Head Pose (3����)
            headpose = temp.data.right.pose(num_i, :); 
            M = rodrigues(headpose);
            Zv = M(:,3);
            theta = asin(Zv(2));
            phi = atan2(Zv(1), Zv(3));
            %Head Pose�� 2����ȭ�Ͽ� Data�� �ִ´�.
            Data.headpose(:,index) = [theta; (-1)*phi]; % flip the direction
        end
        fprintf(1, '%d / %d !\n', num_f, length(files)); 
    end
    
    %Dataȭ 
    Data.data = Data.data/255; %normalize
    Data.data = single(Data.data); % must be single data, because caffe want float type
    Data.label = single(Data.label);
    Data.headpose = single(Data.headpose);

    savename = ['wild_', int2str(num_dir-1), '.h5'];

    %h5 ���Ϸ� ����
     hdf5write(savename,'/data', Data.data, '/label',[Data.label; Data.headpose]); 
    fprintf('done\n');

 end

