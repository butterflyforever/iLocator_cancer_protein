function separate(readpath,writepath,writepath_DNA,writepath_protein)

%J(:,:,2), J(:,:,1) --- pro,dna

%addpath /home/liyang/medical_image/ML_Project/iLocate_code;

options.UMETHOD = 'lin';
options.ITER = 5000;
options.INIT = 'truncated';
options.RSEED = 13;
options.RANK = 2;
options.STOPCONN = 40;
options.VERBOSE = 1;

bf = 'Wbasis.mat';
load( bf);
options.W = W;

%readpath = ['/home/liyang/project/medical_image/ML_Project/iLocate_code/lib/2_separationCode/tmpdata/6960_A_7_4.jpg'];
%writepath = ['/home/liyang/project/medical_image/ML_Project/iLocate_code/lib/2_separationCode/tmpdata/lbaba.png'];
processImage( readpath, writepath, options)

%I = imfinfo( writepath);
%H = imread( writepath);
%J = reconIH( imread(I.Comment), H);

%writepath_1 = ['/home/liyang/medical_image/ML_Project/iLocate_code/lib/2_separationCode/tmpdata/lbaba_1.png'];
%writepath_2 = ['/home/liyang/medical_image/ML_Project/iLocate_code/lib/2_separationCode/tmpdata/lbaba_2.png'];

%imwrite( J(:,:,2), writepath_protein, 'comment', readpath);
%imwrite( J(:,:,1), writepath_DNA, 'comment', readpath);

