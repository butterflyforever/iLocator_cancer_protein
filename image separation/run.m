
maindir = '/home/liyang/project/medical_image/ML_Project/iLocate_code/lib/2_separationCode/HPA_ieee_test_new/testdata/';
logfilename = 'test_log.txt';

subdir = dir( maindir );
filelist = [];
filenum = 0;

fid = fopen(logfilename, 'w');

flag = 0;

for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        isequal( subdir( i ).name, '.DS_Store')||...
        ~subdir( i ).isdir) % skip non-dictionary
        continue;
    end
   
    
    subsubdir = dir( [maindir subdir( i ).name '/'] );
    %length(subsubdir)
    %[maindir subdir(i).name '/']

    for j = 1 : length( subsubdir )
        %flag = flag + 1
        if( isequal( subsubdir( j ).name, '.' )||...
            isequal( subsubdir( j ).name, '..')||...
            isequal( subsubdir( j ).name, '.DS_Store')||...
            ~subsubdir( j ).isdir)               % skip non-dictionary
            continue;
        end
        
        %flag = flag + 1
        subsubdirpath = fullfile( maindir, subdir( i ).name, subsubdir(j).name, '*.jpg' );
        dat = dir( subsubdirpath );               % find.jpg
        
        for k = 1 : length( dat )
            datpath = fullfile( maindir, subdir( i ).name, subsubdir(j).name, dat( k ).name);
            filenum = filenum + 1;
            filelist{filenum} = datpath;
        end
    end
end

fprintf('filelist:%d \n',length(filelist));
fprintf(fid,'filelist:%d \n',length(filelist));

for k = 1 : length( filelist )
    fprintf(fid,'%s \n', filelist{k});
    readpath = filelist{k};
    writepath = [readpath(1:end-3),'png'];
    writepath_protein = [readpath(1:end-4),'_pro.png'];
    writepath_DNA = [readpath(1:end-4),'_dna.png'];
    separate(readpath,writepath,writepath_DNA,writepath_protein); 
    
    % print the process
    if (mod(k,5) == 0)
      fprintf('Have checked %d pictures, process %6.2f %% \n',k,k/length(filelist)*100);
      fprintf(fid,'Have checked %d pictures, process %6.2f %% \n',k,k/length(filelist)*100);
    end
end

fclose(fid);
