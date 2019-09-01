options.FEATSET = 'db8';
options.NLEVELS = 10;
options.FEATTYPE = 'SLFs_LBPs';


maindir = 'E:\MLdata\HPA_ieee_test_new\testdata\';
logfilename = 'log.txt';
fid = fopen(logfilename, 'w');
cnt = 0
subdir = dir( maindir );

for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        isequal( subdir( i ).name, '.DS_Store')||...
        ~subdir( i ).isdir) % skip non-dictionary
        continue;
    end 
    subsubdir = dir( [maindir subdir( i ).name '\'] );
    for j = 1 : length( subsubdir )
        if( isequal( subsubdir( j ).name, '.' )||...
            isequal( subsubdir( j ).name, '..')||...
            isequal( subsubdir( j ).name, '.DS_Store')||...
            ~subsubdir( j ).isdir)               % skip non-dictionary
            continue;
        end
        
        subsubdirpath = fullfile( maindir, subdir( i ).name, subsubdir(j).name, '*.png');
        dat = dir(subsubdirpath);
        %dat.name
        for k = 1 : length( dat )
            A = isstrprop(dat(k).name,'digit');
            if(max(strfind(dat(k).name, '.')) - max(strfind(dat(k).name, '_')) == 2 )
                datpath = fullfile( maindir, subdir( i ).name, subsubdir(j).name, dat( k ).name);
                
                
                
                readpath = datpath;
                writepath = [ maindir, subdir( i ).name,'\', subsubdir(j).name,'\', dat( k ).name(1:end-4), '_feature.txt'];
                calculateFeatures( readpath, writepath,  options.FEATSET, options.NLEVELS, options.FEATTYPE);
                cnt = cnt +  1
                
                
            end
        end
        
        
        
    end
    
end

