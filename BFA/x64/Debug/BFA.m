% ==================================================================
% Beamforming Algorithm
% Back-Projection Algorithm
% 20170308 Zisheng WANG
%
% This matlab script generate a sar image from electronmagnetic scatters
% data by bfa, or more geranelly bp algorithm.
% The algorithm is based on CUDA 8.0, please make sure that your GPU is
% CUDA 8.0 applicability. 
% NOTE, this file will generate the bfa_input.txt in order to change
% the setup of bfa algorithm.If you want to change it by yourself, you can
% just edit the bfa_input.txt. If you change the bfa_input.txt result in
% BFA will not work properly, just run this M-Script and it will generate a
% new one.
% ==================================================================
% The length of the imaging scene, note that, in the m-script, 
% we regard the imaging scene as a square (i.e. length/2 x length/2)
% But, the BFA support rectangle, please change the bfa_input.txt directly.
Length = 10;
% please note that, the resolution doesn't mean that BP will have higher
% resolution than physical resolution. But a higher resolution in there
% will make the image have more pixels which means more beautiful.
% Resolution equals to the physical one or half of the physical is
% recommended. 
Resolution = 0.05;
% This is the name of the es file. Please note that, this file must follow
% the file type of W. Yang's SBR software. If the ES file is not in the
% same directory as the m-script, please write the full path of the Es
% file. 
EsName = 'mono_totRCStotES_ship4_scaled_Center90_Thi60.txt';
% the is the fft point scale factor. for range compression. for example, if
% we have 200 samples in frequency domain, a FFT_POINT_SCALE=20 will make
% the bp to process a 200X20 point FFT. NOTE THAT, this scale will not
% change the physical resolution of SAR image, but it will make the image
% more smooth. 20 is recommended.
FFT_POINT_SCALE = 20;
% This the Polarization of the Es file, please chose as your target.
Polarization = 'HH';
% As the name indicated
DistanceFromAPC2SceneCenter = 20000;

%% Setup Cuda
data=load(EsName);
getlength = data(1,1);
for ii = 2:length(data(:,1))
    if getlength == data(ii,1)
        break;
    end
end
nSampling_f = ii-1;
nSampling_phi = length(data(:,1))/nSampling_f;

delete('bfa_input.txt');
fid = fopen('bfa_input.txt','wt');
fprintf(fid, '================================================================\r\n');
fprintf(fid, 'This is the input of BFA.exe (Back-Projection Imaging Algorithm)\r\n');
fprintf(fid, 'When user uses the BFA.exe, all parameters should be set in this\r\n');
fprintf(fid, 'File. \r\n');
fprintf(fid, 'Note! NO LINE SHOULD BE DELETED IN THIS FILE!!!!\r\n');
fprintf(fid, 'Note! THE NAME OF THIS FILE SHOULD NOT BE CHANGED!!!!\r\n');
fprintf(fid, '================================================================\r\n');
fprintf(fid, '----------------------------------------------------------------\r\n');
fprintf(fid, '##1. The name of Es file (i.e. ship.txt)\r\n');
fprintf(fid, [EsName '\r\n']);
fprintf(fid, '----------------------------------------------------------------\r\n');
fprintf(fid, ['##2. The X range of the imaging range. (min max resolution) (meter).' ... 
                'The resolution is basiclly the increment of the range. \r\n']);
fprintf(fid, [num2str(-Length/2) ' ' num2str(Length/2) ' ' num2str(Resolution) '\r\n']);
fprintf(fid, '----------------------------------------------------------------\r\n');
fprintf(fid, '##3. The Y range of the imaging range. (min max resolution) (meter).\r\n');
fprintf(fid, [num2str(-Length/2) ' ' num2str(Length/2) ' ' num2str(Resolution) '\r\n']);
fprintf(fid, '----------------------------------------------------------------\r\n');
fprintf(fid, '##4. The size of Es file. (NumofFre NumOfPhi) \r\n');
fprintf(fid, [num2str(nSampling_f) ' ' num2str(nSampling_phi) '\r\n']);
fprintf(fid, '----------------------------------------------------------------\r\n');
fprintf(fid, '##5. Chose the polar. in (VV/HH/VH/HV/)\r\n');
fprintf(fid, [Polarization '\r\n']);
fprintf(fid, '----------------------------------------------------------------\r\n');
fprintf(fid, '##6. Set the distance between the APC and scene center\r\n');
fprintf(fid, [num2str(DistanceFromAPC2SceneCenter) '\r\n']);
fprintf(fid, '----------------------------------------------------------------\r\n');
fprintf(fid, ['##7. Set the points of fouries for range compression. i.e. if the' ...  
              'point of frequency is 200, then a 3000 point fft will be processed'... 
              'to compress the signal. 15 times is recommended.Must be EVEN\r\n']);
fprintf(fid, [num2str(FFT_POINT_SCALE) '\r\n']);
fprintf(fid, '----------------------------------------------------------------\r\n');
fclose(fid);


%% Run Cuda
command = ['BFA'];
tic
system(command);
toc

%% Show Cuda Result
load output.txt;
GeoNumber = Length/Resolution;
Geo = zeros(GeoNumber,GeoNumber);
for ii = 1:1:GeoNumber
    for jj = 1:1:GeoNumber
        Geo(ii,jj) = output(jj + (ii - 1) * GeoNumber, 1) + output(jj + (ii - 1) * GeoNumber, 2)*1i;
    end
end

figure;
image = 20*log10(abs(Geo));
image = image - max(max(image));
imageNew = image;
for ii = 1:1:length(image(:,1))
    imageNew(:,length(image(:,1))-ii + 1) = image(:,ii);
end
X_Geo = -Length/2:Resolution:Length/2;
imagesc(X_Geo,X_Geo,imageNew'); caxis([-35 0])

% end

