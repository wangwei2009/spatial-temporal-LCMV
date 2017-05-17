%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  spatial-temporal LCMV   %
%  "A Minimum Distortion Noise Reduction Algorithm With Multiple Microphones " 
%   Jingdong Chen ICASSP2008
%
%  Written by Wangwei 05/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
%clear all;
c = 340; % speed of sound at sea level
fs = 44100; t = 0:1/fs:1;
x = chirp(t,200,1,2000);
% figure,spectrogram(x,256,250,256,fs);


%% load audio
pathname = '../sound/音频采样降噪0417/01/';
% pathname = '../sound/bedroom_MIC2_noise45/';
% pathname = '../sound/num2_MIC5/';
%pathname = '../sound/两人同时发言/0度+180度/';
% pathname = '../sound/meetingroom_MIC4/';
% pathname = '../sound/num_MIC5_45/';
%  pathname = '../sound/1K/';
% pathname = '../sound/1K/LowpassFiltered/';
%pathname = '../sound/1K/SimulateDirectionalNoise/';
%pathname = '../sound/1K/SimulateDirectionalChrip/';
%pathname = '../sound/1K/SimulateRandomNoise/';
% pathname = '../sound/noise/';
% pathname = '../sound/num3_MIC5/';
[wav3,fs] = audioread([pathname,'音轨-4.wav']);
wav0 = audioread([pathname,'音轨.wav']);
wav6 = audioread([pathname,'音轨-7.wav']);
wav4 = audioread([pathname,'音轨-5.wav']);
wav1 = audioread([pathname,'音轨-2.wav']);
wav5 = audioread([pathname,'音轨-6.wav']);
wav2 = audioread([pathname,'音轨-3.wav']);

noisepath = '../sound/noise/';
noise3 = audioread([noisepath,'音轨-4.wav']);
noise0 = audioread([noisepath,'音轨.wav']);
noise6 = audioread([noisepath,'音轨-7.wav']);
noise4 = audioread([noisepath,'音轨-5.wav']);
noise1 = audioread([noisepath,'音轨-2.wav']);
noise5 = audioread([noisepath,'音轨-6.wav']);
noise2 = audioread([noisepath,'音轨-3.wav']);
noise = [noise0';noise1';noise2';noise3';noise4';noise5';noise6';];
% v = noise - mean(noise,2)*ones(1,length(noise0));



%W1o = corrcoef(wav(1,1:L),wav(1,1:L))
%% lowpass filter
% Hd = LowpassFIR_3400_4000_44100;
% filtered_wav0 = filter(Hd.Numerator,1,wav0);
% filtered_wav1 = filter(Hd.Numerator,1,wav1);
% filtered_wav2 = filter(Hd.Numerator,1,wav2);
% filtered_wav3 = filter(Hd.Numerator,1,wav3);
% filtered_wav4 = filter(Hd.Numerator,1,wav4);
% filtered_wav5 = filter(Hd.Numerator,1,wav5);
% filtered_wav6 = filter(Hd.Numerator,1,wav6);

%wav = [filtered_wav0';filtered_wav1';filtered_wav2';filtered_wav3';filtered_wav4';filtered_wav5';filtered_wav6';];
%SaveMultiAudio( '../sound/1K/LowpassFiltered/filtered_',filtered_wav,fs)
wav = [wav0';wav1';wav2';wav3';wav4';wav5';wav6';];
% wav_TimeAligned = TimeAligned(wav,3);
% wav_TimeAligned_DS = sum(wav_TimeAligned)/7;
%audiowrite([pathname,'cutwav_TimeAligned_DS3.wav'],wav_TimeAligned_DS(41000:end),44100);
y = wav;
%y = wav - mean(wav,2)*ones(1,length(wav0));
Wo = {};
%y = y - repmat(mean(y),N,1);

%% 
% delay = zeros(6,1);
% [d1, cv, pro] = delayesttm(wav0,wav1,fs);delay(1) = (d1*fs);
% [d2, cv, pro] = delayesttm(wav0,wav2,fs);delay(2) = (d2*fs);
% [d3, cv, pro] = delayesttm(wav0,wav3,fs);delay(3) = (d3*fs);
% [d4, cv, pro] = delayesttm(wav0,wav4,fs);delay(4) = (d4*fs);
% [d5, cv, pro] = delayesttm(wav0,wav5,fs);delay(5) = (d5*fs);
% [d6, cv, pro] = delayesttm(wav0,wav6,fs);delay(6) = (d6*fs);
% delay


pathname = '../sound/meetingroom_MIC4/';

N = 6;                %通道数
L = 200;              %滤波器长度
t = 200;
u = zeros(1,L);
u(1) = 1;

%% 模拟阵列接收信号,添加噪声
x = load('sigSpeech16KNoPath0.05.mat');
x = x.sigout;
x = x(1000:end,:);
x = x(1.5e4:end,:);%x = x - repmat(mean(x),size(x,1),1);
% x = randn(40000,7)/400;
fs = 16000;
SourceKind = 1;
switch SourceKind
    case 1
        N = 7;
        % %噪声1 手工添加随机噪声
%         n1 = randn(size(x))/400;
        v = n1(1:t*L,:)';
        y = (x+n1)';            %%
        SIRi = 10*log10(sum(y(:,1).^2)/sum(n1(:,1).^2))
        %audiowrite([pathname,'speechx.wav'],x(:,1),fs);
%         y(:,1:L) = v;
%         y = y(:,1.5e4:2.8e4);
    case 2
        % %噪声2 模拟阵列接收（3 3 0.5）处的随机噪声
        v0 = load('NoiseNoPath.mat');
        v0 = v0.sigout/2;
        s = v0(:,27:end);                   %信源s
        t = 27;                             %到第一个麦克风时间t = 27
        k = 400;                            %时刻 k = 400;快拍数L = 200;
        tao = 20;                           %相邻麦克风之间的延时
%         Xn(k) = [xn(k),xn(k-1) ...
%                 xn(k-L+1)]';
%         Xn(k) = [s(k-t+tao),s(k-t+tao-1) ...
%                  s(k-t+tao-L+1)]';
        x = v0(161:end,:)';
        v = randn(size(x))/400;
        y = (x+v); 
        v = v(:,1:t*L);
        x = x';
        %v = v0(160:160+L,:)';
        %v = v0(161:160+t*L,:)';
    case 3
        % % 实际录音
        y = wav(2:7,41000:end);%y = wav(:,7500:end);
%          y = wav(:,20000:end);

        %noise = randn(size(y))/400;
        v = noise;
        v = v(:,1:16000);
%         v = wav(:,end-t*L:end);
           v = wav(2:7,1:t*L);
        fs = 48000;
end
%audiowrite([pathname,'音频cut.wav'],wav(1,41000:end),fs);
%% 求噪声互相关矩阵Rvv
%Rvv = v(:,1:L)'*v(:,1:L)/(L -1);
%Rvv = EstimateCrossCorrelationMatrix(v(:,1:L),v(:,1:L));

tic

%% 估计 Rvv

%Rvv = EstimateMatrixCorrelationMatrix(v,t,L);%Rvv = eye(N*L,N*L)*1.4824e-04;%6.25e-06;
Rvv = EstimateRvv( v,t,L);
% I = eye(N*L,N*L);
% e = (std(diag(Rvv)))*2;%(std(diag(Rvv))+trace(Rvv)/L)/2;
%Rvv = Rvv+e*I;     %对角加载
%figure,plot([fliplr(Rvv(1:L,1)'),Rvv(1,1:L)])
%Rvv = eye(N*L,N*L)*6.25e-06;
% lambda = 0.9975;
% Rvv = zeros(N*L,N*L);
% hwt = waitbar(0,'estimate Rvv');
% t = 100;
% for k = 1:L:t*L
%     Rvvk_1 = Rvv;
%     VkVkT = EstimateMatrixCorrelationMatrix(v(:,k+1-1:k+L-1),1,L);
%     Rvv = lambda*Rvvk_1+(1-lambda)*VkVkT;
%     waitbar(k/(t*L),hwt);
% end
% close(hwt)
% I = eye(N*L,N*L);
% e = (std(diag(Rvv)))*5;%(std(diag(Rvv))+trace(Rvv)/L)/2;
%Rvv = Rvv+e*I;     %对角加载
%figure,plot([fliplr(Rvv(1:L,1)'),Rvv(1,1:L)])
%Rvv = eye(N*L,N*L)*6.25e-06;
toc
%% 估计Ryy
lambda = 0.98;
%Ryy = EstimateMatrixCorrelationMatrix(y(:,1:t*L),t,L);%Rvv = eye(N*L,N*L)*1.4824e-04;%6.25e-06;

% Ryy = zeros(N*L,N*L);
% hwt = waitbar(0,'estimate Ryy');
% t2 = 100;
% for k = 1:L:t2*L
%     Ryyk_1 = Ryy;
%     YkYkT = EstimateMatrixCorrelationMatrix(y(:,k+1-1:k+L-1),1,L);
%     Ryy = lambda*Ryyk_1+(1-lambda)*YkYkT;
%     waitbar(k/(t2*L),hwt);
% end
% close(hwt)


%%

% CalN = [var(wav0(1:44100));var(wav1(1:44100));var(wav2(1:44100));var(wav3(1:44100));var(wav4(1:44100));var(wav5(1:44100));var(wav6(1:44100))];
% CalN_L = repmat(CalN,[1,L])';
% RvvDiag = reshape(CalN_L,N*L,1);
% Rvv = eye(N*L,N*L);
%  Rvv(logical(eye(size(Rvv)))) = 6.25e-06;

% hwt = waitbar(0,'Estmating Rvv...');
% for i = 1:10
%     Rvv = bsxfun(@plus, EstimateMatrixCorrelationMatrix(n1(i*L-L+1:i*L,:)), Rvv);
%     %Rvv = Rvv + EstimateMatrixCorrelationMatrix(n1(i*L-L+1:i*L,:));
%     waitbar(1/10,hwt);
% end
% close(hwt)
% Rvv = Rvv/10;


%Rvv(logical(eye(size(Rvv)))) = 6.2500e-06;
%Rvv = eye(N*L,N*L)*6.2500e-06;
% for i = 1:N
%     for j = 1:N
%         Rvv(i*L-L+1:i*L,j*L-L+1:j*L) = EstimateCrossCorrelationMatrix(v(i,1:L),v(j,1:L));
%     end
% end
% for i = 2:N
%     for j = 1:i-1
%         Rvv(i*L-L+1:i*L,j*L-L+1:j*L) = Rvv(j*L-L+1:j*L,i*L-L+1:i*L);
%     end
% end
%Rvv = eye(N*L,N*L)/400/400;


%Ryy = EstimateMatrixCorrelationMatrix(y(:,1:t*L),t,L);%Rvv = eye(N*L,N*L)*1.4824e-04;%6.25e-06;
%I = eye(N*L,N*L);
%e = (std(diag(Ryy)))*2;%(std(diag(Rvv))+trace(Rvv)/L)/2;
%Ryy = Ryy+e*I;     %对角加载
%Ryy = EstimateMatrixCorrelationMatrix(y(:,1:L));
% Ryy = zeros(N*L,N*L);
% for i = 1:N
%     for j = 1:N
%         Ryy(i*L-L+1:i*L,j*L-L+1:j*L) = EstimateCrossCorrelationMatrix(y(i,1:L),y(j,1:L));
%     end
% end

R_yn_ym = {};
R_vn_vm = {};
%Rvv = eye(N*L,N*L);
%% 

%R_v1_v1 = Rvv(1*L-L+1:1*L,1*L-L+1:1*L);
%R_v1_v1(logical(eye(size(R_v1_v1)))) = 6.2500e-06;
%R_y1_y1 = EstimateCrossCorrelationMatrix(y(1,1:L),y(1,1:L));

% R_v1_v1 = EstimateCCM(v(1,:),v(1,:));
% R_y1_y1 = EstimateCCM(y(1,1:t*L),y(1,1:t*L));
%% 求各自相关矩阵 Rvv以及互相关矩阵R_yn_ym、R_vn_vm
Wo = zeros(N*L,L);
I =eye(L,L);
Rxx = zeros(L,L);
ty = 30;
m = 1;           %选择恢复的通道
R_yn_ym = {};
R_vn_vm = {};
for n = 1:N
  %R_yn_y1{n} = y(n,1:L)'*y(1,1:L)/(L -1);
  %R_vn_v1{n} = v(n,1:L)'*v(1,1:L)/(L -1);
  
  %R_yn_y1 = EstimateCrossCorrelationMatrix(y(n,1:t*L),y(1,1:t*L),t,L);
  R_yn_ym{n} = EstimateXnXm( y(n,1:ty*L),y(m,1:ty*L),ty,L);%大于0则Ym超前Yn
  %R_xn_x1{n} = EstimateCrossCorrelationMatrix(x(1:L,n)',x(1:L,1)');
  %R_vn_vm{n} = Rvv(n*L-L+1:n*L,1*L-L+1:1*L);
  R_vn_vm{n} = EstimateXnXm( v(n,1:t*L),v(m,1:t*L),t,L);
  %R_vn_v1 = R_vn_v1+(std(diag(R_vn_v1)))*2*I;
  %R_vn_v1{n} = EstimateCrossCorrelationMatrix(v(n,1:L),v(1,1:L));R_vn_v1{n} = eye(L,L)*6.2500e-06;

   %R_yn_y1 = EstimateCCM(y(n,1:t*L),y(1,1:t*L),L);
   %R_yn_y1 = Ryy(n*L-L+1:n*L,1*L-L+1:1*L);

   if n == m
       R_ym_ym = R_yn_ym{n};
       R_vm_vm = R_vn_vm{n};
   else
       %R_vn_vm{n} = zeros(L,L);
   end
   %R_vn_vm{n} = zeros(L,L);
%   Wo(n*L-L+1:n*L,:) = (R_yn_ym - R_vn_vm)/((R_ym_ym - R_vm_vm));
  %R_xn_x1{n} = (R_yn_y1 - R_vn_v1);
  %Rxx = Rxx+R_xn_x1{n}'*R_xn_x1{n};
  %Wo(n*L-L+1:n*L,:) = R_xn_x1{n}/R_xn_x1{1};
end
%R_vm_vm = eye(L,L)*6.25e-6;
%% 求滤波矩阵 W
for n = 1:N
    Rynym = R_yn_ym{n};
    Rvnvm = R_vn_vm{n};
    Wo(n*L-L+1:n*L,:) = (Rynym - Rvnvm)/((R_ym_ym - R_vm_vm));
end
%Wo(1:L,:) = eye(L,L);
%% 估计最优权矢量 h
Ho = zeros(N*L,1);
for i = 1:N
%    Ho(i*L-L+1:i*L) = R_xn_x1{i}/(Rxx)*R_xn_x1{1}*u';
end
H = (Rvv)\Wo/((Wo'/(Rvv)*Wo))*u';

figure,
subplot(3,1,1),plot(H(1*L+1:2*L));
subplot(3,1,2),plot(H(3*L+1:4*L));
subplot(3,1,3),plot(H(5*L+1:6*L));


%% 根据最优权矢量恢复期望信号 x1
%Estimate_W_H(yy,L,N,R_v1_v1,R_vn_v1);
x1 = zeros(1,length(y(1,:)));
v1 = zeros(1,length(y(1,:)));
for k = L+1:1:length(y(1,:))
    for n = 1:N
        x1(k) = x1(k) + y(n,k-L+1:k)*H(n*L-L+1:n*L);
        v1(k) = v1(k) + n1(k-L+1:k,n)'*H(n*L-L+1:n*L);
    end
end
%Wo'*H
H'*Rvv*H
SNRo = 10*log10(sum(x1.^2)/sum(v1.^2))
display('playing y1 ...');
sound(y(1,:),fs);
%audiowrite([pathname,'cut_meetingroom_mic4.wav'],y(1,:),fs);

display('playing x1 ...');
sound(x1,fs);
close
% audiowrite([pathname,'ST4_meetingroom_mic4_.wav'],x1,fs);
% ds = y(1,79:end-12)+y(2,81:end-10)+y(3,83:end-8)+y(4,85:end-6)+y(5,87:end-4)+y(6,89:end-2)+y(7,91:end);
% display('playing ds...');
% sound(ds/7,fs);
% audiowrite([pathname,'speech_ds.wav'],ds/7,fs);
%% BLOCK -BASED
% Hb = ((Wo'/(Rvv)*Wo))\Wo'/(Rvv);
% x2 = zeros(length(y(1,:)),1);
% for k = L+1:L:length(y(1,:))-L
%     for n = 1:N
%         x2(k:k+L-1) = x2(k:k+L-1) + Hb(:,n*L-L+1:n*L)'*y(n,k-L+1:k)';
%     end
% end
% sound(x2,fs);
% tic
%% 分段处理数据
tWin = 0.025;           %帧长20ms
nWin = round(tWin*fs);  % Audio window size in samples
if nWin/2 ~= fix(nWin/2)  % Ensure samples are even for overlap and add
    nWin = nWin+1;
end
nInc = round(nWin/2);  % Window increment %50 overlap


hwin = triang(nWin);  %  Tappering window for overlap and add triang、hamming
% hwin = hwin(1:end-1);  % Make adjustment so even windows align
t = 5;
x2 = zeros(1,length(y(1,:)));%
hwt = waitbar(0,'spatial-temporal LCMV');
xx = zeros(1,nWin);
Ryy = zeros(N*L,N*L);
lambda = 0.98;
for k = 1:nInc:length(y(1,:))-nWin-L
    yy = y(:,k:k+nWin-1);
        %Ryy = EstimateMatrixCorrelationMatrix(yy(:,1:t*L),t,L);%Rvv = eye(N*L,N*L)*1.4824e-04;%6.25e-06;
        for p = k:L:k+nWin-L
            for i = 1:N
                yk = y(n,p+1:p+L)';
%                 vk = v(n,p+1:p+L)';
                 R_yn_ym_k_1{n} = R_yn_ym{n};
                 R_yn_ym{n} = lambda*R_yn_ym_k_1{n}+(1-lambda)*yk*yk';
                
                R_vn_vm_k_1{n} = R_vn_vm{n};
%                R_vn_vm{n} = lambda*R_vn_vm_k_1{n}+(1-lambda)*vk*vk';
                
%                 Ryyk_1 = Ryy;
%                 YkYkT = EstimateMatrixCorrelationMatrix(y(:,p+1-1:p+L-1),1,L);
%                 Ryy = lambda*Ryyk_1+(1-lambda)*YkYkT;
            end
        end
        
    for n = 1:N
      Rynym = R_yn_ym{n};
      Rvnvm = R_vn_vm{n};
      if n == m
          R_ym_ym = R_yn_ym{n};
          R_vm_vm = R_vn_vm{n};
      else
       %R_vn_vm{n} = zeros(L,L);
      end
   
      Wo(n*L-L+1:n*L,:) = (Rynym - Rvnvm)/((R_ym_ym - R_vm_vm));
    end
    H = (Rvv)\Wo/((Wo'/(Rvv)*Wo))*u';



    %[Wo,H] = EstimateW_H(yy,L,N,R_vm_vm,R_vn_vm,R_yn_ym,Rvv,Ryy);
    for i = k:k+nWin-1
        Yk = y(:,i:i+L-1);
        Yk = reshape(Yk',1,N*L);
        xx(i-k+1) = Yk*H;
%         for n = 1:N
%             x1(i) = x1(i) + yk(n,:)*H(n*L-L+1:n*L);
%         end
    end
    x2(k:k+nWin-1) = x2(k:k+nWin-1)+xx.*hwin';
%     x1(k+inc:k+nWin-1) = x1(k+inc:k+nWin-1)/4;
    waitbar(k/(length(y(1,:))-1000),hwt);
end
toc
display('playing y1');sound(y(1,:),fs);
display('playing x2...');sound(x2,fs);
close(hwt)
%audiowrite([pathname,'testST2.wav'],x2,fs);
%%

% a = [1 3 2 5];
% b = [4 7 9 2];
% aa = EstimateCrossCorrelationMatrix( a,a );
% bb = EstimateCrossCorrelationMatrix( b,b );
% ab = EstimateCrossCorrelationMatrix( a,b );
% ba = EstimateCrossCorrelationMatrix( b,a );
% R = [aa ab;ba bb];
% rank([aa ab;ab bb])
% 
% ran = randn(2,2);
% [N,L] = size(ran);
% Rran = zeros(N*L,N*L);
% for i = 1:N
%     for j = 1:N
%         Rran(i*L-L+1:i*L,j*L-L+1:j*L) = EstimateCrossCorrelationMatrix(v(i,1:L),v(j,1:L));
%     end
% end
% for i = 2:N
%     for j = 1:i-1
%         Rran(i*L-L+1:i*L,j*L-L+1:j*L) = Rran(j*L-L+1:j*L,i*L-L+1:i*L);
%     end
% end
% rank(Rran(1:end,:))
%[U S V] = svd(Rran);

a = [1 3 2 5];
b = [4 7 9 2];
EstimateCCM(a,b,4)*4;
%R = EstimateMatrixCorrelationMatrix( [a;b],1,4)*4;
%Rab = EstimateCrossCorrelationMatrix(b,a)*length(a);

ab = [a,b];
xcorrR = xcorr(ab);
c = fliplr(xcorrR(1:8));
r = xcorrR(8:15);
Rx1x2 = toeplitz(c,r);
% Rab(logical(eye(size(Rab)))) = 99
% c = randn(63504,1);
% d = randn(63504,1);
% cxcorr = xcorr(c(1:2000),c(1:2000),'bias');figure,plot(cxcorr);
% e = xcorr(c,d,'bias');
% figure,plot(e)

% r = zeros(2*L-1,1);
% for i = -L+1:1:L-1
%     if i<=0
%         r(i+L) = c(401+i:600+i)'*c(401:600)/200;
%     else
%         r(i+L) = c(401:600)'*c(401+i:600+i)/200;
%     end  
% end
% figure,plot(r);




