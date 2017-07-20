close all

x = load('presteeredSignal.mat');
x = x.x;


K = 7;
N = K;
Lh = 64;

Wo = zeros(N*Lh,Lh);
Rvv = zeros(N*Lh,N*Lh);

%% 利用静音段估计噪声相关矩阵
for i = Lh:4500
    n = reshape(x(i-Lh+1:i,:),[],1);
    Rvv = Rvv + n*n';
end
for i = 52000:length(x(:,1))
    n = reshape(x(i-Lh+1:i,:),[],1);
    Rvv = Rvv + n*n';
end
Rvv = Rvv/(12000/Lh);
Rv1v1 = Rvv(1:Lh,1:Lh);

%% 语音段估计信号相关矩阵
Ryy = zeros(N*Lh,N*Lh);
for i = 5000:6000
    yy = reshape(x(i-Lh+1:i,:),[],1);
    Ryy = Ryy + yy*yy';
end
Ry1y1 = Ryy(1:Lh,1:Lh);

%% 计算最优矩阵
for i = 1:N
    Ryny1 = Ryy(i*Lh-Lh+1:i*Lh,1:Lh);
    Rvnv1 = Rvv(i*Lh-Lh+1:i*Lh,1:Lh);
    Wo(i*Lh-Lh+1:i*Lh,:) = (Ryny1 - Rvnv1)*inv(Ry1y1 - Rv1v1);
end

%% 计算最优权向量
u = zeros(1,Lh);
u(1) = 1;
h = inv(Rvv)*Wo*inv(Wo'*inv(Rvv)*Wo)*u';

%% 滤波
% hwt = waitbar(0,'please wait...');
ya = zeros(1,length(x(:,1)));
tic
for i = Lh+1:length(x(:,1))
    xx = reshape(x(i-Lh+1:i,:),[],1);
    ya(i) = h'*xx;
%     waitbar(i/length(x(:,1)));
end
toc
% close(hwt);

writeFilePath = 'sound/';
% audiowrite([writeFilePath,'滤波信号Lh128_2.wav'],ya,fs);











