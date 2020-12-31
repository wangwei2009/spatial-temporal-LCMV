%% filterbank initialization
cfg.K = 512; % FFT size
cfg.N = 128; % frame shift                    % 4 x oversample decimation
cfg.Lp = 1024; % prototype filter length
D = cfg.N;                    % Decimation factor for 4x oversampling(shift)
%p=IterLSDesign(cfg.Lp,cfg.K,cfg.N);
load('filterbank/prototype_K512_N128_Lp1024.mat');

%%
close all

x = load('presteeredSignal.mat');
x = x.x;
M = size(x,2);

band_data = {M};
for m =1:M
    X=DFTAnaRealEntireSignal(x(:,m),cfg.K,cfg.N,p);
    band_data{m} = X;
end

band_num = cfg.K;
shift = cfg.N;
ya = zeros(size(band_data{1}));
for k = 1:band_num/2+1
    
    K = 7;
    N = K;
    Lh = 3;

    Wo = zeros(N*Lh,Lh);
    
    % 利用静音段估计噪声相关矩阵
    Rvv = zeros(N*Lh,N*Lh);
    data_k_t = zeros(K*Lh,1);
    
    for n = Lh:fix(4500/shift)
        for m = 1:M
            data_k_t((m-1)*Lh+1:m*Lh) = reshape(band_data{m}(k,n-Lh+1:n),[],1);
        end
        Rvv = Rvv + data_k_t*data_k_t';
    end
    Rvv = Rvv/(fix(4500/shift)-Lh+1);
    Rv1v1 = Rvv(1:Lh,1:Lh);
    
    % 语音段估计信号相关矩阵
    Ryy = zeros(N*Lh,N*Lh);
    yy = zeros(K*Lh,1);
    for n = fix(5000/shift):fix(6000/shift)
        for m = 1:M
            yy((m-1)*Lh+1:m*Lh) = reshape(band_data{m}(k,n-Lh+1:n),[],1);
        end
        Ryy = Ryy + yy*yy';
    end
    Ryy = Ryy/(fix(6000/shift)-fix(5000/shift)+1);
    Ry1y1 = Ryy(1:Lh,1:Lh);
    
    % 计算最优矩阵
    for i = 1:N
        Ryny1 = Ryy(i*Lh-Lh+1:i*Lh,1:Lh);
        Rvnv1 = Rvv(i*Lh-Lh+1:i*Lh,1:Lh);
        Wo(i*Lh-Lh+1:i*Lh,:) = (Ryny1 - Rvnv1)*inv(Ry1y1 - Rv1v1);
    end

    % 计算最优权向量
    u = zeros(1,Lh);
    u(1) = 1;
    h = inv(Rvv)*Wo*inv(Wo'*inv(Rvv)*Wo)*u';
    
    % 滤波
    x_k_t = zeros(K*Lh,1);
    for n = Lh+1:size(band_data{1},2)
        for m = 1:M
            x_k_t((m-1)*Lh+1:m*Lh) = reshape(band_data{m}(k,n-Lh+1:n),[],1);
        end
        ya(k,n) = h'*x_k_t;
    end
end

y = DFTSynRealEntireSignal(ya,cfg.K,cfg.N,p);
