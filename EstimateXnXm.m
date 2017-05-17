function [ Rxn_xm ] = EstimateXnXm( Xn,Xm,t,L)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   input:
%         v: N-by-L input matrix,each row is a observation
%   output:
%         R: N*L-by-N*L 
%
%
[N,tL] = size(Xn);
%L = length(x1);
%L = L/t;
%% construct cross-correlation matrix
%   _                                                        _
%  | Rxx(0)  conj(Rxx(1))    .....             conj(Rxx(L -1))|
%  | Rxx(1)  Rxx(0)        conj(Rxx(1))   .... conj(Rxx(L -2))|
%  |   .                    Rxx(0)        .            .      |
%  |   .                      .               .        .      |
%  | Rxx(L-1) Rxx(L -2)    .......        .....Rxx(0)         |
%  |_                                                        _|

Rxn_xm = zeros(N*L,N*L);
method  = 3;
switch method
    case 1
%% method 1

    case 2
%% method 2

    case 3
%% method 3

        xcorrR = xcorr(Xn,Xm,'bias');
        c = fliplr(xcorrR(tL-L+1:tL));
        r = xcorrR(tL:tL+L-1);
        Rx1x2 = toeplitz(c,r);
        Rxn_xm = Rx1x2';%EstimateCCM(v(i,:),v(j,:))';     若以最近的信号为参考加转置
end
end
