function ID_test

Npan      = 200;  % The number of Gaussian panels on each smooth component.
Ngau      = 10;    % The number of Gaussian nodes on each panel.

acc       = 1e-8; % The tolerance in the HSS compression.


[ww,C,tt,t_sta,t_end] = construct_geom(Npan,Ngau);
Ntot = size(C,2);

ind = 1751:Ntot;

indoffd = 1:1750;

Aall = LOCAL_construct_A_offd(C,ww,ind,indoffd);


[U,J,k] = constructID(Aall,acc);

err = norm(Aall-U*Aall(J(1:k),:));

fprintf(1,'err = %12.5e\n', err)

plot(C(1,ind),C(4,ind),'r-','LineWidth',4)
hold on
plot(C(1,indoffd),C(4,indoffd),'b-','LineWidth',4)
plot(C(1,ind(J(1:k))),C(4,ind(J(1:k))),'cx','LineWidth',2)

return
end

function [U,J,k] = constructID(Amat,acc)

f = 1.2;
[U,J] = ID(Amat,'tol', acc, f);
k = size(U,2);

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ww,C,tt,t_sta,t_end] = construct_geom(Npan,Ngau)

t_sta         = 0;    % Start point of the discretization.
t_end         = 1; % End   point of the discretization.
tt_panels     = linspace(t_sta,t_end,Npan+1);
[tt,ww_gauss] = LOCAL_get_gauss_nodes(tt_panels,Ngau);

C             = zeros(6,length(tt));
C(1,:)        = (1+0.3*cos(5*2*pi*tt)).*cos(2*pi*tt);
C(2,:)        = 2*pi*(-sin(2*pi*tt).*(0.3*cos(5*2*pi*tt)+1)-1.5*sin(5*2*pi*tt).*cos(2*pi*tt));

C(3,:)        =  (2*pi)^2*(3*sin(2*pi*tt).*sin(5*2*pi*tt)+cos(2*pi*tt).*(-7.8*cos(5*2*pi*tt)-1));
C(4,:)        =  sin(2*pi*tt).*(1+0.3*cos(5*2*pi*tt));
C(5,:)        =  2*pi*(-1.5*sin(2*pi*tt).*sin(5*2*pi*tt)+0.3*cos(5*2*pi*tt).*cos(2*pi*tt)+cos(2*pi*tt));
C(6,:)        = (2*pi)^2*(sin(2*pi*tt).*(-7.8*cos(5*2*pi*tt)-1)-3*sin(5*2*pi*tt).*cos(2*pi*tt));

ww            = ww_gauss.*sqrt(C(2,:).*C(2,:) + C(5,:).*C(5,:));
return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function B = LOCAL_construct_A_offd(C,ww,ind1,ind2)

n1   = length(ind1);
n2   = length(ind2);
dd1  = C(1,ind1)' * ones(1,n2) - ones(n1,1) * C(1,ind2);
dd2  = C(4,ind1)' * ones(1,n2) - ones(n1,1) * C(4,ind2);
ddsq = dd1.*dd1 + dd2.*dd2;
nn1  = ones(n1,1)*( C(5,ind2)./sqrt(C(2,ind2).*C(2,ind2) + C(5,ind2).*C(5,ind2)));
nn2  = ones(n1,1)*(-C(2,ind2)./sqrt(C(2,ind2).*C(2,ind2) + C(5,ind2).*C(5,ind2)));
B    = -(1/(2*pi))*(nn1.*dd1 + nn2.*dd2)./ddsq;
B    = B.*(ones(n1,1)*ww(ind2));

return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [tt,ww] = LOCAL_get_gauss_nodes(tt_panels,Ngau)

npanels = length(tt_panels)-1;
tt   = zeros(1,npanels*Ngau);
ww   = zeros(1,npanels*Ngau);
[t_ref,w_ref] = LOCAL_lgwt(Ngau,0,1);
t_ref         = t_ref(end:(-1):1);
w_ref         = w_ref(end:(-1):1);
for i = 1:npanels
    h       = tt_panels(i+1) - tt_panels(i);
    ind     = (i-1)*Ngau + (1:Ngau);
    tt(ind) = tt_panels(i) + h*t_ref;
    ww(ind) = h*w_ref;
end

return
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,w] = LOCAL_lgwt(N,a,b)

% lgwt.m
%
% This script is for computiNgau definite integrals usiNgau Legendre-Gauss
% Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
% [a,b] with truncation order N
%
% Suppose you have a continuous function f(x) which is defined on [a,b]
% which you can evaluate at any x in [a,b]. Simply evaluate it at all of
% the values contained in the x vector to obtain a vector f. Then compute
% the definite integral usiNgau sum(f.*w);
%
% Written by Greg von Winckel - 02/25/2004
N=N-1;
N1=N+1; N2=N+2;

xu=linspace(-1,1,N1)';

% Initial guess
y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);

% Legendre-Gauss Vandermonde Matrix
L=zeros(N1,N2);

% Derivative of LGVM
Lp=zeros(N1,N2);

% Compute the zeros of the N+1 Legendre Polynomial
% usiNgau the recursion relation and the Newton-Raphson method

y0=2;

% Iterate until new points are uniformly within epsilon of old points
while max(abs(y-y0))>eps


    L(:,1)=1;
    Lp(:,1)=0;

    L(:,2)=y;
    Lp(:,2)=1;

    for k=2:N1
        L(:,k+1)=( (2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1) )/k;
    end

    Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);

    y0=y;
    y=y0-L(:,N2)./Lp;

end

% Linear map from[-1,1] to [a,b]
x=(a*(1-y)+b*(1+y))/2;

% Compute the weights
w=(b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2;

return
end
