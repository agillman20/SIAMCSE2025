function proxy_demo

Npan      = 200;  % The number of Gaussian panels on each smooth component.
Ngau      = 10;    % The number of Gaussian nodes on each panel.

acc       = 1e-8; % The tolerance in the HSS compression.

radius_rel = 1.75;
nproxy     = 50;

%%% Set up the geometry and the quadrature weights
[ww,C] = construct_geom(Npan,Ngau);
Ntot = size(C,2);

ind = 1751:Ntot;

indoffd = 1:1750;

Aall = construct_A_offd(C,ww,ind,indoffd);

[Cproxy,U,k,J,indnear,indfar] = create_proxy_factor(C,ww,ind,radius_rel,indoffd,nproxy,acc);

err = norm(Aall- U*Aall(J(1:k),:));

fprintf(1,'err = %12.5e\n', err)

% create plot
plot(C(1,ind),C(4,ind),'r-',C(1,indnear),C(4,indnear),'b.',C(1,indfar),C(4,indfar),'k-','LineWidth',4)
hold on
plot(Cproxy(1,:),Cproxy(4,:),'bx','LineWidth',4)
hold on
plot(C(1,ind(J(1:k))),C(4,ind(J(1:k))),'cx','LineWidth',2)

return
end

function [Cproxy,U,k,J,indnear,indfar] = create_proxy_factor(C,ww,ind,radius_rel,indoffd,nproxy,acc)


[indnear,indfar,xxc,R] = find_nearfar(C,ind,radius_rel,indoffd);

Cproxy  = construct_circle(xxc, radius_rel*R, nproxy);

A12proxy = [construct_A_offd(C,ww,ind,indnear),...
    construct_A_cont_prox(C,ind,Cproxy)];

f = 1.2;
[U,J] = ID(A12proxy,'tol', acc, f);
k = size(U,2);




return
end

function [indnear,indfar,xxc,R] = find_nearfar(C,ind,radius_rel,indoffd)

[xxc, R] = get_circum_circle(C(:,ind));

xc = xxc(1);
yc = xxc(2);
relind =  ((C(1,indoffd)-xc).^2 + ...
    (C(4,indoffd)-yc).^2) < ((radius_rel*R)^2) ;
indnear = indoffd(relind);

relind =  ((C(1,indoffd)-xc).^2 + ...
    (C(4,indoffd)-yc).^2) >= ((radius_rel*R)^2) ;
indfar = indoffd(relind);

return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ww,C] = construct_geom(Npan,Ngau)

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

function [xxc, R] = get_circum_circle(C)

nloc       = size(C,2);

% First we TRY to create the circle based on the endpoints.
xxc        = 0.5*(C([1,4],1) + C([1,4],end));
distsquare = (C(1,:)-xxc(1)).^2 + (C(4,:)-xxc(2)).^2;
R          = sqrt(max(distsquare));

% The result is an absurdly large circle, then we instead
% base it on the center of mass of the points.
if ( (1.2*R*R) > ( (C(1,1) - C(1,end))^2 + (C(4,1) - C(4,end))^2) )
    xxc        = (1/nloc)*[sum(C(1,:)); sum(C(4,:))];
    distsquare = (C(1,:)-xxc(1)).^2 + (C(4,:)-xxc(2)).^2;
    R          = sqrt(max(distsquare));
end

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C = construct_circle(xxc, R, n)

tt = linspace(0, 2*pi*(1-1/n), n);
C  = [xxc(1) + R*cos(tt);...
    - R*sin(tt);...
    - R*cos(tt);...
    xxc(2) + R*sin(tt);...
    + R*cos(tt);...
    - R*sin(tt)];

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function B = construct_A_offd(C,ww,ind1,ind2)

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function B = construct_A_cont_prox(C,ind,Cprox)

n1   = length(ind);
n2   = size(Cprox,2);
dd1  = C(1,ind)' * ones(1,n2) - ones(n1,1) * Cprox(1,:);
dd2  = C(4,ind)' * ones(1,n2) - ones(n1,1) * Cprox(4,:);
ddsq = dd1.*dd1 + dd2.*dd2;
B    = -(1/(4*pi))*log(ddsq);

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