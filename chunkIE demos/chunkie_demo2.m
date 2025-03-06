%% Step 1: define geometry
nv = 7; zv = exp(1j*(1:nv)/nv*2*pi); verts = [real(zv); imag(zv)]; % define vertices
einfo = zeros(2,nv); einfo(1,:) = 1:nv; einfo(2,:) = circshift(1:nv,-1); % define edges

% Ask audience input for curves
amp = 0.5; freq = 6; fcurve = @(t) sinearc(t,amp,freq); % define curves


cparams = []; zk = 30; cparams.maxchunklen = 2*pi/zk; % curve parameters
cgrph = chunkgraph(verts, einfo, fcurve, cparams); % build chunkgraph
plot_regions(cgrph);


% Ask audience input for type of boundary conditions

idir = [1 4];
ineu = [2 7];
iimp = setdiff(1:nv, [idir, ineu]);

Sk = kernel('helm', 's', zk);
Dk = kernel('helm', 'd', zk);
Skp = kernel('helm', 'sp', zk);
Dkp = kernel('helm', 'dp', zk);

eta = 3; % impedance strength
K(nv, nv) = kernel();
K(idir, idir) = 2*Dk;
K(ineu, idir) = 2*Dkp;
K(iimp, idir) = 2*(Dkp + 1j*eta*Dk);

K(idir, [ineu, iimp]) = -2*Sk;
K(ineu, [ineu, iimp]) = -2*Skp;
K(iimp, [ineu, iimp]) = -2*(Skp + 1j*eta*Sk);

A = chunkermat(cgrph, K); A = A + eye(cgrph.npt);

%% Generate boundary data for test solution
rhs = zeros(cgrph.npt, 1);

npts = horzcat(cgrph.echnks(:).npt);
nn = [1 cumsum(npts)+1];
iinds = cell(nv,1);
for i=1:nv
    iinds{i} = nn(i):(nn(i+1)-1);
end

idir_ind = horzcat(iinds{idir});
ineu_ind = horzcat(iinds{ineu});
iimp_ind = horzcat(iinds{iimp});

src = [0.04; 0.31];
srcinfo = []; srcinfo.r = src;
targinfo = []; targinfo.r = cgrph.r(:,:); targinfo.n = cgrph.n(:,:);
pot = Sk.eval(srcinfo, targinfo);
dudn = Skp.eval(srcinfo, targinfo);

rhs(idir_ind) = pot(idir_ind);
rhs(ineu_ind) = dudn(ineu_ind);
rhs(iimp_ind) = dudn(iimp_ind) + 1j*eta*pot(iimp_ind);


soln = A \rhs;
%% Postprocess
% test solution at interior points;
Keval(1,nv) = kernel();
Keval(1,idir) = 2*Dk;
Keval(1,[ineu, iimp]) = -2*Sk;

x1 = linspace(-4,4,300);
[xx,yy] = meshgrid(x1,x1);

nt = length(xx(:));
opts = [];
opts.flam = true;
start = tic; in = chunkerinterior(cgrph,{x1,x1},opts); t4 = toc(start);

out = ~in;
targs = [xx(:).'; yy(:).'];
u = nan(nt,1);

upot = chunkerkerneval(cgrph, Keval, soln, targs(:,out)); 
u(out) = upot;

%% Test accuracy
targinfo = []; targinfo.r = targs(:,:);
uex = nan(nt,1);
utmp = Sk.eval(srcinfo, targinfo);
uex(out) = utmp(out);

errs = nan(nt, 1);
errs(out) = log10(abs(uex(out) - u(out)));

pcolor(xx, yy, reshape(errs, size(xx))); shading interp; hold on;
plot(cgrph, 'k.')

%% Compute plane wave scattering
d = pi/3;
x = cgrph.r(1,:); x = x(:);
y = cgrph.r(2,:); y = y(:);
rnx = cgrph.n(1,:); rnx = rnx(:);
rny = cgrph.n(2,:); rny = rny(:);
pot = -exp(1j*zk*(x*cos(d) + y*sin(d)));
dudn = -1j*zk*(cos(d)*rnx + sin(d)*rny).*exp(1j*zk*(x*cos(d) + y*sin(d)));


rhs(idir_ind) = pot(idir_ind);
rhs(ineu_ind) = dudn(ineu_ind);
rhs(iimp_ind) = dudn(iimp_ind) + 1j*eta*pot(iimp_ind);


soln = A \ rhs;

u = nan(nt,1);

upot = chunkerkerneval(cgrph, Keval, soln, targs(:,out)); 
u(out) = upot;

xt = targs(1,:); xt = xt(:);
yt = targs(2,:); yt = yt(:);
uin = exp(1j*zk*(cos(d)*xt + sin(d)*yt));

utot = uin + u;
utot(in) = nan;

%%
pcolor(xx, yy, reshape(real(utot), size(xx))); shading interp; hold on;
clim([-2,2]);
plot(cgrph, 'k.')







function [r,d,d2] = sinearc(t,amp,frq)
    xs = t;
    ys = amp*sin(frq*t);
    xp = ones(size(t));
    yp = amp*frq*cos(frq*t);
    xpp = zeros(size(t));
    ypp = -frq*frq*amp*sin(t);
    
    r = [(xs(:)).'; (ys(:)).'];
    d = [(xp(:)).'; (yp(:)).'];
    d2 = [(xpp(:)).'; (ypp(:)).'];
end
