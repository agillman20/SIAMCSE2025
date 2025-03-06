%%  chunkIE demo 1:
%  This demo illustrates the solution to an interior Laplace Dirichlet 
%  problem 

%% Step 1: define geometry
n = 10; modes = zeros(2*n + 1,1); 
modes(1) = 1; modes(2*n+1) = 0.3; % set params for function handle
chnkr = chunkerfunc(@(t) chnk.curves.bymode(t, modes)); % build chunker
plot(chnkr, 'k.'); hold on;

%% Step 2: Identify integral representation: u = -2*D

%% Step 3: Specify kernels for integral representation
K = kernel('laplace', 'd');

%% Step 4: Discretize
A = chunkermat(chnkr, K);  
A = A - 0.5*eye(size(A,1));

%% Setup boundary data
S = kernel('laplace', 's');
nsrc = 100;
rt = rand(1,nsrc)*2 + 2;
tt = rand(1,nsrc)*2*pi;
src = [rt.*cos(tt); rt.*sin(tt)];
srcinfo = []; srcinfo.r = src;
charges = rand(nsrc,1)-0.5;
targinfo = []; targinfo.r = chnkr.r(:,:);
rhs = S.eval(srcinfo, targinfo)*charges;

%% Step 5: Solve
soln = A \ rhs;

x1 = linspace(-1.5,1.5,300);
[xx,yy] = meshgrid(x1,x1);

opts = [];
opts.flam = true;
start = tic; in = chunkerinterior(chnkr,{x1,x1},opts); t4 = toc(start);

targs = [xx(:).'; yy(:).'];
u = nan(length(xx(:)),1);

%% Step 6: Postprocess
upot = chunkerkerneval(chnkr, K, soln, targs(:,in)); 
u(in) = upot;
%% Plot the solution
figure(1)
clf()
pcolor(xx, yy, reshape(u, size(xx))); shading interp; colorbar(); hold on;
plot(chnkr, 'k.'); hold on; 
plot(src(1,:), src(2,:), 'rx')
xlim([-4,4]); ylim([-4,4])

targinfo = [];
targinfo.r = targs;
uex = S.eval(srcinfo,targinfo)*charges;
errs = nan(size(u));
errs(in) = log10(abs(u(in)-uex(in)));

figure(2)
clf();
pcolor(xx, yy, reshape(errs, size(xx))); shading interp; colorbar(); 
clim([-16,0])
hold on;
plot(chnkr, 'k.')
