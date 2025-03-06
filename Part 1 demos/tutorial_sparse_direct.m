% March 6, 2025: Per-Gunnar Martinsson, UT-Austin
%
% These codes illustrate the 1-1 relationship between invertible 
% tridiagonal matrices and semi-separable matrices. (And more generally
% between banded and quasi-separable matrices.)

function tutorial_sparse_direct

DRIVER_2D
DRIVER_3D

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function sets up a two dimensional model problem: The 5-point
% stencil on a regular grid. It illustrates sparse LU factorization using
% Matlab built-in routines.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DRIVER_2D

%%% Set problem parameters.
n2    = 11;
n1    = 31;
h     = 1/(n1-1);
L1    = 1;
L2    = h*(n2-1);
ntot  = n1*n2;

%%% Build the finite difference matrix.
D1    = spdiags(ones(n1,1)*[-1,2,-1],-1:1,n1,n1);
D2    = spdiags(ones(n2,1)*[-1,2,-1],-1:1,n2,n2);
A     = (1/(h*h))*(kron(D1,speye(n2,n2)) + ...
                   kron(speye(n1,n1),D2));

%%% Build the mesh.
[XX1,XX2] = meshgrid(linspace(0,L1,n1),linspace(0,L2,n2));
xx        = [reshape(XX1,1,ntot);...
             reshape(XX2,1,ntot)];

%%% Perform LU
fprintf(1,'ntot = %d\n',ntot)
fprintf(1,'nnz(A) = %d\n',nnz(A))
tic
[L,U,I,J] = lu(A,'vector');
t_default = toc;
fprintf(1,'Time = %0.3f seconds\n',t_default)
fprintf(1,'Default: nnz(L) = %d\n',nnz(L))
keyboard

%%% Build a few different mesh orderings and compare them.
p1 = dissect(A);
p2 = amd(A);
p3 = symrcm(A);

[L1,U1,I1,J1] = lu(A(p1,p1),'vector');
fprintf(1,'Nested dissection: nnz(L) = %d\n',nnz(L))

[L2,U2,I2,J2] = lu(A(p2,p2),'vector');
fprintf(1,'amd: nnz(L) = %d\n',nnz(L))

[L3,U3,I3,J3] = lu(A(p3,p3),'vector');
fprintf(1,'symrcm: nnz(L) = %d\n',nnz(L))

keyboard

%%% Set problem parameters.
n2    = 160;
n1    = 41;
h     = 1/(n1-1);
L1    = 1;
L2    = h*(n2-1);
ntot  = n1*n2;

%%% Build the finite difference matrix.
D1    = spdiags(ones(n1,1)*[-1,2,-1],-1:1,n1,n1);
D2    = spdiags(ones(n2,1)*[-1,2,-1],-1:1,n2,n2);
A     = (1/(h*h))*(kron(D1,speye(n2,n2)) + ...
                   kron(speye(n1,n1),D2));

%%% Build the mesh.
[XX1,XX2] = meshgrid(linspace(0,L1,n1),linspace(0,L2,n2));
xx        = [reshape(XX1,1,ntot);...
             reshape(XX2,1,ntot)];

plot(xx(1,:),xx(2,:),'k.','MarkerSize',20)
I1 = find(xx(1,:) < (0.5 - 0.5*h));
I2 = find(xx(1,:) > (0.5 + 0.5*h));
I3 = find((xx(1,:) < 0.5 + 0.5*h) & ... 
          (xx(1,:) > 0.5 - 0.5*h));
hold off
plot(xx(1,I1),xx(2,I1),'c.',...
     xx(1,I2),xx(2,I2),'b.',...
     xx(1,I3),xx(2,I3),'r.',...
     'MarkerSize',20)
axis equal
legend('I1','I2','I3')

S = full(A(I3,I3) - A(I3,I1)*(A(I1,I1)\A(I1,I3)) ...
                  - A(I3,I2)*(A(I2,I2)\A(I2,I3)));

keyboard
nhalf = round(n2/2);
semilogy(svd(S(1:nhalf,(nhalf+1):end)),'k.','MarkerSize',20)
keyboard
plot_BS_ranks(S,20,n2/20,1e-8);
keyboard
plot_HODLR_ranks(S,n2/8,3,1e-8)
keyboard
plot_HODLR_ranks(inv(S),n2/8,3,1e-8)
keyboard

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function sets up a three dimensional model problem: The 7-point
% stencil on a regular grid. It illustrates sparse LU factorization using
% Matlab built-in routines.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DRIVER_3D

%%% Set problem parameters.
n1    = 20;
n2    = 20;
n3    = 20;
h     = 1/(n2-1);
L1    = h*(n1-1);
L2    = 1;
L3    = 1;
ntot  = n1*n2*n3;

%%% Build the finite difference matrix.
D1    = spdiags(ones(n1,1)*[-1,2,-1],-1:1,n1,n1);
D2    = spdiags(ones(n2,1)*[-1,2,-1],-1:1,n2,n2);
D3    = spdiags(ones(n3,1)*[-1,2,-1],-1:1,n3,n3);
A     = (1/(h*h))*(kron(kron(D1,speye(n2,n2)),speye(n3,n3)) + ...
                   kron(kron(speye(n1,n1),D2),speye(n3,n3)) + ...
                   kron(kron(speye(n1,n1),speye(n2,n2)),D3));

%%% Build the mesh.
[XX2,XX3,XX1] = meshgrid(linspace(0,L2,n2),linspace(0,L3,n3),linspace(0,L1,n1));
xx            = [reshape(XX1,1,ntot);...
                 reshape(XX2,1,ntot);...
                 reshape(XX3,1,ntot)];
         
%%% Perform LU
fprintf(1,'ntot = %d\n',ntot)
fprintf(1,'nnz(A) = %d\n',nnz(A))
tic
[L,U,I,J] = lu(A,'vector');
t_default = toc;
fprintf(1,'Time = %0.3f seconds\n',t_default)
fprintf(1,'Default: nnz(L) = %d\n',nnz(L))

%%% Build a few different mesh orderings and compare them.
p1 = dissect(A);
p2 = amd(A);
p3 = symrcm(A);

[L1,U1,I1,J1] = lu(A(p1,p1),'vector');
fprintf(1,'Nested dissection: nnz(L) = %d\n',nnz(L))

[L2,U2,I2,J2] = lu(A(p2,p2),'vector');
fprintf(1,'amd: nnz(L) = %d\n',nnz(L))

[L3,U3,I3,J3] = lu(A(p3,p3),'vector');
fprintf(1,'symrcm: nnz(L) = %d\n',nnz(L))

keyboard

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the complement to the vector "ind" in the set (1:N).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function indc = LOCAL_complement_vector(ind,N)
indtmp = 1:N;
indtmp(ind) = 2*N*ones(1,length(ind));
indtmp = sort(indtmp);
indc = indtmp(1:(N - length(ind)));
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the key function in this program. 
% For documention on the call see the top of the file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_get_A(nside1,nside2,flag_matrix,flag_anchor)

if (nargin < 4)
  flag_anchor = 0;
end

if (nargin == 1)
  nside2 = nside1;
end

BARS  = LOCAL_get_links(nside1,nside2);
nbars = size(BARS,2);

if strcmp(flag_matrix,'orig')
    
  BARS(3,:) = ones(1,nbars);
  
elseif strcmp(flag_matrix,'rcon')
    
  BARS(3,:) = 1 + rand(1,nbars);
  
elseif strcmp(flag_matrix,'rcut')
    
  p = 0.04;
  BARS(3,:) = (rand(1,nbars) > p);
  
elseif strcmp(flag_matrix,'peri')
  
  [XX1,XX2] = meshgrid(0:(nside1-1),0:(nside2-1));
  xx        = [reshape(XX1,1,nside1*nside2);reshape(XX2,1,nside1*nside2)];
  h         = 1/(max([nside1,nside2])-1);
  Ncell     = round(min(nside1,nside2)/25);
  XX1sc     = h*xx(1,BARS(1,:));
  XX2sc     = h*xx(2,BARS(1,:));
  YY1sc     = h*xx(1,BARS(2,:));
  YY2sc     = h*xx(2,BARS(2,:));
  ZZ1sc     = 0.5*(XX1sc+YY1sc);
  ZZ2sc     = 0.5*(XX2sc+YY2sc);
  BARS(3,:) = 1 - 0.9*((cos(Ncell*pi*ZZ1sc).*cos(Ncell*pi*ZZ2sc)).^2);
    
elseif strcmp(flag_matrix,'crac')

  % We create a list of bars to be removed:
  n1a = round(0.4*nside1);
  n1b = round(0.6*nside1);
  n2a = round(0.4*nside2);
  n2b = round(0.7*nside2);
  gap = 4;
  
  ii1 = n1a*nside2 + (1:n2a);
  jj1 = ii1 + nside2;
  ii2 = n1a*nside2 + n2a + 1 + nside2*(1:(n1b-n1a));
  jj2 = ii2 - 1;
  ii3 = n1b*nside2 + 1 + (n2a:n2b);
  jj3 = ii3 + nside2;
  ii4 = (n1b-gap)*nside2 + (nside2:(-1):(n2a+gap+2));
  jj4 = ii4-nside2;
  ii5 = (n1a-1)*nside2 + (n2a+gap+1) + nside2*(1:(n1b-n1a-gap));
  jj5 = ii5+1;
  
  ii = [ii1,ii2,ii3,ii4,ii5];
  jj = [jj1,jj2,jj3,jj4,jj5];
  
  ni   = length(ii);
  BARS(3,:) = ones(1,nbars);
  BARS = [BARS,[ii;jj;-ones(1,ni)]];
      
end

if (flag_anchor == 0)
    
  A = sparse([BARS(1,:), BARS(1,:), BARS(2,:), BARS(2,:)],...
             [BARS(1,:), BARS(2,:), BARS(1,:), BARS(2,:)],...
             [BARS(3,:),-BARS(3,:),-BARS(3,:), BARS(3,:)]);
         
else
    
  ii_rim = [1:nside1,...
            (nside1+1):nside1:(nside1*(nside2-2)+1),...
            (2*nside1):nside1:(nside1*(nside2-1)),...
             nside1*(nside2-1) + (1:nside1)];
     
  A = sparse([BARS(1,:), BARS(1,:), BARS(2,:), BARS(2,:),ii_rim],...
             [BARS(1,:), BARS(2,:), BARS(1,:), BARS(2,:),ii_rim],...
             [BARS(3,:),-BARS(3,:),-BARS(3,:), BARS(3,:),0.01*ones(1,2*(nside1+nside2-2))]);
    
end

if strcmp(flag_matrix,'crac')
  [ii,jj,aa] = find(A);
  ind = find(abs(aa) < 1e-12);
  aa(ind) = zeros(size(ind));
  A = sparse(ii,jj,aa);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This supplementary functions returns a list of all links in a
% rectangular grid of size n1 x n2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function BARS = LOCAL_get_links(n1,n2)

if (nargin == 1)
  n2 = n1;
end

BARS  = zeros(3,2*n1*n2 - n1 - n2);

ndone = 0;
for i1 = 1:(n1-1)
  for i2 = 1:(n2-1)
    j_c = n2*(i1-1) + i2;
    j_n = n2*(i1-1) + i2 + 1;
    j_e = n2*i1 + i2;
    BARS(1:2,ndone + (1:2)) = [j_c, j_c; j_n, j_e];
    ndone = ndone + 2;
  end
  j_c = n2*i1;
  j_e = n2*(i1+1);
  BARS(1:2,ndone + 1) = [j_c; j_e];
  ndone = ndone + 1;
end
for i2 = 1:(n2-1)
  j_c = n2*(n1-1) + i2;
  j_n = n2*(n1-1) + i2 + 1;
  BARS(1:2,ndone + 1) = [j_c; j_n];
  ndone = ndone + 1;
end

BARS(3,:) = ones(1,ndone);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function draws the grid corresponding to the sparse matrix A.
% Node "i" has coordinates xx(:,i) in the plane.
% Two nodes i and j are connected iff A(i,j) is non-zero.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function LOCAL_draw_lattice(xx,A)

[ii,jj,aa] = find(A);

hold off
plot([xx(1,ii);xx(1,jj)],[xx(2,ii);xx(2,jj)],'k')
hold on
plot(xx(1,:),xx(2,:),'b.')
hold off

bord = 0.2;
axis([min(xx(1,:))-bord,...
      max(xx(1,:))+bord,...
      min(xx(2,:))-bord,...
      max(xx(2,:))+bord])
axis equal

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TD,TU,TL] = LOCAL_reduce(A,n2,nslab)

ntot = size(A,1);
b    = round((ntot-1)/(n2*nslab))-1;

%%% Initialize the tridiagonal matrix
TD = cell(1,nslab+1);
TU = cell(1,nslab);
TL = cell(1,nslab);

%%% Initialize the diagonal blocks of T.
for islab = 1:(nslab+1)
  I         = (islab-1)*n2*(b+1) + (1:n2);
  TD{islab} = A(I,I);
end

%Jslab = zeros(1,n2*b);
%for i = 1:b
%  Jslab(i:b:b*n2) = (i-1)*n2 + (1:n2);
%end
    
%%% Eliminate the nodes in the buffers
for islab = 1:nslab
  %%% Build the index vectors.
  IL = (islab-1)*n2*(b+1) + (1:n2);
  IC = (islab-1)*n2*(b+1) + n2 + (1:(b*n2));
  IR = (islab-1)*n2*(b+1) + (b+1)*n2 + (1:n2);
  %%% Perform the local solves.
  TMPL = A(IC,IC)\full(A(IC,IL));
  TMPR = A(IC,IC)\full(A(IC,IR));
%  TMPL          = zeros(b*n2,n2);
%  TMPR          = A(IC,IC)\full(A(IC,IR));
%  TMPL(Jslab,:) = A(IC(Jslab),IC(Jslab))\full(A(IC(Jslab),IL));
%  TMPR(Jslab,:) = A(IC(Jslab),IC(Jslab))\full(A(IC(Jslab),IR));
  %%% Form and store the T blocks.
  TD{islab}   = TD{islab  } - A(IL,IC)*TMPL;
  TU{islab}   =             - A(IL,IC)*TMPR;
  TL{islab}   =             - A(IR,IC)*TMPL;
  TD{islab+1} = TD{islab+1} - A(IR,IC)*TMPR;
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TD,TU,TL] = LOCAL_reduce_par(A,n2,nslab)

ntot = size(A,1);
b    = round((ntot-1)/(n2*nslab))-1;

%%% Initialize the tridiagonal matrix
TD  = cell(1,nslab+1);
TDX = cell(1,nslab+1);
TU  = cell(1,nslab);
TL  = cell(1,nslab);

%Jslab = zeros(1,n2*b);
%for i = 1:b
%  Jslab(i:b:b*n2) = (i-1)*n2 + (1:n2);
%end
    
%%% Eliminate the nodes in the buffers
parfor islab = 1:nslab
  %%% Build the index vectors.
  IL = (islab-1)*n2*(b+1) + (1:n2);
  IC = (islab-1)*n2*(b+1) + n2 + (1:(b*n2));
  IR = (islab-1)*n2*(b+1) + (b+1)*n2 + (1:n2);
  %%% Perform the local solves.
  TMPL = A(IC,IC)\full(A(IC,IL));
  TMPR = A(IC,IC)\full(A(IC,IR));
%  TMPL          = zeros(b*n2,n2);
%  TMPR          = A(IC,IC)\full(A(IC,IR));
%  TMPL(Jslab,:) = A(IC(Jslab),IC(Jslab))\full(A(IC(Jslab),IL));
%  TMPR(Jslab,:) = A(IC(Jslab),IC(Jslab))\full(A(IC(Jslab),IR));
  %%% Form and store the T blocks.
  TD{islab}    = - A(IL,IC)*TMPL;
  TU{islab}    = - A(IL,IC)*TMPR;
  TL{islab}    = - A(IR,IC)*TMPL;
  TDX{islab+1} = - A(IR,IC)*TMPR;
end

%%% Assemble TD.
I     = 1:n2;
TD{1} = TD{1} + A(I,I);
for islab = 2:nslab
  I         = (islab-1)*n2*(b+1) + (1:n2);
  TD{islab} = TD{islab} + A(I,I) + TDX{islab};
end
I         = nslab*n2*(b+1) + (1:n2);
TD{nslab+1} = TDX{nslab+1} + A(I,I);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [LLL,UUU] = LOCAL_LU(TD,TU,TL)

nblock = size(TD,2);

%%% Perform LU factorization of T.
tic
LLL = cell(1,nblock);
UUU = cell(1,nblock);
S   = TD{1};
%optsL.LT    = true;
%optsU.UT    = true;
for islab = 1:(nblock-1)
  [Lloc,Uloc] = lu(S);
  LLL{islab}  = Lloc;
  UUU{islab}  = Uloc;
  S           = TD{islab+1} - TL{islab}*(Uloc\(Lloc\TU{islab}));
%  [L7,U7,ind] = lu(S,'vector');
%  S           = TD{islab+1} - TL{islab}*linsolve(U7,linsolve(L7,TU{islab}(ind,:),optsL),optsU);
end
[Lloc,Uloc]  = lu(S);
LLL{nblock} = Lloc;
UUU{nblock} = Uloc;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [LLL,UUU] = LOCAL_LU_gpu(TD,TU,TL)

nblock = size(TD,2);

%%% Perform LU factorization of T.
tic
LLL = cell(1,nblock);
UUU = cell(1,nblock);
gS  = gpuArray(TD{1});
%optsL.LT    = true;
%optsU.UT    = true;
for islab = 1:(nblock-1)
  [gL,gU]     = lu(gS);
  LLL{islab}  = gather(gL);
  UUU{islab}  = gather(gU);
  gS          = gpuArray(TD{islab+1}) - gpuArray(TL{islab})*(gU\(gL\gpuArray(TU{islab})));
end
[gL,gU]  = lu(gS);
LLL{nblock} = gather(gL);
UUU{nblock} = gather(gU);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve a block tridiagonal system, given the L and U factors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vv = LOCAL_solve_LU(LLL,UUU,TL,TU,ff)

%%% Determine the number of blocks on the diagonal.
nblock = size(LLL,2);

%%% Extract block sizes.
nnloc = zeros(1,nblock);
for i = 1:nblock
  nnloc(i) = size(LLL{i},1);
end

%%% Find yy = L\gg
yy       = zeros(size(ff));
JR       = 1:nnloc(1);
yy(JR,:) = LLL{1}\ff(JR,:);
for i = 2:nblock
  JL       = sum(nnloc(1:(i-2))) + (1:nnloc(i-1));
  JR       = sum(nnloc(1:(i-1))) + (1:nnloc(i  ));
  yy(JR,:) = LLL{i}\(ff(JR,:) - TL{i-1}*(UUU{i-1}\yy(JL,:)));
end

%%% Find vv = U\yy.
vv       = zeros(size(ff));
JL       = sum(nnloc(1:(nblock-1))) + (1:nnloc(nblock));
vv(JL,:) = UUU{nblock}\yy(JL,:);
for i = (nblock-1):(-1):1
  JL       = sum(nnloc(1:(i-1))) + (1:nnloc(i  ));
  JR       = sum(nnloc(1:(i  ))) + (1:nnloc(i+1));
  vv(JL,:) = UUU{i}\(yy(JL,:) - LLL{i}\(TU{i}*vv(JR,:)));
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve a block tridiagonal system, given the L and U factors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vv = LOCAL_solve_LU_gpu(LLL,UUU,TL,TU,ff)

%%% Determine the number of blocks on the diagonal.
nblock = size(LLL,2);

%%% Extract block sizes.
nnloc = zeros(1,nblock);
for i = 1:nblock
  nnloc(i) = size(LLL{i},1);
end

gff = gpuArray(ff);
gyy = gpuArray(zeros(size(ff)));
gvv = gpuArray(zeros(size(ff)));

%%% Find yy = L\ff
JR        = 1:nnloc(1);
gL        = gpuArray(LLL{1});
gyy(JR,:) = gL\gff(JR,:);
for i = 2:nblock
  JL        = sum(nnloc(1:(i-2))) + (1:nnloc(i-1));
  JR        = sum(nnloc(1:(i-1))) + (1:nnloc(i  ));
  gL        = gpuArray(LLL{i});
  gU        = gpuArray(UUU{i-1});
  gTL       = gpuArray(TL{i-1});
  gyy(JR,:) = gL\(gff(JR,:) - gTL*(gU\gyy(JL,:)));
end

%%% Find vv = U\yy.
JL        = sum(nnloc(1:(nblock-1))) + (1:nnloc(nblock));
gU        = gpuArray(UUU{nblock});
gvv(JL,:) = gU\gyy(JL,:);
for i = (nblock-1):(-1):1
  JL        = sum(nnloc(1:(i-1))) + (1:nnloc(i  ));
  JR        = sum(nnloc(1:(i  ))) + (1:nnloc(i+1));
  gL        = gpuArray(LLL{i});
  gU        = gpuArray(UUU{i});
  gTU       = gpuArray(TU{i});
  gvv(JL,:) = gU\(gyy(JL,:) - gL\(gTU*gvv(JR,:)));
end

vv = gather(gvv);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the Chebyshev nodes on [-1,1].
% It also computes a differentiation operator.
% It is modified from a code by Nick Trefethen.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [D,x] = LOCAL_cheb(N)

if N==0
  D=0; 
  x=1; 
  return
end
x = cos(pi*(0:N)/N)'; 
c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
X = repmat(x,1,N+1);
dX = X-X';                  
D  = (c*(1./c)')./(dX+(eye(N+1)));      % off-diagonal entries
D  = D - diag(sum(D'));                 % diagonal entries

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function displays the ranks of a block separable matrix.
% The inputs are:
%    A   : the input matrix
%    b   : block size
%    n   : the number of blocks
%    tol : the cutoff for numerical rank
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_BS_ranks(A,b,n,tol)

ntot = size(A,1);

figure(1)
hold off
plot([1;1]*linspace(0,ntot,n+1),[0;ntot]*ones(1,n+1),'k',...
     [0;ntot]*ones(1,n+1),[1;1]*linspace(0,ntot,n+1),'k',...
     'LineWidth',2)
axis ij
axis equal
axis([0,ntot,0,ntot] + 5*[-1,1,-1,1])
axis off
hold on
for ibox = 1:n
  indi = (ibox-1)*b + (1:b);
  for jbox = 1:n %[1:(ibox-1),(ibox+1):n]
    indj = (jbox-1)*b + (1:b);
    text((ibox-0.5)*b,(jbox-0.5)*b,sprintf('%d',sum(svd(A(indi,indj)) > tol)))
  end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_HODLR_ranks(A,b,L,tol)

ntot = size(A,1);

figure(1)
subplot(1,1,1)
hold off
plot([0,ntot,ntot,0,0],[0,0,ntot,ntot,0],'k',...
    'LineWidth',2)
axis ij
axis equal
axis([0,ntot,0,ntot] + 5*[-1,1,-1,1])
axis off
hold on
for ilevel = 1:L
  m = b*(2^(L-ilevel));
  for ibox = 1:2^ilevel
    jbox = ibox + 1 - 2*mod(ibox+1,2);
    indi = (ibox-1)*m + (1:m);
    indj = (jbox-1)*m + (1:m);
    text((ibox - 0.5)*m,(jbox-0.5)*m,...
         sprintf('%d',sum(svd(A(indi,indj)) > tol)))
    plot((ibox - 1)*m + [0,m,m,0,0],(jbox-1)*m + [0,0,m,m,0],'k',...
         'LineWidth',2)
  end
end
for ibox = 1:2^L
  ind = (ibox-1)*b + (1:b);
  text((ibox - 0.5)*m,(ibox-0.5)*m,sprintf('%d',sum(svd(A(ind,ind)) > tol)))
end

title(sprintf('HODLR ranks (tol=%8.1e)',tol))
hold off

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
