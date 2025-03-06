% March 6, 2025: Per-Gunnar Martinsson, UT-Austin
%
% These codes illustrate the 1-1 relationship between invertible 
% tridiagonal matrices and semi-separable matrices. (And more generally
% between banded and quasi-separable matrices.)

function tutorial_tridiagonal

%%% Set up the problem size. We think of the matrix as split into b x b
%%% blocks, and these blocks organized into a tree structure. This is not
%%% relevant to the key point about tridiagonal matrices, but is helpful in
%%% plotting the rank structure.
b    = 10;   
L    = 3;
n    = 2^L;
ntot = n*b;    % The matrix size.
tol  = 1e-12;  % Tolerance for 


%%% Investigate the inverse of a tridiagonal matrix.
A = triu(tril(randn(ntot),1),-1);
A = 2*norm(A)*eye(ntot) + A;  % (Make the matrix well-conditioned.)
B = inv(A);
plot_HODLR_ranks(A,b,L,tol)
keyboard
plot_HODLR_ranks(B,b,L,tol)
keyboard
%plot_HBS_ranks_v2(A,b,L,tol);
%plot_HBS_ranks_v2(B,b,L,tol);

%%% Investigate the inverse of a semi-separable matrix.
%%% (Note that here, the matrix may end up being quite ill-conditioned, 
%%% so spurious singular modes may arise.)
A = build_semisep(ntot);
B = inv(A);
plot_HODLR_ranks(A,b,L,tol)
keyboard
plot_HODLR_ranks(B,b,L,tol)
keyboard
%plot_HBS_ranks_v2(A,b,L,tol);
%plot_HBS_ranks_v2(B,b,L,tol);

%%% Now extend to a *banded* matrix.
k = 3; % The bandwidth
A = triu(tril(randn(ntot),k),-k);
A = 2*norm(A)*eye(ntot) + A;  % (Make the matrix well-conditioned.)
B = inv(A);
plot_HODLR_ranks(A,b,L,tol)
keyboard
plot_HODLR_ranks(B,b,L,tol)
keyboard
%plot_HBS_ranks_v2(A,b,L,tol);
%plot_HBS_ranks_v2(B,b,L,tol);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = build_semisep(n)

u = randn(n,1);
v = randn(n,1);
w = randn(n,1);
A = triu(u*v');
A(2:n,1) = w(2:end);
for i = 2:(n-1)
  A((i+1):n,i) = (A(i,i)/A(i,1))*A((i+1):n,1);
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

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the numerical ranks of an HBS matrix.
% The inputs are:
%    A   : the input matrix
%    b   : block size
%    L   : the number of levels in the tree
%    tol : the cutoff for numerical rank
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_HBS_ranks(A,b,L,tol)

ntot = size(A,1);

for ilevel = 1:L
  figure(1)
  subplot(1,L,ilevel)
  m = b*(2^(L-ilevel));
  hold off
  plot([0,ntot,ntot,0,0],[0,0,ntot,ntot,0],'k',...
       [0;ntot]*ones(1,2^ilevel+1),[1;1]*linspace(0,ntot,2^ilevel+1),'k',...
      'LineWidth',2)
  axis ij
  axis equal
  axis([0,ntot,0,ntot] + 5*[-1,1,-1,1])
  axis off
  hold on
  for ibox = 1:2^ilevel
    jbox = ibox + 1 - 2*mod(ibox+1,2);
    indi = (ibox-1)*m + (1:m);
    indj = [1:((ibox-1)*m),(ibox*m+1):ntot];
    text((ibox - 0.5)*m,(jbox-0.5)*m,...
         sprintf('%d',sum(svd(A(indi,indj)) > tol)))
    plot((ibox-1)*m + [0,m,m,0,0],(ibox-1)*m + [0,0,m,m,0],'k',...
         'LineWidth',2)
  end
  title(sprintf('HBS ranks - level %d',ilevel))
end
keyboard
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_HBS_ranks_v2(A,b,L,tol)

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
    indj = [1:((ibox-1)*m),(ibox*m+1):ntot];
    text((jbox - 0.5)*m,(ibox-0.5)*m,...
         sprintf('%d',sum(svd(A(indi,indj)) > tol)))
    plot((ibox - 1)*m + [0,m,m,0,0],(jbox-1)*m + [0,0,m,m,0],'k',...
         'LineWidth',2)
  end
end
for ibox = 1:2^L
  ind = (ibox-1)*b + (1:b);
  text((ibox - 0.5)*m,(ibox-0.5)*m,sprintf('%d',sum(svd(A(ind,ind)) > tol)))
end
title(sprintf('HBS ranks (tol=%8.1e)',tol))

keyboard
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

