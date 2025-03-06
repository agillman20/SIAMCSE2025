% March 6, 2025: Per-Gunnar Martinsson, UT-Austin
%
% These codes illustrate some properties of different types of rank
% structured matrices.

function tutorial_rank_structure

%%% Set up the problem size. We think of the matrix as split into b x b
%%% blocks, and these blocks organized into a tree structure. The tree has
%%% "L" levels. 
b    = 20;
L    = 3;
n    = 2^L;
ntot = n*b;

%%% Test compressibility of a Hilbert matrix.
A = hilb(ntot);
A = 2*norm(A)*eye(ntot) + A;
tol = 1e-12;
plot_HODLR_ranks(A,b,L,tol);
%keyboard
plot_HODLR_ranks(inv(A),b,L,tol);
%keyboard

%%% Tests for pretend BIE
[A,xx] = pretend_BIE(ntot);
tol = 1e-6;
plot(xx(1,:),xx(2,:),'k.'); axis equal;
%keyboard
plot_HODLR_ranks(A,b,L,tol);
%keyboard
plot_HODLR_ranks(inv(A),b,L,tol);
%keyboard
plot_BS_ranks(A,b,n,tol);
keyboard
% Let us look at a specific block:
I = 1:30;
J = 40:60;
plot(xx(1,:),xx(2,:),'k.',...
     xx(1,I),xx(2,I),'r.',...
     xx(1,J),xx(2,J),'b.',...
     'MarkerSize',20); 
axis equal;
%keyboard
semilogy(svd(A(I,J)),'k.','MarkerSize',15)
%keyboard
% The ordering of the points matter greatly!
% If we draw indices at random, then nothing works.
[~,ind] = sort(rand(1,ntot));
I = ind(1:30);
J = ind(40:60);
plot(xx(1,:),xx(2,:),'k.',...
     xx(1,I),xx(2,I),'r.',...
     xx(1,J),xx(2,J),'b.',...
     'MarkerSize',20); 
axis equal;
%keyboard
semilogy(svd(A(I,J)),'k.','MarkerSize',15)
%keyboard

%%% Tests for HODLR matrix.
%%% Let us build a random HODLR matrix, and test some things.
k = 3; % The rank.
A = build_HODLR_matrix(b,L,k);
A = A + 2*norm(A)*eye(ntot);  % Make it well-conditioned.
plot_HODLR_ranks(A,b,L,tol)
%keyboard
plot_HODLR_ranks(inv(A),b,L,tol)
%keyboard
%plot_HBS_ranks(A,b,L,tol);
%plot_HBS_ranks_v2(A,b,L,tol);
%plot_HBS_ranks_v2(inv(A),b,L,tol);
%plot_BS_ranks(A,b,n,tol);

%%% Tests for BS matrix.
k = 3; % The rank.
A = build_BS_matrix(b,n,k);
A = A + 2*norm(A)*eye(ntot);   % Make it well-conditioned.
plot_BS_ranks(A,b,n,tol);
keyboard
plot_BS_ranks(inv(A),b,n,tol);
keyboard

%%% Tests for HBS matrix.
A = build_HBS_matrix(b,L,k);
A = A + 2*norm(A)*eye(ntot);
plot_HBS_ranks(A,b,L,tol);
keyboard
plot_HBS_ranks_v2(A,b,L,tol);
keyboard
plot_HBS_ranks_v2(inv(A),b,L,tol);
keyboard
plot_HODLR_ranks(A,b,L,tol)
keyboard
plot_HODLR_ranks(inv(A),b,L,tol)
keyboard

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function builds a "random" block separable matrix. 
% The inputs are:
%    b : block size
%    n : the number of blocks
%    k : the ranks of the off-diagonal blocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = build_BS_matrix(b,n,k)

UU   = build_basis_matrix(b,n,k);
VV   = build_basis_matrix(b,n,k);
A    = UU*randn(n*k,n*k)*VV';
for ibox = 1:n
  ind        = (ibox-1)*b + (1:b);
  A(ind,ind) = randn(b);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function builds a block diagonal ON matrix.
% The inputs are:
%    b : block size
%    n : the number of blocks
%    k : the number of columns of the off-diagonal blocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function U = build_basis_matrix(b,n,k)

U = zeros(b*n,k*n);
for ibox = 1:n
  indi = (ibox-1)*b + (1:b);
  indj = (ibox-1)*k + (1:k);
  [Uloc,~] = qr(randn(b,k),0);
  U(indi,indj) = Uloc;
end

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
% This function builds an HBS (or HSS) matrix.
% The inputs are:
%    b : block size
%    L : the number of levels in the tree
%    k : the number of columns of the off-diagonal blocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = build_HBS_matrix(b,L,k)

%%% Build the matrix recursively. Start with the top level.
A = [zeros(k),randn(k);...
     randn(k),zeros(k)];

%%% Sweep over levels, coarser to finer. 
%%% Form the "AT" matrix at each level.
for ilevel = 1:(L-1)
  UU = build_basis_matrix(2*k,2^ilevel,k);
  VV = build_basis_matrix(2*k,2^ilevel,k);
  A  = UU*A*VV';
  for ibox = 1:2^ilevel
    ind = (ibox-1)*2*k + (1:(2*k));
    A(ind,ind) = [zeros(k),randn(k);...
                  randn(k),zeros(k)];
  end
end

%%% Now expand AT to the full matrix at the leaf level.
UU = build_basis_matrix(b,2^L,k);
VV = build_basis_matrix(b,2^L,k);
A  = UU*A*VV';
for ibox = 1:2^L
  ind        = (ibox-1)*b + (1:b);
  A(ind,ind) = randn(b);
end

return

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
hold off

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
hold off

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

function A = build_HODLR_matrix(b,L,k)

ntot = b*(2*L);
A    = randn(ntot,ntot);
for ilevel = 1:L
  m = b*(2^(L-ilevel));
  for ibox = 1:2^ilevel
    jbox = ibox + 1 - 2*mod(ibox+1,2);
    indi = (ibox-1)*m + (1:m);
    indj = (jbox-1)*m + (1:m);
    [Uloc,~] = qr(randn(m,k),0);
    [Vloc,~] = qr(randn(m,k),0);
    A(indi,indj) = Uloc*randn(k)*Vloc';
  end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,xx] = pretend_BIE(ntot)

tt = linspace(0,2*pi*(1 - 1/ntot),ntot);
xx = [cos(tt).*(1 + 0.2*cos(5*tt));...
      sin(tt).*(1 + 0.2*cos(5*tt))];
DD1 = xx(1,:)'*ones(1,ntot) - ones(ntot,1)*xx(1,:);
DD2 = xx(2,:)'*ones(1,ntot) - ones(ntot,1)*xx(2,:);
A   = log(DD1.*DD1 + DD2.*DD2 + eye(ntot));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

