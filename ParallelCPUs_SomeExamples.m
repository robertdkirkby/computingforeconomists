%% Parallel CPUs: some guidelines and nice applications.
%
% Perhaps the easiest bit of parallelization you can do in Matlab (and most
% other high-level programming languages) is to parallelize for-loops.
% Obviously this will only work when your for-loops do not involve each
% step depending on the previous step. If the contents of each for-loop are
% simple you will be better of vectorizing (and perhaps then using the
% gpu).
%
% One nice example of something you might want to do is to solve some
% problem for many different parameter values. Obviously the solution for
% one parameter value is going to be different from the solution for
% another parameter value, and so a parallel for loop is likely a good idea
% here.
%
% [Common applications of this in Economics would be as part of estimation, or as part of robustness.]

N=1000;
thetavec=linspace(1,5,N);

options=optimoptions('fmincon','Display','off');
tic;
xyvec=nan(length(thetavec),2);
for ii = 1:length(thetavec)
    theta=thetavec(ii);
    modelfn=@(xy) xy(1)+3*xy(1)-theta*xy(1)^2-2*xy(2)+theta*xy(1)*xy(2); % Note: Idea is that xy(1) is x, and xy(2) is y. Just need to write it this way as fmincon requires it to be a function that has a vector input.
    xy0=[1,1];
    xy=fmincon(modelfn,xy0,[],[],[],[],[-100,-100],[100,100],[],options); % -100 and 100 are upper and lower bounds on x
    xyvec(ii,:)=xy;
end
toc

tic;
xyvec=nan(length(thetavec),2);
parfor ii = 1:length(thetavec)
    theta=thetavec(ii);
    modelfn=@(xy) xy(1)+3*xy(1)-theta*xy(1)^2-2*xy(2)+theta*xy(1)*xy(2);
    xy0=[1,1];
    xy=fmincon(modelfn,xy0,[],[],[],[],[-100,-100],[100,100],[],options); % -100 and 100 are upper and lower bounds on x
    xyvec(ii,:)=xy;
end
toc
% Note: if you actually wanted to find the zeros of a function there are
% specific Matlab commands for doing this (like fzero). This example is
% just intended as an illustration of parfor.

%%
% The gains from parfor in this example are small. This is because
% the fmincon of modelfn was itself not something that took very long. If
% this command had taken a long time then the speed boost of parfor would
% be much larger.

%% Slicing matrices
%
% Say you want your parallel for loop so that each iteration works with a
% specific line of a matrix. In principle Matlab will look at this and
% automatically figure out that each cpu should only be sent the relevant
% row of the matrix. In practice this is not always true, and telling it
% that this is the case can substantially boost speed by reducing overhead.
% 
% Take the following example, you can see that Matlab has done almost as well automatically
% slicing this up as we have done by manually doing it (in the second case). In more complicated
% applications however Matlab will be unable to spot all of the slicing
% tricks and so the automated version will be substantially slower. (Often
% Matlab gives a warning that it suspects a certain variable could be
% sliced but was unable to do so.)

N=5000;

A=randn(N,N);
tic;
parfor ii=1:N
    A(ii,:)=log(abs(A(ii,:))+0.01).*A(ii,:);
end
toc

A=randn(N,N);
tic;
parfor ii=1:N
    A_ii=A(ii,:);
    A_ii=log(abs(A_ii)+0.01).*A_ii;
    A(ii,:)=A_ii;
end
toc

%%
% Remember that Matlab is storing matrices so that their columns are
% continguous in memory. So actually if we were smarter we would do the
% same thing but with A in such a form that we take column slices.
%

A=randn(N,N);
tic;
parfor ii=1:N
    A_ii=A(:,ii);
    A_ii=log(abs(A_ii)+0.01).*A_ii;
    A(:,ii)=A_ii;
end
toc

%%
% Notice also that we could actually here just do the whole thing as
% matrix operations with no need to loop (we might even then be able
% to use the GPU and further speed things up). Both of these would give further speed boosts. The point
% here however was about the slicing of matrices with for-loops which can
% be important boost of speed in practice.
% (GPU time here is misleading vs its performance in most applications, but provides a good lesson in how it is not a solution to every situation.)

A=randn(N,N);
tic;
A=log(abs(A)+0.01).*A;
toc

A_gpu=randn(N,N,'gpuArray');
tic;
A_gpu=log(abs(A_gpu)+0.01).*A_gpu;
toc



%% Some links:
% <https://au.mathworks.com/help/distcomp/decide-when-to-use-parfor.html>
    

