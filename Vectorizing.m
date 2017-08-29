%% Vectorizing is good.
%
% Matlab is much faster when working with vectors than with loops.
% [This is not so true of all languages, in C and Julia it makes little difference.]
%
% So whenever we can vectorize code we can make it run much faster.
%
% The following two examples illustrate this difference.
N=10^4;

tic;
A=ones(N,N);
for a_c=1:N
    A(N,a_c)=a_c; 
    % Matlab lets you put a scalar value into all the elements of a vector. 
    % (a scalar on the RHS is about the only case in which LHS and RHS are allowed to be different sizes)
end
toc

tic;
a1=ones(1,N);
a2=1:1:N;
A=a2'*a1;
toc

%% With gpu it is even better.
%
% Notice that vectorizing things, instead of using loops, often involves
% making all the operations parallel. This means that it works even better
% in combination with gpu.

tic;
a1_gpu=ones(1,N,'gpuArray');
a2_gpu=linspace(1,N,N,'gpuArray'); % I have switched to linspace as not sure how to use 1:1:N to generate on the gpu.
A_gpu=a2_gpu'*a1_gpu;
toc

%% When you work with gpus not all commands can be used.
% A list of Matlab commands that can be run on gpu is:
% <https://au.mathworks.com/help/distcomp/run-built-in-functions-on-a-gpu.html Matlab: MATLAB Functions with gpuArray Arguments>
%
% [Side note: GPU computing in Julia looks likely to be seriously
% impressive, but for now (August 2017) the corresponding list of functions
% that have been already adapted for use on gpu is small. For more see:
% http://julialang.org/blog/2017/03/cudanative
% http://mikeinnes.github.io/2017/08/24/cudanative.html
% Once this issue has been resolved, which one imagines it will be over the
% next year or two, it seems likely that Julia will provide the nicest gpu
% computing experience of any language.]
