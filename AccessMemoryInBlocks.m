%% Example of locality in memory allocation
% Matlab stores matrices by column, so working with columns, rather than
% rows, can make a big difference.
% [I am not sure in what order other languages store matrices.]

M = 2000; % Try 1000, 2000, 5000
N = 10000;

% A smart way to do it
tic
a = zeros(M,N);
toc

clear a

% A not so smart way to do it
% a = zeros(m,n); % Comment in to check latency even with preallocation

tic
for n = 1:N
    for m = 1:M
        a(m,n) = 0;
    end
end
toc

clear a

% A really dumb way to do it
% a = zeros(m,n); % Comment in to check latency even with preallocation

tic
for m = 1:M
    for n = 1:N
        a(m,n) = 0;
    end
end
toc

% When objects of any kind are stored in memory they will be stored in
% blocks. For example an 64-bit floating number will be stored as a single
% 'block' of 64 contiguous (consecutive) bits (a bit is a single 0 or 1,
% the most basic unit of all computing).
% If one object immediately follows the previous then the computer can
% quickly move from one to the next, and this will be 'fast'. 
% If one object does not immediately follow the previous then the computer
% cannot move directly onto the next, instead it needs to look at a
% 'pointer' that tells it where to find the next one and then go to there.
% This takes much longer.
% Matlab stores matrices in memory as one column following immediately on
% from another. So when we stick to the same column, do everything there,
% and then move to the next column, we are sticking to 'contiguous' blocks
% of memory. This is what we do in the second example, and are not doing in
% the third example. Hence why second is faster than third.
% Preallocation means that Matlab knows how much memory will be needed and
% can find a large enough chuck of contiguous memory to begin with.
% If we don't preallocate then Matlab will use a small amount of contiguous
% memory to begin with. Once it turns out this is not enough the it has to
% create a complete copy of what it has done so far in an amount of
% contiguous memory twice the original size and then continue to fill this
% in. (If this runs out then it repeats the process until done: find 
% contiguous memory of twice the size, copy what we have so far, fill out 
% the rest.)






