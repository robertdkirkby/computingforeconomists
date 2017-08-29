%% Importing data from FRED using getFredData
%
% Using _getFredData()_ you can automatically download data from FRED
% (Federal Reserve Economic Database, <https://fred.stlouisfed.org/>)
%
% This has two main advantages. First is that it is easy to do and allows
% you to get a lot of Economic data into Matlab without any intermediate
% steps. Second is that by just changing the 'End Date' you can
% automatically update/extend all of your work/analysis to include the most recent
% data.
%
% To do this you will first need to download the getFredData 
% Download <https://www.dropbox.com/s/u32d5f0wds99iyi/getFredData.m?dl=1 getFredData.m> 
% (you can either put it in your active folder, or better yet add it to the <http://au.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html Matlab path>). [<https://github.com/robertdkirkby/getfreddata-matlab Github>]
%
% Now just import the series you want (here M1 money for period 1960 to 2000):
M1 = getFredData('M1SL', '1960-01-01', '2000-12-31')
%%
% (You can find the codes, here 'M1SL' next to the name when looking at the
% data you want in FRED; the code is also part of the url FRED uses.)
% This will give you a structure M1, the data series itself is in M1.Data(:,2).
% The M1 structure also contains other info, eg. M1.Title, or the dates in M1.Data(:,1).
% Dates are in Matlab's _datenum_ format, counts days starting from 0000-01-01. 
% You can convert them using Matlab functions _datestr_ and _datevec_, graph them with _dateticks_. 
% Sometimes you will want them in a format telling you the quarter; use _datevec_, get months, and then 
% turn months to quarters using ceil('month'/3).
% More options: M1 is monthly by default, but we can ask for quarterly data ('q') and expressed as Percentage Change of Year Ago ('pca').
M1 = getFredData('M1SL', '1960-01-01', '2000-12-31','pca','q')
%%
%There are other, more advanced options in getFredData, including ALFRED historical data, either use Matlab help or read the m-file for details.
%[Matlab has a built-in function _fetch_ for accessing FRED, but it can’t change frequency or units. getFredData.m builds on an earlier version by <http://www.kimjruhl.com/ Kim Ruhl>]