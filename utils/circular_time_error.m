function err = circular_time_error(predH, predM, trueH, trueM)
%CIRCULAR_TIME_ERROR Compute circular time distance in minutes.
%   err = circular_time_error(predH, predM, trueH, trueM)
%
%   Handles the 12-hour wrap-around correctly:
%     e.g. 11:55 vs 12:05 = 10 min error (not 710)
%
%   Inputs:
%     predH, trueH  - hour values (1..12)
%     predM, trueM  - minute values (0..59)
%     All inputs can be vectors of the same length.
%
%   Output:
%     err - absolute time error in minutes (0..360 range)

    % Convert to total minutes on a 12-hour clock (0..719)
    predTotal = mod(predH - 1, 12) * 60 + predM;
    trueTotal = mod(trueH - 1, 12) * 60 + trueM;

    % Raw absolute difference
    diff = abs(predTotal - trueTotal);

    % Wrap-around: take the shorter path on the 720-minute circle
    err = min(diff, 720 - diff);
end
