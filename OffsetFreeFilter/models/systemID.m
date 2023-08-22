%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads data from a csv file to identify a state space system.
% x(k+1) = A*x(k) + B*u(k)
% y(k) = C*x(k)
%
% Inputs:
% * file name of csv file where data is located
%
% Outputs:
% * A, B, C matrices corresponding to process form with state feedback
% * nominal steady state values
% * maximum errors between linear model and data
% * minimum errors between linear model and data
% * information on the performance metrics of the identified model
% * copy of information on the data used
%
% REQUIREMENTS:
% * MATLAB R2019b or later
% * System Identification Toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USER INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = '2023_08_21_17h31m03s_APPJ_model_train_data.mat';
data_direction = 0; % 0 for column-wise data, 1 for row-wise data
ny = 3;
nu = 2;

% labels corresponding to the data
if ny == 3
    y_labels = {'T (^\circC)', 'I(He706) (arb. units.)', 'I(O777) (arb. units.)'}; % outputs
elseif ny == 2
    y_labels = {'T (^\circC)', 'I (arb. units.)'}; % outputs
else
    string_labels = cellstr(string(1:ny-1));
    y_labels = [{'T (^\circC)'} string_labels]; % outputs
end
u_labels = {'P (W)', 'q (SLM)'}; % inputs

Ts = 1; % Sampling time (in seconds)

plot_fit = 1; % 1 for yes, 0 for no; plot a comparison of the data
% and identified model

est_function = 'n4sid'; % choose 'ssest' or 'n4sid'

validate_sys = 0;   % option to validate data (1 for yes, 0 for no)
valid_split = 0.25;  % validation split, i.e. how much of the data to reserve for validation

saveModel = 1; % 1 for yes, 0 for no
% specify an output file name, otherwise a default is used; please include
% '.mat' in your filename
out_filename = ['APPJmodel_UCB_LAM_2023_08_21', '.mat'];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
data = load(filename);
ydata = data.y';
udata = data.u';
yss = data.yss;
uss = data.uss;
out_filename = [data.timestamp, '_APPJmodel.mat'];

%% Plot data to visualize it
disp('Plotting data to visualize it... See Figure 1 to verify data.')
figure(1)
if ny < 4
    for i = 1:ny
        subplot(ny, nu, i)
        plot(ydata(:,i), 'linewidth', 2)
        ylabel(y_labels{i})
        set(gca, 'fontsize', 15)
    end
else
    y_idxs = [1, 1015, 1228];
    for i = 1:length(y_idxs)
        subplot(length(y_idxs), nu, i)
        plot(ydata(:,y_idxs(i)), 'linewidth', 2)
        ylabel(y_labels{y_idxs(i)})
        set(gca, 'fontsize', 15)
    end
    xlabel('Time Step')
end

set(gcf,'color','w');

for i = 1:nu
    if ny < 4
        subplot(ny, nu, i+ny)
    else
        subplot(length(y_idxs), nu, i+length(y_idxs))
    end
    stairs(udata(:,i), 'linewidth', 2)
    ylabel(u_labels{i})
    set(gca, 'fontsize', 15)
end
xlabel('Time Step')
set(gcf,'color','w');

%% Identify the model
disp('Identifying the model...')

modelOrder = 3;
subIDdata = iddata(ydata, udata, Ts);
Ndata = subIDdata.N;

if strcmpi(est_function, 'ssest')
    opt = ssestOptions;
    opt.SearchOptions.Tolerance = 1e-8;
    opt.OutputWeight = [100,0;0,1];
    sys = ssest(subIDdata,modelOrder, 'DisturbanceModel', 'none', ...
        'Form', 'canonical', 'Ts', Ts);
%     sys = ssest(subIDdata,modelOrder, 'DisturbanceModel', 'none', ...
%         'Form', 'canonical', 'Ts', Ts, opt);
elseif strcmpi(est_function, 'n4sid')
    sys = n4sid(subIDdata, modelOrder, 'DisturbanceModel', 'none', ...
        'Form', 'canonical', 'Ts', Ts);
else
    warning('Invalid estimation function! Using ''ssest''...')
    opt = ssestOptions('OutputWeight', [1,0;0,1]);
    sys = ssest(subIDdata,modelOrder, 'DisturbanceModel', 'none', ...
        'Form', 'canonical', 'Ts', Ts);
end

A = sys.A;
B = sys.B;
C = sys.C;

if plot_fit
    % Verify the model graphically
    disp('Verifying model graphically... See Figure 3.')
    simTime = 0:Ts:Ts*(Ndata-1);
    yCompare = lsim(sys, udata, simTime);

    opt = compareOptions('InitialCondition', zeros(modelOrder,1));
    figure(3)
    compare(subIDdata, sys, opt)
    xlabel('Time/s')
    legend('Experimental Data', 'Linear Model')
    title('Trained Model')
    set(gcf,'color','w');

end

wmaxTrain = max(ydata-yCompare);
wminTrain = min(ydata-yCompare);


% determine max and min errors
maxErrors = max(wmaxTrain, [], 1);
minErrors = min(wminTrain, [], 1);
disp(['Maximum Output Errors: ', num2str(maxErrors)])
disp(['Minimum Output Errors: ', num2str(minErrors)])

%% save model if specified
% save information on the data
dataInfo.yLabels = y_labels;    % user-defined labels for the outputs
dataInfo.uLabels = u_labels;    % user-defined labels for the inputs
dataInfo.samplingTime = Ts;     % sampling time from the data
dataInfo.fileName = filename;   % file name of where the data was taken
dataInfo.ydata = ydata;
dataInfo.udata = udata;
dataInfo.sys = sys;
dataInfo.ypred = yCompare;
    
% save information on the performance of the identified model
if saveModel
    % if output filename is not specified, generate one from a default
    if isempty(out_filename)
        out_filename = ['APPJmodel_', filedate, '.mat'];
    end

    % check if the file already exists to ensure no overwritten files
    if isfile(out_filename)
        overwrite_file = input('Warning: File already exists in current path! Do you want to overwrite? [1 for yes, 0 for no]: ');
        if overwrite_file
            save(out_filename, 'A', 'B', 'C', 'yss', 'uss', 'maxErrors', 'minErrors', 'dataInfo')
        else
            out_filename = input('Input a new filename (ensure .mat is included in your filename) or press Enter if you no longer want to save the identified model: \n', 's');
            if isempty(out_filename)
                disp('Identified system not saved.')
            else
                save(out_filename, 'A', 'B', 'C', 'yss', 'uss', 'maxErrors', 'minErrors', 'dataInfo')
            end
        end
    else
        save(out_filename, 'A', 'B', 'C', 'yss', 'uss', 'maxErrors', 'minErrors', 'dataInfo')
    end
end