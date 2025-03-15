%Preliminary analysis of DBS effects
%Written by Melanie Morrison & David Mikhael

%% Choose brain regions of interest from ROI file

ROI = load('Ex_Subj_ROI.mat'); %sample data to extract ROI names

names = strings(1,size(ROI.names,2));
for i = 1:size(ROI.names,2)
    names(1,i) = ROI.names{i};
end 
names=names'; %extract names of all ROIs

%chose regions of interest by name
globuspallidus=find(contains(names,'qsm_dgm.Globus'))
ventralpallidum=find(contains(names, 'qsm_dgm.Ventral-Pallidum'))
substantianigra=find(contains(names,'qsm.Substantia'))
thalamus=find(contains(names,'atlas.Thalamus'))
caudate=find(contains(names,'qsm.Caudate'))
putamen=find(contains(names,'qsm.Putamen'))
cerebelum=find(contains(names,'qsm.Cerebelum'))
denatate=find(contains(names,'qsm.Dentate'))
somatosensory=find(contains(names,'qsm.Postcentral'))
motor=find(contains(names,'qsm.Precentral'))
supplmotor=find(contains(names,'qsm.Supp'))
subthalamic=find(contains(names, 'qsm_dgm.Subthalamic'))
rednucleus=find(contains(names, 'qsm.Red'))
nucaccumbens=find(contains(names, 'qsm_dgm.Nuc-Accumbens'))

allrois=[
globuspallidus
ventralpallidum
substantianigra
thalamus
caudate
putamen
cerebelum
denatate
somatosensory
motor
supplmotor
subthalamic
rednucleus
nucaccumbens
];


allrois_names=[
names(globuspallidus)
names(ventralpallidum) 
names(substantianigra)
names(thalamus)
names(caudate)
names(putamen)
names(cerebelum)
names(denatate)
names(somatosensory)
names(motor)
names(supplmotor)
names(subthalamic)
names(rednucleus)
names(nucaccumbens)]

corrname=string(allrois_names);
corrname=erase(corrname,'qsm_atlas.');
corrname=erase(corrname,'qsm_dgm.');
%corrname=erase(corrname,'QSM.');
match="(" + wildcardPattern + ")";
corrname=erase(corrname,match);
corrname=strip(corrname);


%% Extract FC for all subjects and ROIs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%preop 

subjects = 1:133; % if multiple subjects, load as vector
allrois_fc = zeros(length(subjects), length(allrois), length(allrois));

for i = 1:length(subjects)  
    j = sprintf('%03d', subjects(i)); % Ensures two-digit formatting
    filename = ['ROI_Subject' j '_Condition001.mat']; % Correct filename formatting
    if exist(filename, 'file')  % Check if the file exists before loading
        disp(filename)
        ROI = load(filename);
    else
        disp('1');
        warning(['File not found: ' filename]);
        continue;  % Skip iteration if the file is missing
    end

    % fc matrix
    for n = 1:length(allrois)
        for l = 1:length(allrois)
            data_n = ROI.data{allrois(n)}(:);
            data_l = ROI.data{allrois(l)}(:);
    
            % Remove NaN values before correlation
            valid_idx = ~(isnan(data_n) | isnan(data_l));
            data_n = data_n(valid_idx);
            data_l = data_l(valid_idx);
            allrois_fc(i, l, n) = corr(data_n(:), data_l(:), 'Rows', 'complete'); % Ignore NaNs

        end
    end
end

%%

% Find indices of non-motor cerebellum regions to remove
regions = allrois_names;
indices_to_remove = find(contains(regions, 'Cerebelum-7') | ...
                         contains(regions, 'Cerebelum-8') | ...
                         contains(regions, 'Cerebelum-9') | ...
                         contains(regions, 'Cerebelum-10') | ...
                         contains(regions, 'Cerebelum-Crus'));


% Remove the cerebellum regions along the first & second axes
 allrois_fc(:, indices_to_remove, :) = [];
 allrois_fc(:, :, indices_to_remove) = [];
