%% DGA_Extended_Experiments.m
%  Second MATLAB script of the study. It computes every number that the paper reports beyond DGA_Rigorous_Eval.m
%  and writes them as CSV files into the folder results_matlab/ (the paper figures are drawn from these files).
%
%  Files needed in the same folder:
%    DGA_transformer_dataset.csv   the 589 records
%    picl_records.csv              independent records (made from the public PICL data file, see data/README.md)
%  Optional (only for experiment 2):
%    dataset1_(240).xlsx           source file 1 of the DGALab package (github.com/Saleh860/DGA)
%
%  Experiments
%    1  Pooled confusion matrix, per-class recall, feature importance, Duval recall per class
%    2  One source against all other sources (needs the optional file)
%    3  Same-transformer leakage: record-wise versus transformer-wise cross-validation inside one source
%    4  Confidence and abstention (accuracy versus coverage) inside the 589 records and on independent records
%    5  Value of local records, and conformal prediction sets, on a new source (transformer-wise splits)
%  Needs: Statistics and Machine Learning Toolbox. Run time: about 15 to 40 minutes.
clear; clc; close all;
SEED = 42; EPS_ = 1e-4; NTREES = 300; MF = 5;
classes = {'PD','D1','D2','T1','T2','T3'};
% results folder: next to the script if writable, otherwise in the Documents/MATLAB folder or the temp folder
OUTDIR = 'results_matlab';
try
  if ~exist(OUTDIR,'dir'), mkdir(OUTDIR); end
  fid = fopen(fullfile(OUTDIR,'write_test.tmp'),'w'); assert(fid > 0); fclose(fid); delete(fullfile(OUTDIR,'write_test.tmp'));
catch
  OUTDIR = fullfile(userpath, 'DGA_results_matlab');
  try
    if ~exist(OUTDIR,'dir'), mkdir(OUTDIR); end
  catch
    OUTDIR = fullfile(tempdir, 'DGA_results_matlab'); if ~exist(OUTDIR,'dir'), mkdir(OUTDIR); end
  end
end
fprintf('Results will be written to: %s\n', OUTDIR);

T = readtable('DGA_transformer_dataset.csv');
G = T{:,{'H2_ppm','CH4_ppm','C2H6_ppm','C2H4_ppm','C2H2_ppm'}};
[~, y] = ismember(string(T.Fault_Type), string(classes)); y = y(:);
X = makeFeatures(G, EPS_);
nrm = @(a) round(arrayfun(@(v) (v > 0.0011) * v, a), 2);

%% Independent records (PICL file): labelled, not duplicate, above the IEC attention level, not in the 589
PR = readtable('picl_records.csv'); Gp = PR{:,{'H2','CH4','C2H6','C2H4','C2H2'}};
[~, yp] = ismember(string(PR.fault_type), string(classes));
inSet = ismember(nrm(Gp), nrm(G), 'rows');
keep = yp > 0 & PR.label_confirmed == 1 & PR.is_duplicate == 0 & PR.above_attention == 1 & ~inSet;
[~, ui] = unique(nrm(Gp(keep,:)), 'rows', 'stable'); idxK = find(keep); idxK = idxK(ui);
Gi = Gp(idxK,:); yi = yp(idxK); srcI = string(PR.source(idxK)); cidI = string(PR.case_id(idxK));
Xi = makeFeatures(Gi, EPS_);
fprintf('Independent records: %d (from %d transformers/cases)\n', numel(yi), numel(unique(cidI)));

%% Experiment 1: pooled confusion matrix, per-class recall, importance
fprintf('\nExperiment 1: confusion matrix and importance\n');
cm = zeros(6,6);
for rep = 1:3
  rng(SEED + rep); cvo = cvpartition(y, 'KFold', 5, 'Stratify', true);
  for f = 1:5
    tr = training(cvo,f); te = test(cvo,f);
    m = TreeBagger(NTREES, X(tr,:), y(tr), 'Method','classification', 'NumPredictorsToSample', MF);
    pr = str2double(predict(m, X(te,:)));
    cm = cm + confusionmat(y(te), pr, 'Order', 1:6);
  end
end
rec = diag(cm) ./ sum(cm,2); prec = diag(cm)' ./ max(sum(cm,1),1);
dv = duvalPredict(G); recDv = arrayfun(@(c) mean(dv(y==c) == c), 1:6)';
writematrix(cm, fullfile(OUTDIR, 'exp1_confusion_rf.csv'));
writetable(table(classes', rec, prec', recDv, 'VariableNames', {'class','rf_recall','rf_precision','duval_recall'}), fullfile(OUTDIR, 'exp1_per_class.csv'));
rng(SEED);
mi = TreeBagger(500, X, y, 'Method','classification', 'NumPredictorsToSample', MF, 'OOBPrediction','on', 'OOBPredictorImportance','on');
imp = mi.OOBPermutedPredictorDeltaError';
names = [strcat('log_', {'H2','CH4','C2H6','C2H4','C2H2'}), {'CH4/H2','C2H2/C2H4','C2H4/C2H6','C2H2/CH4','C2H2/H2','C2H6/C2H2','total_gas','CH4_share','C2H4_share','C2H2_share'}]';
writetable(table(names, imp, 'VariableNames', {'feature','oob_permutation_importance'}), fullfile(OUTDIR, 'exp1_importance.csv'));
fprintf('RF pooled accuracy over all test folds: %.3f\n', sum(diag(cm))/sum(cm(:)));
disp(table(classes', rec, prec', recDv, 'VariableNames', {'class','rf_recall','rf_precision','duval_recall'}));

%% Experiment 2: one source against all others (optional file)
fprintf('\nExperiment 2: single source versus other sources\n');
if isfile('dataset1_(240).xlsx')
  D1 = readtable('dataset1_(240).xlsx'); g1 = D1{:,1:5}; l1 = D1{:,6};
  [tf, loc] = ismember(nrm(G), nrm(g1), 'rows');
  idx = find(tf); idx = idx(l1(loc(idx)) == y(idx)); rest = setdiff((1:numel(y))', idx);
  a1 = cvAcc(X(idx,:), y(idx), 5, 3, SEED, NTREES, MF); a2 = cvAcc(X(rest,:), y(rest), 5, 3, SEED, NTREES, MF);
  fprintf('source set 1 only (n=%d): %.3f\nall other sources (n=%d): %.3f\n', numel(idx), a1, numel(rest), a2);
  writetable(table(["source_set_1";"other_sources"], [numel(idx);numel(rest)], [a1;a2], 'VariableNames', {'subset','n','rf_accuracy'}), fullfile(OUTDIR, 'exp2_single_source.csv'));
else
  disp('dataset1_(240).xlsx not found: experiment 2 skipped.');
end

%% Experiment 3: record-wise versus transformer-wise cross-validation inside one source (NCEPR)
fprintf('\nExperiment 3: same-transformer leakage inside NCEPR\n');
mN = srcI == "NCEPR"; XN = Xi(mN,:); yN = yi(mN); [gN, ~] = findgroups(cidI(mN));
fprintf('NCEPR records: %d, transformers: %d\n', numel(yN), max(gN));
accR = []; accG = [];
for rep = 1:5
  rng(SEED + rep); cvo = cvpartition(yN, 'KFold', 5, 'Stratify', true);
  for f = 1:5
    tr = training(cvo,f); te = test(cvo,f);
    m = TreeBagger(NTREES, XN(tr,:), yN(tr), 'Method','classification', 'NumPredictorsToSample', MF);
    accR(end+1) = mean(str2double(predict(m, XN(te,:))) == yN(te)); %#ok<AGROW>
  end
  ug = randperm(max(gN)); fold = zeros(max(gN),1); fold(ug) = mod(0:max(gN)-1, 5) + 1;   % each transformer goes to one fold
  fN = fold(gN);
  for f = 1:5
    tr = fN ~= f; te = fN == f;
    m = TreeBagger(NTREES, XN(tr,:), yN(tr), 'Method','classification', 'NumPredictorsToSample', MF);
    accG(end+1) = mean(str2double(predict(m, XN(te,:))) == yN(te)); %#ok<AGROW>
  end
end
fprintf('record-wise: %.3f +/- %.3f   transformer-wise: %.3f +/- %.3f\n', mean(accR), std(accR), mean(accG), std(accG));
writetable(table(["record_wise";"transformer_wise"], [mean(accR);mean(accG)], [std(accR);std(accG)], 'VariableNames', {'split','rf_accuracy','sd_over_folds'}), fullfile(OUTDIR, 'exp3_same_transformer.csv'));

%% Experiment 4: confidence and abstention
fprintf('\nExperiment 4: accuracy versus coverage\n');
confs = []; oks = [];
for rep = 1:3
  rng(SEED + 7*rep); cvo = cvpartition(y, 'KFold', 5, 'Stratify', true);
  for f = 1:5
    tr = training(cvo,f); te = test(cvo,f);
    m = TreeBagger(NTREES, X(tr,:), y(tr), 'Method','classification', 'NumPredictorsToSample', MF);
    P = rfProba(m, X(te,:)); [c, pc] = max(P, [], 2);
    confs = [confs; c]; oks = [oks; pc == y(te)]; %#ok<AGROW>
  end
end
rng(SEED); mAll = TreeBagger(500, X, y, 'Method','classification', 'NumPredictorsToSample', MF);
Pi = rfProba(mAll, Xi); [ci, pci] = max(Pi, [], 2); oi = pci == yi;
covs = [1 0.9 0.8 0.7 0.6 0.5]; R4 = zeros(numel(covs), 6);
for k = 1:numel(covs)
  if covs(k) < 1, thr = quantile(confs, 1 - covs(k)); else, thr = 0; end
  s = confs >= thr; si = ci >= thr;
  R4(k,:) = [covs(k), thr, mean(oks(s)), mean(s), mean(oi(si)), mean(si)];
end
T4 = array2table(R4, 'VariableNames', {'target_coverage','threshold','acc_589_cv','coverage_589','acc_independent','coverage_independent'});
disp(T4); writetable(T4, fullfile(OUTDIR, 'exp4_selective.csv'));

%% Experiment 5: local records and conformal prediction sets (NCEPR, transformer-wise splits)
fprintf('\nExperiment 5: local records and conformal sets\n');
XL = XN; yL = yN; gL = gN; NT = 100; ks = [0 10 25 50 100 150]; nRep = 20; alpha = 0.1;
accA = nan(nRep, numel(ks)); accW = nan(nRep, numel(ks)); accO = nan(nRep, numel(ks)); accD = nan(nRep,1);
covA = zeros(nRep,3); sizeA = zeros(nRep,3);
% out-of-fold probabilities on the 589 for calibration
rng(SEED); cvo = cvpartition(y, 'KFold', 5, 'Stratify', true); oof = zeros(numel(y), 6);
for f = 1:5
  tr = training(cvo,f); te = test(cvo,f);
  m = TreeBagger(NTREES, X(tr,:), y(tr), 'Method','classification', 'NumPredictorsToSample', MF);
  oof(te,:) = rfProba(m, X(te,:));
end
q589 = conformalQ(1 - oof(sub2ind(size(oof), (1:numel(y))', y)), alpha);
GLn = Gi(mN,:); base = TreeBagger(NTREES, X, y, 'Method','classification', 'NumPredictorsToSample', MF);
for rep = 1:nRep
  rng(SEED + 1000 + rep);
  cs = randperm(max(gL)); te = [];
  for c = cs, if numel(te) >= NT, break; end, te = [te; find(gL == c)]; end %#ok<AGROW>
  pool = setdiff((1:numel(yL))', te); pool = pool(randperm(numel(pool)));
  accD(rep) = mean(dvPred(GLn(te,:)) == yL(te));
  for ki = 1:numel(ks)
    k = ks(ki); loc = pool(1:k);
    Xtr = [X; XL(loc,:)]; ytr = [y; yL(loc)];
    m = TreeBagger(NTREES, Xtr, ytr, 'Method','classification', 'NumPredictorsToSample', MF);
    accA(rep,ki) = mean(str2double(predict(m, XL(te,:))) == yL(te));
    w = [ones(numel(y),1); 5*ones(k,1)];
    m = TreeBagger(NTREES, Xtr, ytr, 'Method','classification', 'NumPredictorsToSample', MF, 'Weights', w);
    accW(rep,ki) = mean(str2double(predict(m, XL(te,:))) == yL(te));
    if k >= 25
      m = TreeBagger(NTREES, XL(loc,:), yL(loc), 'Method','classification', 'NumPredictorsToSample', MF);
      accO(rep,ki) = mean(str2double(predict(m, XL(te,:))) == yL(te));
    end
  end
  Pt = rfProba(base, XL(te,:));
  qs = [q589, localQ(base, XL, yL, pool(1:50), alpha), localQ(base, XL, yL, pool(1:100), alpha)];
  for j = 1:3
    sets = (1 - Pt) <= qs(j);
    covA(rep,j) = mean(sets(sub2ind(size(sets), (1:numel(te))', yL(te))));
    sizeA(rep,j) = mean(sum(sets, 2));
  end
  fprintf('  repetition %d of %d done\n', rep, nRep);
end
cur = table(ks', mean(accA)', std(accA)', mean(accW)', std(accW)', mean(accO,'omitnan')', std(accO,'omitnan')', ...
    'VariableNames', {'local_records','acc_589_plus_local','sd_589_plus_local','acc_weighted5','sd_weighted5','acc_local_only','sd_local_only'});
disp(cur); writetable(cur, fullfile(OUTDIR, 'exp5_local_curve.csv'));
fprintf('Duval Triangle on the same test sets: %.3f +/- %.3f\n', mean(accD), std(accD));
writetable(table(mean(accD), std(accD), 'VariableNames', {'duval_acc','duval_sd'}), fullfile(OUTDIR, 'exp5_duval.csv'));
cf = table(["calibrated_on_589_oof";"calibrated_on_50_local";"calibrated_on_100_local"], mean(covA)', std(covA)', mean(sizeA)', ...
    'VariableNames', {'calibration','observed_coverage','sd_coverage','mean_set_size'});
disp(cf); writetable(cf, fullfile(OUTDIR, 'exp5_conformal.csv'));
fprintf('\nAll experiments finished. Results are in %s\n', OUTDIR);

%% ===================== local functions =====================
function X = makeFeatures(G, e)
  H2=G(:,1); CH4=G(:,2); C2H6=G(:,3); C2H4=G(:,4); C2H2=G(:,5);
  s = CH4 + C2H4 + C2H2 + e;
  X = [log1p(G), CH4./(H2+e), C2H2./(C2H4+e), C2H4./(C2H6+e), C2H2./(CH4+e), C2H2./(H2+e), C2H6./(C2H2+e), ...
       sum(G,2), CH4./s*100, C2H4./s*100, C2H2./s*100];
end

function P = rfProba(m, Xte)
  [~, sc] = predict(m, Xte);
  cn = str2double(m.ClassNames); P = zeros(size(Xte,1), 6); P(:, cn) = sc;
end

function a = cvAcc(X, y, nOuter, nRep, seed, ntrees, mf)
  accs = [];
  for rep = 1:nRep
    rng(seed + 11*rep); cvo = cvpartition(y, 'KFold', nOuter, 'Stratify', true);
    for f = 1:nOuter
      m = TreeBagger(ntrees, X(training(cvo,f),:), y(training(cvo,f)), 'Method','classification', 'NumPredictorsToSample', mf);
      accs(end+1) = mean(str2double(predict(m, X(test(cvo,f),:))) == y(test(cvo,f))); %#ok<AGROW>
    end
  end
  a = mean(accs);
end

function q = conformalQ(scores, alpha)
  s = sort(scores); n = numel(s); k = min(n, ceil((n + 1) * (1 - alpha))); q = s(k);
end

function q = localQ(m, XL, yL, idx, alpha)
  P = rfProba(m, XL(idx,:)); q = conformalQ(1 - P(sub2ind(size(P), (1:numel(idx))', yL(idx))), alpha);
end

function pred = dvPred(G)
  pred = duvalPredict(G);
end

function [x, yy] = tri(a, b, c)
  t = a + b + c; a = a/t; b = b/t; c = c/t; x = 0.5*(2*b + c); yy = (sqrt(3)/2)*c;
end

function pred = duvalPredict(G)
  Z = {{[98 1 98],[0 0 2],[2 0 0]}, {[0 0 64 87],[1 77 13 13],[0 23 23 0]}, ...
       {[0 0 31 47 64],[77 29 29 13 13],[23 71 40 40 23]}, ...
       {[76 80 98 98 96],[4 0 0 2 4],[20 20 2 0 0]}, {[46 50 80 76],[4 0 0 4],[50 50 20 20]}, ...
       {[0 0 50 35],[15 0 0 15],[85 1 50 50]}, ...
       {[0 0 35 46 96 87 47 31],[29 15 15 4 4 13 13 29],[71 85 50 50 0 0 40 40]}};
  code = [1 2 3 4 5 6 0];
  pred = zeros(size(G,1),1);
  for r = 1:size(G,1)
    t = G(r,2) + G(r,4) + G(r,5); if t <= 0, continue; end
    [px, py] = tri(100*G(r,2)/t, 100*G(r,5)/t, 100*G(r,4)/t);
    for z = 1:numel(Z)
      vx = zeros(1, numel(Z{z}{1})); vy = vx;
      for v = 1:numel(vx), [vx(v), vy(v)] = tri(Z{z}{1}(v), Z{z}{2}(v), Z{z}{3}(v)); end
      if inpolygon(px, py, vx, vy), pred(r) = code(z); break; end
    end
  end
end
