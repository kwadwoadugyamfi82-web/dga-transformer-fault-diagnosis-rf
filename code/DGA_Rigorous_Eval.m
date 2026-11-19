%% DGA_Rigorous_Eval.m
%  MATLAB version of the evaluation protocol in dga_eval.py / dga_tc10.py.
%  Needs: Statistics and Machine Learning Toolbox (TreeBagger, fitcknn, fitcecoc, cvpartition),
%         Deep Learning Toolbox (patternnet).
%  Files in the same folder: DGA_transformer_dataset.csv, tc10_cases.csv
%
%  Protocol (fixed before looking at results)
%   * Outer: repeated stratified 5-fold CV, 3 repeats (15 test folds).
%   * Inner: 3-fold CV on the outer-training part chooses hyper-parameters (nested CV).
%   * Every model gets a small grid of equal effort.
%   * Scaling is fitted on the training part only. SMOTE (variant "SMOTE") only on training part.
%   * Duval Triangle 1 (simplified polygons) and IEC 60599 ratio rules as no-training baselines;
%     unclassified outputs and the Duval "DT" zone count as wrong.
%   * Part B: train on literature rows only, test on the IEC TC 10 inspection-confirmed cases
%     (every row matching a TC 10 case is removed from training first).
clear; clc; close all;
SEED = 42; EPS_ = 1e-4;
classes = {'PD','D1','D2','T1','T2','T3'};
T = readtable('DGA_transformer_dataset.csv');
G = T{:,{'H2_ppm','CH4_ppm','C2H6_ppm','C2H4_ppm','C2H2_ppm'}};
[~, y] = ismember(string(T.Fault_Type), string(classes)); y = y(:);

X = makeFeatures(G, EPS_);
nOuter = 5; nRep = 3;
RUN_NESTED = false;   % true = repeat the long nested CV (30-60 min); results already obtained are in the report

%% Part A: baselines on all rows
dv = duvalPredict(G); ie = iecPredict(G, EPS_);
fprintf('Duval Triangle 1 (simplified): accuracy %.3f (all rows)\n', mean(dv == y));
fprintf('IEC ratio table: accuracy %.3f (all rows), coverage %.3f, accuracy on classified %.3f\n', ...
    mean(ie == y), mean(ie > 0), mean(ie(ie>0) == y(ie>0)));

models = {'RF','KNN','SVM','ANN'};
if RUN_NESTED
%% Part A: nested repeated CV
variants = {false, true};   % use SMOTE?
results = struct('model',{},'smote',{},'acc',{});
for mi = 1:numel(models)
  for vi = 1:numel(variants)
    name = models{mi}; useSmote = variants{vi};
    if useSmote && ~ismember(name, {'RF','SVM'}), continue; end   % SMOTE only compared on RF and SVM
    accs = [];
    for rep = 1:nRep
      rng(SEED + rep);
      cvo = cvpartition(y, 'KFold', nOuter, 'Stratify', true);
      for f = 1:nOuter
        tr = training(cvo, f); te = test(cvo, f);
        % inner tuning
        grid = modelGrid(name); bestAcc = -1; bestP = grid{1};
        rng(SEED + 100*rep + f);
        cvi = cvpartition(y(tr), 'KFold', 3, 'Stratify', true);
        Xt_ = X(tr,:); yt_ = y(tr);
        for gi = 1:numel(grid)
          s = zeros(3,1);
          for k = 1:3
            pr = fitPredict(name, grid{gi}, Xt_(training(cvi,k),:), yt_(training(cvi,k)), ...
                            Xt_(test(cvi,k),:), useSmote);
            s(k) = mean(pr == yt_(test(cvi,k)));
          end
          if mean(s) > bestAcc, bestAcc = mean(s); bestP = grid{gi}; end
        end
        pr = fitPredict(name, bestP, X(tr,:), y(tr), X(te,:), useSmote);
        accs(end+1,1) = mean(pr == y(te)); %#ok<SAGROW>
      end
    end
    results(end+1) = struct('model',name,'smote',useSmote,'acc',accs); %#ok<SAGROW>
    fprintf('%-4s SMOTE=%d : %.3f +/- %.3f (n=%d folds)\n', name, useSmote, mean(accs), std(accs), numel(accs));
  end
end
% paired comparisons, Nadeau-Bengio corrected t-test
getAcc = @(n,s) results(find(strcmp({results.model},n) & [results.smote]==s, 1)).acc;
ntest = round(numel(y)/nOuter); ntrain = numel(y) - ntest;
pairs = {{'RF',false,'SVM',false},{'RF',false,'KNN',false},{'RF',false,'ANN',false},{'RF',false,'RF',true}};
for i = 1:numel(pairs)
  p = pairs{i}; d = getAcc(p{1},p{2}) - getAcc(p{3},p{4});
  t = mean(d) / sqrt((1/numel(d) + ntest/ntrain) * var(d) + 1e-12);
  pv = 2*(1 - tcdf(abs(t), numel(d)-1));
  fprintf('%s(SMOTE=%d) - %s(SMOTE=%d): diff %.4f, NB p = %.3f\n', p{1},p{2},p{3},p{4}, mean(d), pv);
end
try, save('DGA_rigorous_results.mat','results'); catch, disp('Could not save results file (folder not writable); continuing.'); end

end

%% Part B: train on literature rows, test on IEC TC 10 inspection-confirmed cases
TC = readtable('tc10_cases.csv');
Gt = TC{:,{'h2','ch4','c2h6','c2h4','c2h2'}}; lab = string(TC.lab);
nrm = @(a) round(arrayfun(@(v) (v > 0.0011) * v, a), 2);
inTC = ismember(nrm(G), nrm(Gt), 'rows');
fprintf('\nTC10: %d cases; removed %d matching rows from training, %d left\n', numel(lab), sum(inTC), sum(~inTC));
Xtr = X(~inTC,:); ytr = y(~inTC); Xte = makeFeatures(Gt, EPS_);
typeOf = containers.Map({'PD','D1','D2','T1','T2','T3','T1T2'},{'PD','E','E','T','T','T','T'});
score = @(predIdx) deal( ...
    mean(arrayfun(@(i) predIdx(i) > 0 && (strcmp(classes{predIdx(i)}, lab(i)) || (lab(i)=="T1T2" && ismember(predIdx(i),[4 5]))), 1:numel(lab))), ...
    mean(arrayfun(@(i) predIdx(i) > 0 && strcmp(typeOf(classes{predIdx(i)}), typeOf(char(lab(i)))), 1:numel(lab))));
for mi = 1:numel(models)
  name = models{mi}; grid = modelGrid(name); bestAcc = -1; bestP = grid{1};
  rng(SEED); cvi = cvpartition(ytr, 'KFold', 5, 'Stratify', true);
  for gi = 1:numel(grid)
    s = zeros(5,1);
    for k = 1:5
      pr = fitPredict(name, grid{gi}, Xtr(training(cvi,k),:), ytr(training(cvi,k)), Xtr(test(cvi,k),:), false);
      s(k) = mean(pr == ytr(test(cvi,k)));
    end
    if mean(s) > bestAcc, bestAcc = mean(s); bestP = grid{gi}; end
  end
  a6 = zeros(5,1); aT = zeros(5,1);
  for seed = 1:5
    rng(seed); pr = fitPredict(name, bestP, Xtr, ytr, Xte, false);
    [a6(seed), aT(seed)] = score(pr);
  end
  fprintf('TC10 %-4s: 6-class %.3f, 3-type %.3f (inner CV %.3f)\n', name, mean(a6), mean(aT), bestAcc);
end
[a6, aT] = score(duvalPredict(Gt)); fprintf('TC10 Duval Triangle: 6-class %.3f, 3-type %.3f\n', a6, aT);
[a6, aT] = score(iecPredict(Gt, EPS_)); fprintf('TC10 IEC ratios:     6-class %.3f, 3-type %.3f\n', a6, aT);

%% Part C: why published accuracies are higher (controlled experiments, RF, 5x3 repeated CV)
fprintf('\nPart C: leakage and label-origin experiments (random forest)\n');
rfCV = @(Xa, ya) cvAcc(Xa, ya, nOuter, nRep, SEED);
fprintf('A correct protocol, pooled 589 rows       : %.3f\n', rfCV(X, y));
mu = mean(X); sd = std(X); sd(sd==0) = 1; Z = (X - mu)./sd;
rng(SEED); [Zb, yb] = smoteBalance(Z, y, 5);
fprintf('B SMOTE on whole data BEFORE CV (leaky)    : %.3f  (n=%d)\n', rfCV(Zb, yb), numel(yb));
dz = duvalPredict(G); ok = dz > 0;                        % rule-derived labels, DT/unclassified rows dropped
fprintf('C labels = simplified Duval output         : %.3f  (n=%d)\n', rfCV(X(ok,:), dz(ok)), sum(ok));
% D: optional single-source subset, only if dataset1_(240).xlsx from the DGALab package is in this folder
if isfile('dataset1_(240).xlsx')
  D1 = readtable('dataset1_(240).xlsx'); g1 = D1{:,1:5}; l1 = D1{:,6};
  [tf, loc] = ismember(nrm(G), nrm(g1), 'rows');
  idx = find(tf); idx = idx(l1(loc(idx)) == y(idx));
  fprintf('D single source (DGALab set 1)             : %.3f  (n=%d)\n', rfCV(X(idx,:), y(idx)), numel(idx));
  rest = setdiff(1:numel(y), idx);
  fprintf('D2 all other sources pooled                : %.3f  (n=%d)\n', rfCV(X(rest,:), y(rest)), numel(rest));
end

%% Part D: independent records from the public PICL data file (github.com/yhtang714/PICL-for-Risk-Aware-Transformer-Fault-Diagnosis-from-Dissolved-Gas-Analysis)
%  Train on the 589 rows, test on labelled, non-duplicate records that are NOT in the 589 and are above the IEC attention level.
if isfile('picl_records.csv')
  PR = readtable('picl_records.csv'); Gp = PR{:,{'H2','CH4','C2H6','C2H4','C2H2'}};
  [~, yp] = ismember(string(PR.fault_type), string(classes));
  inSet = ismember(nrm(Gp), nrm(G), 'rows');
  keep = yp > 0 & PR.label_confirmed == 1 & PR.is_duplicate == 0 & PR.above_attention == 1 & ~inSet;
  [~, ui] = unique(nrm(Gp(keep,:)), 'rows', 'stable'); idxK = find(keep); idxK = idxK(ui);
  Gp = Gp(idxK,:); yp = yp(idxK); src = string(PR.source(idxK));
  fprintf('\nPart D: %d independent labelled records (above attention level). Sources: ', numel(yp));
  us = unique(src); for k = 1:numel(us), fprintf('%s=%d ', us(k), sum(src==us(k))); end, fprintf('\n');
  Xp = makeFeatures(Gp, EPS_);
  rng(SEED); pr = fitPredict('RF', struct('mf',5,'leaf',1), X, y, Xp, false);
  fprintf('RF trained on 589:      accuracy %.3f\n', mean(pr == yp));
  tp = [1 2 2 3 3 3]; fprintf('RF trained on 589:      3-type accuracy %.3f\n', mean(tp(pr(:)) == tp(yp(:))));
  for k = 1:numel(us), m = src == us(k); fprintf('   %-14s n=%3d  RF %.3f\n', us(k), sum(m), mean(pr(m) == yp(m))); end
  dvp = duvalPredict(Gp); iep = iecPredict(Gp, EPS_);
  fprintf('Duval Triangle (simplified): %.3f   IEC ratios: %.3f\n', mean(dvp == yp), mean(iep == yp));
end

%% ===================== local functions =====================
function a = cvAcc(X, y, nOuter, nRep, seed)
  accs = [];
  for rep = 1:nRep
    rng(seed + 11*rep); cvo = cvpartition(y, 'KFold', nOuter, 'Stratify', true);
    for f = 1:nOuter
      m = TreeBagger(300, X(training(cvo,f),:), y(training(cvo,f)), 'Method','classification', 'NumPredictorsToSample', 5);
      accs(end+1) = mean(str2double(predict(m, X(test(cvo,f),:))) == y(test(cvo,f))); %#ok<AGROW>
    end
  end
  a = mean(accs);
end

function X = makeFeatures(G, e)
  H2=G(:,1); CH4=G(:,2); C2H6=G(:,3); C2H4=G(:,4); C2H2=G(:,5);
  s = CH4 + C2H4 + C2H2 + e;
  X = [log1p(G), CH4./(H2+e), C2H2./(C2H4+e), C2H4./(C2H6+e), C2H2./(CH4+e), C2H2./(H2+e), C2H6./(C2H2+e), ...
       sum(G,2), CH4./s*100, C2H4./s*100, C2H2./s*100];
end

function grid = modelGrid(name)
  grid = {};
  switch name
    case 'RF',  for mf = [3 5], for l = [1 3], grid{end+1} = struct('mf',mf,'leaf',l); end, end
    case 'KNN', for k = [1 3 5 9], grid{end+1} = struct('k',k); end
    case 'SVM', for C = [1 10], for ks = [4 10], grid{end+1} = struct('C',C,'ks',ks); end, end
    case 'ANN', for h = {[20 10],50,[50 25]}, grid{end+1} = struct('h',h{1}); end
  end
end

function pred = fitPredict(name, p, Xtr, ytr, Xte, useSmote)
  mu = mean(Xtr); sd = std(Xtr); sd(sd==0) = 1;
  A = (Xtr - mu)./sd; B = (Xte - mu)./sd;
  if useSmote, [A, ytr] = smoteBalance(A, ytr, 5); end
  switch name
    case 'RF'
      m = TreeBagger(300, A, ytr, 'Method','classification', 'NumPredictorsToSample', p.mf, 'MinLeafSize', p.leaf);
      pred = str2double(predict(m, B));
    case 'KNN'
      m = fitcknn(A, ytr, 'NumNeighbors', p.k); pred = predict(m, B);
    case 'SVM'
      t = templateSVM('KernelFunction','rbf','BoxConstraint',p.C,'KernelScale',p.ks);
      m = fitcecoc(A, ytr, 'Learners', t); pred = predict(m, B);
    case 'ANN'
      net = patternnet(p.h); net.trainParam.showWindow = false; net.divideFcn = 'dividetrain';
      net = train(net, A', full(ind2vec(ytr', 6)));
      pred = vec2ind(net(B'))';
  end
  pred = pred(:);
end

function [Xb, yb] = smoteBalance(X, y, k)
  cls = unique(y); counts = histcounts(y, [cls; max(cls)+1]); target = max(counts);
  Xb = X; yb = y;
  for c = 1:numel(cls)
    idx = find(y == cls(c)); n = numel(idx); need = target - n;
    if need <= 0 || n < 2, continue; end
    Xc = X(idx,:); kk = min(k, n-1); S = zeros(need, size(X,2));
    for i = 1:need
      b = randi(n); d = sum((Xc - Xc(b,:)).^2, 2); [~, o] = sort(d);
      nb = o(1 + randi(kk)); S(i,:) = Xc(b,:) + rand() * (Xc(nb,:) - Xc(b,:));
    end
    Xb = [Xb; S]; yb = [yb; repmat(cls(c), need, 1)]; %#ok<AGROW>
  end
end

function [x, yy] = tri(a, b, c)
  t = a + b + c; a = a/t; b = b/t; c = c/t; x = 0.5*(2*b + c); yy = (sqrt(3)/2)*c;
end

function pred = duvalPredict(G)
  % zones as [%CH4; %C2H2; %C2H4] vertex lists (commonly used simplified Duval Triangle 1 polygons)
  Z = {{[98 1 98],[0 0 2],[2 0 0]}, {[0 0 64 87],[1 77 13 13],[0 23 23 0]}, ...
       {[0 0 31 47 64],[77 29 29 13 13],[23 71 40 40 23]}, ...
       {[76 80 98 98 96],[4 0 0 2 4],[20 20 2 0 0]}, {[46 50 80 76],[4 0 0 4],[50 50 20 20]}, ...
       {[0 0 50 35],[15 0 0 15],[85 1 50 50]}, ...
       {[0 0 35 46 96 87 47 31],[29 15 15 4 4 13 13 29],[71 85 50 50 0 0 40 40]}};
  code = [1 2 3 4 5 6 0];   % PD D1 D2 T1 T2 T3 then DT (0 = counts as wrong)
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

function pred = iecPredict(G, e)
  pred = zeros(size(G,1),1);
  for r = 1:size(G,1)
    H2=G(r,1); CH4=G(r,2); C2H6=G(r,3); C2H4=G(r,4); C2H2=G(r,5);
    r1 = C2H2/(C2H4+e); r2 = CH4/(H2+e); r3 = C2H4/(C2H6+e);
    if r2 < 0.1 && r3 < 0.2, pred(r) = 1;                                   % PD
    elseif r1 > 1 && r2 >= 0.1 && r2 <= 0.5 && r3 > 1, pred(r) = 2;        % D1
    elseif r1 >= 0.6 && r1 <= 2.5 && r2 >= 0.1 && r2 <= 1 && r3 > 2, pred(r) = 3;   % D2
    elseif r1 < 0.1 && r2 > 1 && r3 >= 1 && r3 <= 4, pred(r) = 5;          % T2
    elseif r1 < 0.2 && r2 > 1 && r3 > 4, pred(r) = 6;                       % T3
    elseif r2 > 1 && r3 < 1, pred(r) = 4;                                   % T1
    end
  end
end
