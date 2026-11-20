%% ========================================================================
%  DGA-BASED POWER TRANSFORMER FAULT DIAGNOSIS USING RANDOM FOREST (v2)
%  Improved version: SMOTE-style oversampling + repeated CV + deeper
%  hyperparameter search, aimed at pushing test accuracy toward 90%+.
% ========================================================================
%  Dataset : DGA_transformer_dataset.csv (589 samples, 5 dissolved gases)
%  Source  : Enwen Li (2019), IEEE DataPort, doi:10.21227/h8g0-8z59
%  Classes : PD, D1, D2, T1, T2, T3  (IEC 60599 fault codes)
%
%  WHAT CHANGED FROM v1 (and why it should raise accuracy):
%   1. SMOTE-style oversampling of minority classes (T2=60, PD=74) so the
%      forest sees a balanced number of examples per class - this alone
%      typically gives the biggest accuracy jump on imbalanced DGA data.
%      (Oversampling is done ONLY inside each training fold, never on
%      test data, so the reported accuracy stays honest / no data leakage.)
%   2. A wider, deeper hyperparameter grid (more tree counts, leaf sizes,
%      and feature-subsampling options).
%   3. Repeated Stratified K-Fold Cross-Validation (5 folds x 4 repeats =
%      20 runs) instead of one single 80/20 split. With only 589 samples,
%      a single split is noisy - averaging over many splits gives a much
%      more reliable, and usually HIGHER and more stable, accuracy number.
%   4. Extra engineered features (more Duval/Rogers-style ratios) that
%      several published DGA-RF papers report as accuracy boosters.
% ========================================================================

clear; clc; close all;
rng(42);

%% 1. LOAD DATA
T = readtable('DGA_transformer_dataset.csv');
gasNames = {'H2_ppm','CH4_ppm','C2H6_ppm','C2H4_ppm','C2H2_ppm'};
X_raw    = T{:, gasNames};
Y        = categorical(T.Fault_Type);

fprintf('Total samples: %d\n', height(T));
summary(Y)

%% 2. FEATURE ENGINEERING (expanded ratio set)
eps_ = 1e-4;
H2   = X_raw(:,1); CH4  = X_raw(:,2); C2H6 = X_raw(:,3);
C2H4 = X_raw(:,4); C2H2 = X_raw(:,5);

R1  = CH4  ./ (H2   + eps_);   % CH4/H2      (Rogers)
R2  = C2H2 ./ (C2H4 + eps_);   % C2H2/C2H4   (Rogers/IEC)
R3  = C2H4 ./ (C2H6 + eps_);   % C2H4/C2H6   (Rogers/IEC)
R4  = C2H2 ./ (CH4  + eps_);   % C2H2/CH4
R5  = C2H2 ./ (H2   + eps_);   % C2H2/H2     (Dornenburg)
R6  = C2H6 ./ (C2H2 + eps_);   % C2H6/C2H2   (Dornenburg)
TCG = H2 + CH4 + C2H6 + C2H4 + C2H2;
% Duval-triangle-style relative percentages (gas / sum of the 3 key gases)
DuvalSum = CH4 + C2H4 + C2H2 + eps_;
D_CH4 = CH4  ./ DuvalSum * 100;
D_C2H4= C2H4 ./ DuvalSum * 100;
D_C2H2= C2H2 ./ DuvalSum * 100;

X_log = log1p(X_raw);
featureNames = [strcat('log_', gasNames), ...
    {'R_CH4_H2','R_C2H2_C2H4','R_C2H4_C2H6','R_C2H2_CH4','R_C2H2_H2','R_C2H6_C2H2', ...
     'TCG','Duval_CH4pct','Duval_C2H4pct','Duval_C2H2pct'}];
X = [X_log, R1, R2, R3, R4, R5, R6, TCG, D_CH4, D_C2H4, D_C2H2];

fprintf('Total engineered features: %d\n', size(X,2));

%% 3. HOLD OUT A FINAL, UNTOUCHED TEST SET (20%) - for the honest final number
cvOuter = cvpartition(Y, 'HoldOut', 0.20, 'Stratify', true);
Xtrainval = X(training(cvOuter), :);  Ytrainval = Y(training(cvOuter));
Xtest     = X(test(cvOuter), :);      Ytest     = Y(test(cvOuter));
fprintf('Train+Val samples: %d | Held-out Test samples: %d\n', numel(Ytrainval), numel(Ytest));

%% 4. SMOTE-STYLE OVERSAMPLING FUNCTION (defined at bottom of script)
% Balances classes by generating synthetic samples: for each minority
% sample, pick its nearest same-class neighbor and create a new point
% somewhere between them. This is safer than plain duplication because
% it doesn't just copy-paste identical rows (which would let the forest
% "memorize" a duplicate rather than learn a general pattern).

%% 5. HYPERPARAMETER SEARCH USING REPEATED STRATIFIED K-FOLD CV
numTreesGrid    = [200, 300, 500, 700];
minLeafGrid     = [1, 2, 3, 5];
numFeaturesGrid = [3, 5, 7, size(X,2)];   % try several feature-subsample sizes
kFolds          = 5;
nRepeats        = 4;

bestCVAcc  = -Inf;
bestParams = struct('NumTrees',[],'MinLeaf',[],'NumFeat',[]);
resultsLog = {};

fprintf('\n--- Hyperparameter search with SMOTE + repeated CV (please wait) ---\n');
for nt = numTreesGrid
  for ml = minLeafGrid
    for nf = numFeaturesGrid
      accAll = zeros(kFolds*nRepeats,1);
      run = 0;
      for rep = 1:nRepeats
        cvInner = cvpartition(Ytrainval, 'KFold', kFolds, 'Stratify', true);
        for f = 1:kFolds
          run = run + 1;
          Xtr = Xtrainval(training(cvInner,f), :); Ytr = Ytrainval(training(cvInner,f));
          Xva = Xtrainval(test(cvInner,f), :);     Yva = Ytrainval(test(cvInner,f));

          % Oversample ONLY the training fold (never the validation fold)
          [Xtr_bal, Ytr_bal] = smoteBalance(Xtr, Ytr, 5);

          mdl = TreeBagger(nt, Xtr_bal, Ytr_bal, ...
              'Method','classification', 'MinLeafSize', ml, ...
              'NumPredictorsToSample', nf, 'Prior','Uniform');

          pred = categorical(predict(mdl, Xva));
          accAll(run) = sum(pred == Yva) / numel(Yva);
        end
      end
      meanAcc = mean(accAll);
      resultsLog(end+1,:) = {nt, ml, nf, meanAcc}; %#ok<SAGROW>
      if meanAcc > bestCVAcc
        bestCVAcc = meanAcc;
        bestParams.NumTrees = nt; bestParams.MinLeaf = ml; bestParams.NumFeat = nf;
      end
    end
  end
end

fprintf('\nBest hyperparameters (by %d-fold x %d-repeat CV):\n', kFolds, nRepeats);
fprintf('  NumTrees=%d | MinLeafSize=%d | NumPredictorsToSample=%d\n', ...
    bestParams.NumTrees, bestParams.MinLeaf, bestParams.NumFeat);
fprintf('  Cross-validated accuracy = %.2f%%\n', bestCVAcc*100);

%% 6. TRAIN FINAL MODEL: oversample the FULL train+val set, then fit
[Xtrainval_bal, Ytrainval_bal] = smoteBalance(Xtrainval, Ytrainval, 5);
fprintf('\nBalanced training set size after SMOTE: %d (was %d)\n', ...
    numel(Ytrainval_bal), numel(Ytrainval));
summary(Ytrainval_bal)

finalModel = TreeBagger(bestParams.NumTrees, Xtrainval_bal, Ytrainval_bal, ...
    'Method','classification', 'MinLeafSize', bestParams.MinLeaf, ...
    'NumPredictorsToSample', bestParams.NumFeat, 'Prior','Uniform', ...
    'OOBPrediction','on', 'OOBPredictorImportance','on', ...
    'PredictorNames', featureNames);

%% 7. EVALUATE ON THE UNTOUCHED, ORIGINAL (NOT oversampled) TEST SET
predLabels = categorical(predict(finalModel, Xtest));
testAccuracy = sum(predLabels == Ytest) / numel(Ytest) * 100;

fprintf('\n=== FINAL HELD-OUT TEST RESULTS ===\n');
fprintf('Cross-validated accuracy estimate : %.2f%%\n', bestCVAcc*100);
fprintf('Final held-out test accuracy      : %.2f%%\n', testAccuracy);

figure;
cm = confusionchart(Ytest, predLabels);
cm.Title = sprintf('Random Forest (SMOTE + tuned) - Test Accuracy = %.2f%%', testAccuracy);
cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';

confMat = confusionmat(Ytest, predLabels);
classList = categories(Ytest);
fprintf('\n%-6s %-10s %-10s %-10s %-8s\n','Class','Precision','Recall','F1-Score','Support');
macroP=0; macroR=0; macroF1=0; nC=numel(classList);
for i = 1:nC
    TP = confMat(i,i); FP = sum(confMat(:,i))-TP; FN = sum(confMat(i,:))-TP;
    precision = TP/max(TP+FP,1); recall = TP/max(TP+FN,1);
    f1 = 2*precision*recall/max(precision+recall,eps);
    fprintf('%-6s %-10.3f %-10.3f %-10.3f %-8d\n', classList{i}, precision, recall, f1, sum(confMat(i,:)));
    macroP=macroP+precision; macroR=macroR+recall; macroF1=macroF1+f1;
end
fprintf('\nMacro-average Precision: %.3f | Recall: %.3f | F1: %.3f\n', macroP/nC, macroR/nC, macroF1/nC);

%% 8. FEATURE IMPORTANCE
imp = finalModel.OOBPermutedPredictorDeltaError;
[impSorted, idxSorted] = sort(imp, 'descend');
figure;
barh(impSorted);
set(gca,'YTick',1:numel(featureNames),'YTickLabel',featureNames(idxSorted),'YDir','reverse');
xlabel('OOB Predictor Importance'); title('Feature Importance - Random Forest (SMOTE + tuned)');
grid on;

%% 9. SAVE MODEL AND RESULTS
save('RandomForest_DGA_Model_v2.mat', 'finalModel', 'bestParams', 'bestCVAcc', ...
     'testAccuracy', 'confMat', 'classList', 'featureNames', ...
     'Xtrainval_bal','Ytrainval_bal','Xtest','Ytest');
fprintf('\nSaved to RandomForest_DGA_Model_v2.mat\n');

%% ========================================================================
%  LOCAL FUNCTION: SMOTE-style oversampling
%  Balances every class up to the size of the largest class by creating
%  synthetic points between each minority sample and one of its nearest
%  same-class neighbors (classic SMOTE idea, implemented from scratch -
%  no extra toolbox required).
%% ========================================================================
function [Xb, Yb] = smoteBalance(X, Y, k)
    classes = categories(Y);
    counts  = countcats(Y);
    targetN = max(counts);

    Xb = X; Yb = Y;
    for c = 1:numel(classes)
        idxC = find(Y == classes{c});
        nC   = numel(idxC);
        nNeeded = targetN - nC;
        if nNeeded <= 0 || nC < 2
            continue; % already the majority class, or too few points to interpolate
        end
        Xc = X(idxC, :);
        kUse = min(k, nC-1);
        synth = zeros(nNeeded, size(X,2));
        for i = 1:nNeeded
            baseIdx = randi(nC);
            dists = sum((Xc - Xc(baseIdx,:)).^2, 2);
            [~, order] = sort(dists);
            neighborPool = order(2:kUse+1);          % exclude itself (distance 0)
            neighborIdx  = neighborPool(randi(numel(neighborPool)));
            gap = rand();
            synth(i,:) = Xc(baseIdx,:) + gap * (Xc(neighborIdx,:) - Xc(baseIdx,:));
        end
        Xb = [Xb; synth]; %#ok<AGROW>
        Yb = [Yb; repmat(classes(c), nNeeded, 1)]; %#ok<AGROW>
    end
end
