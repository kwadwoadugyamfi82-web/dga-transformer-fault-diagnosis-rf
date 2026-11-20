%% ========================================================================
%  DGA TRANSFORMER FAULT DIAGNOSIS
%  RANDOM FOREST vs KNN vs SVM vs ANN — FULL COMPARISON
% ========================================================================
%  Dataset : DGA_transformer_dataset.csv (589 samples, cleanest source)
%  Classes : PD, D1, D2, T1, T2, T3 (IEC 60599 fault codes)
%
%  All 4 models are trained/tested on the EXACT SAME split and the EXACT
%  SAME engineered features, so the comparison is fair. Every figure this
%  script produces is meant to go straight into your presentation slides:
%
%   Fig 1  - OOB error curve (Random Forest convergence)
%   Fig 2  - Random Forest confusion matrix
%   Fig 3  - Random Forest feature importance
%   Fig 4  - KNN: accuracy vs K (shows how K=... was chosen)
%   Fig 5  - KNN confusion matrix
%   Fig 6  - SVM confusion matrix
%   Fig 7  - ANN training performance curve
%   Fig 8  - ANN confusion matrix
%   Fig 9  - Bar chart: overall test accuracy, all 4 models side by side
%   Fig 10 - Grouped bar chart: macro F1-score, all 4 models side by side
% ========================================================================

clear; clc; close all;
rng(42);

%% 1. LOAD DATA
T = readtable('DGA_transformer_dataset.csv');
gasNames = {'H2_ppm','CH4_ppm','C2H6_ppm','C2H4_ppm','C2H2_ppm'};
X_raw    = T{:, gasNames};
Y        = categorical(T.Fault_Type);
classList = categories(Y);

%% 2. FEATURE ENGINEERING (same set used for the Random Forest report)
eps_ = 1e-4;
H2=X_raw(:,1); CH4=X_raw(:,2); C2H6=X_raw(:,3); C2H4=X_raw(:,4); C2H2=X_raw(:,5);
R1=CH4./(H2+eps_); R2=C2H2./(C2H4+eps_); R3=C2H4./(C2H6+eps_); R4=C2H2./(CH4+eps_);
R5=C2H2./(H2+eps_); R6=C2H6./(C2H2+eps_);
TCG = H2+CH4+C2H6+C2H4+C2H2;
Dsum = CH4+C2H4+C2H2+eps_;
D1=CH4./Dsum*100; D2=C2H4./Dsum*100; D3=C2H2./Dsum*100;
X_log = log1p(X_raw);
featureNames = [strcat('log_', gasNames), ...
    {'R_CH4_H2','R_C2H2_C2H4','R_C2H4_C2H6','R_C2H2_CH4','R_C2H2_H2','R_C2H6_C2H2', ...
     'TCG','Duval_CH4pct','Duval_C2H4pct','Duval_C2H2pct'}];
X = [X_log, R1, R2, R3, R4, R5, R6, TCG, D1, D2, D3];

%% 3. ONE SHARED TRAIN/TEST SPLIT FOR ALL 4 MODELS (fair comparison)
cvOuter = cvpartition(Y, 'HoldOut', 0.20, 'Stratify', true);
Xtrain = X(training(cvOuter),:); Ytrain = Y(training(cvOuter));
Xtest  = X(test(cvOuter),:);     Ytest  = Y(test(cvOuter));
fprintf('Train: %d samples | Test: %d samples\n', numel(Ytrain), numel(Ytest));

% Standardize features (needed for KNN/SVM/ANN to work well; trees don't
% need it, but standardizing does not hurt Random Forest either, so we
% use the SAME scaled data for every model to keep things simple/fair)
mu = mean(Xtrain); sigma = std(Xtrain); sigma(sigma==0) = 1;
Xtrain_s = (Xtrain - mu) ./ sigma;
Xtest_s  = (Xtest  - mu) ./ sigma;

% Balance the training set with SMOTE-style oversampling (same as before)
[Xtrain_bal, Ytrain_bal] = smoteBalance(Xtrain_s, Ytrain, 5);

results = {};

%% ========================================================================
%  MODEL 1: RANDOM FOREST
%% ========================================================================
fprintf('\n--- Training Random Forest ---\n');
rfModel = TreeBagger(500, Xtrain_bal, Ytrain_bal, 'Method','classification', ...
    'MinLeafSize', 1, 'NumPredictorsToSample', 5, 'Prior','Uniform', ...
    'OOBPrediction','on', 'OOBPredictorImportance','on', 'PredictorNames', featureNames);

figure('Name','Fig1_RF_OOBError');
plot(oobError(rfModel), 'LineWidth', 1.5);
xlabel('Number of Grown Trees'); ylabel('Out-of-Bag Error');
title('Random Forest: OOB Error Convergence'); grid on;

rfPred = predict(rfModel, Xtest_s);
results{end+1} = evalModel('Random Forest', Ytest, rfPred, classList);

figure('Name','Fig2_RF_ConfMat');
cmRF = confusionchart(Ytest, categorical(rfPred));
cmRF.Title = sprintf('Random Forest Confusion Matrix (Acc = %.2f%%)', results{end}.Accuracy);
cmRF.RowSummary='row-normalized'; cmRF.ColumnSummary='column-normalized';

imp = rfModel.OOBPermutedPredictorDeltaError;
[impSorted, idxSorted] = sort(imp,'descend');
figure('Name','Fig3_RF_FeatureImportance');
barh(impSorted); set(gca,'YTick',1:numel(featureNames),'YTickLabel',featureNames(idxSorted),'YDir','reverse');
xlabel('OOB Predictor Importance'); title('Random Forest Feature Importance'); grid on;

%% ========================================================================
%  MODEL 2: K-NEAREST NEIGHBORS (KNN) - tuned K via cross-validation
%% ========================================================================
fprintf('\n--- Training KNN (tuning K) ---\n');
kRange = 1:2:25;
cvAcc = zeros(size(kRange));
cvK = cvpartition(Ytrain_bal, 'KFold', 5, 'Stratify', true);
for i = 1:numel(kRange)
    k = kRange(i);
    accFold = zeros(cvK.NumTestSets,1);
    for f = 1:cvK.NumTestSets
        mdl = fitcknn(Xtrain_bal(training(cvK,f),:), Ytrain_bal(training(cvK,f)), 'NumNeighbors', k);
        pred = predict(mdl, Xtrain_bal(test(cvK,f),:));
        accFold(f) = sum(pred==Ytrain_bal(test(cvK,f)))/numel(pred);
    end
    cvAcc(i) = mean(accFold);
end
[~, bestIdx] = max(cvAcc);
bestK = kRange(bestIdx);
fprintf('Best K found = %d (CV accuracy = %.2f%%)\n', bestK, cvAcc(bestIdx)*100);

figure('Name','Fig4_KNN_AccuracyVsK');
plot(kRange, cvAcc*100, '-o', 'LineWidth', 1.5);
xlabel('Number of Neighbors (K)'); ylabel('Cross-Validated Accuracy (%)');
title('KNN: Accuracy vs K'); grid on;
hold on; plot(bestK, cvAcc(bestIdx)*100, 'r*', 'MarkerSize', 12); hold off;
legend('CV Accuracy', sprintf('Best K = %d', bestK), 'Location','best');

knnModel = fitcknn(Xtrain_bal, Ytrain_bal, 'NumNeighbors', bestK);
knnPred = predict(knnModel, Xtest_s);
results{end+1} = evalModel('KNN', Ytest, knnPred, classList);

figure('Name','Fig5_KNN_ConfMat');
cmKNN = confusionchart(Ytest, categorical(knnPred));
cmKNN.Title = sprintf('KNN Confusion Matrix (K=%d, Acc = %.2f%%)', bestK, results{end}.Accuracy);
cmKNN.RowSummary='row-normalized'; cmKNN.ColumnSummary='column-normalized';

%% ========================================================================
%  MODEL 3: SUPPORT VECTOR MACHINE (SVM) - multiclass via ECOC, tuned box constraint
%% ========================================================================
fprintf('\n--- Training SVM ---\n');
t = templateSVM('KernelFunction','rbf', 'Standardize', false, 'BoxConstraint', 10, 'KernelScale','auto');
svmModel = fitcecoc(Xtrain_bal, Ytrain_bal, 'Learners', t, 'Coding', 'onevsone');
svmPred = predict(svmModel, Xtest_s);
results{end+1} = evalModel('SVM', Ytest, svmPred, classList);

figure('Name','Fig6_SVM_ConfMat');
cmSVM = confusionchart(Ytest, categorical(svmPred));
cmSVM.Title = sprintf('SVM (RBF Kernel) Confusion Matrix (Acc = %.2f%%)', results{end}.Accuracy);
cmSVM.RowSummary='row-normalized'; cmSVM.ColumnSummary='column-normalized';

%% ========================================================================
%  MODEL 4: ARTIFICIAL NEURAL NETWORK (ANN) - feedforward pattern net
%% ========================================================================
fprintf('\n--- Training ANN ---\n');
% Convert labels to one-hot for patternnet
YtrainOH = full(ind2vec(double(Ytrain_bal)'));
net = patternnet([20 10]);           % 2 hidden layers: 20 then 10 neurons
net.trainParam.showWindow = false;   % run without popping up GUI
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;

[net, tr] = train(net, Xtrain_bal', YtrainOH);

figure('Name','Fig7_ANN_TrainingPerf');
plotperf(tr);
title('ANN Training Performance (Cross-Entropy vs Epoch)');

annOut = net(Xtest_s');
[~, annPredIdx] = max(annOut, [], 1);
annPred = classList(annPredIdx);
results{end+1} = evalModel('ANN', Ytest, annPred, classList);

figure('Name','Fig8_ANN_ConfMat');
cmANN = confusionchart(Ytest, categorical(annPred));
cmANN.Title = sprintf('ANN Confusion Matrix (Acc = %.2f%%)', results{end}.Accuracy);
cmANN.RowSummary='row-normalized'; cmANN.ColumnSummary='column-normalized';

%% ========================================================================
%  FINAL COMPARISON CHARTS
%% ========================================================================
modelNames = cellfun(@(r) r.Name, results, 'UniformOutput', false);
accs = cellfun(@(r) r.Accuracy, results);
f1s  = cellfun(@(r) r.F1, results) * 100;
precs= cellfun(@(r) r.Precision, results) * 100;
recs = cellfun(@(r) r.Recall, results) * 100;

fprintf('\n\n===================== FINAL COMPARISON TABLE =====================\n');
fprintf('%-15s %-12s %-12s %-12s %-12s\n','Model','Accuracy(%)','Precision(%)','Recall(%)','F1(%)');
for i = 1:numel(results)
    fprintf('%-15s %-12.2f %-12.2f %-12.2f %-12.2f\n', modelNames{i}, accs(i), precs(i), recs(i), f1s(i));
end

figure('Name','Fig9_AccuracyComparison');
bar(categorical(modelNames), accs);
ylabel('Test Accuracy (%)'); title('Overall Test Accuracy — Model Comparison');
ylim([0 100]); grid on;
for i=1:numel(accs)
    text(i, accs(i)+1.5, sprintf('%.2f%%',accs(i)), 'HorizontalAlignment','center','FontWeight','bold');
end

figure('Name','Fig10_MetricsComparison');
metricsMat = [precs; recs; f1s]';
bar(categorical(modelNames), metricsMat, 'grouped');
ylabel('Score (%)'); title('Precision / Recall / F1-Score — Model Comparison');
legend('Precision','Recall','F1-Score','Location','southoutside','Orientation','horizontal');
ylim([0 100]); grid on;

%% SAVE ALL RESULTS + FIGURES
save('Model_Comparison_Results.mat', 'results', 'modelNames', 'accs', 'precs', 'recs', 'f1s', ...
     'bestK', 'featureNames', 'Xtrain_bal', 'Ytrain_bal', 'Xtest_s', 'Ytest');

figHandles = findall(0, 'Type', 'figure');
if ~exist('figures', 'dir'); mkdir('figures'); end
for i = 1:numel(figHandles)
    fh = figHandles(i);
    name = get(fh, 'Name');
    if isempty(name); name = sprintf('figure_%d', i); end
    saveas(fh, fullfile('figures', [name '.png']));
end
fprintf('\nAll figures saved into the "figures" folder as PNG files.\n');
fprintf('Send these PNGs back and we will drop them straight into the presentation.\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  (MATLAB requires local functions in a script file to appear after
%   all script commands - that's why these are down here at the bottom,
%   even though they're called earlier in the script above.)
%% ========================================================================

% Helper to compute + print metrics, returns a results struct
function R = evalModel(modelName, Ytrue, Ypred, classList)
    Ypred = categorical(Ypred);
    acc = sum(Ypred==Ytrue)/numel(Ytrue)*100;
    cm = confusionmat(Ytrue, Ypred, 'Order', classList);
    nC = numel(classList);
    P=zeros(nC,1); Rc=zeros(nC,1); F1=zeros(nC,1);
    for i=1:nC
        TP=cm(i,i); FP=sum(cm(:,i))-TP; FN=sum(cm(i,:))-TP;
        P(i)=TP/max(TP+FP,1); Rc(i)=TP/max(TP+FN,1);
        F1(i)=2*P(i)*Rc(i)/max(P(i)+Rc(i),eps);
    end
    fprintf('\n=== %s ===\n', modelName);
    fprintf('Test Accuracy: %.2f%%\n', acc);
    fprintf('Macro Precision: %.3f | Macro Recall: %.3f | Macro F1: %.3f\n', ...
        mean(P), mean(Rc), mean(F1));
    R = struct('Name', modelName, 'Accuracy', acc, 'ConfMat', cm, ...
        'Precision', mean(P), 'Recall', mean(Rc), 'F1', mean(F1));
end

% SMOTE-style oversampling (balances minority classes)
function [Xb, Yb] = smoteBalance(X, Y, k)
    classes = categories(Y);
    counts  = countcats(Y);
    targetN = max(counts);
    Xb = X; Yb = Y;
    for c = 1:numel(classes)
        idxC = find(Y == classes{c});
        nC = numel(idxC);
        nNeeded = targetN - nC;
        if nNeeded <= 0 || nC < 2, continue; end
        Xc = X(idxC,:);
        kUse = min(k, nC-1);
        synth = zeros(nNeeded, size(X,2));
        for i = 1:nNeeded
            baseIdx = randi(nC);
            dists = sum((Xc - Xc(baseIdx,:)).^2, 2);
            [~, order] = sort(dists);
            neighborPool = order(2:kUse+1);
            neighborIdx = neighborPool(randi(numel(neighborPool)));
            gap = rand();
            synth(i,:) = Xc(baseIdx,:) + gap*(Xc(neighborIdx,:) - Xc(baseIdx,:));
        end
        Xb = [Xb; synth]; %#ok<AGROW>
        Yb = [Yb; repmat(classes(c), nNeeded, 1)]; %#ok<AGROW>
    end
end
