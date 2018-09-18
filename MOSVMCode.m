%% Required Inputs:
    % data -> matrix with rows corresponding to data points
    % labels -> column vector of class labels
    
%% Converting the dataset to compatable format
kpart = 5; %number of partitions to be made for cross-validation
clearvars -except data labels kpart

%% Calculating the dot product and distance matrices
[n,d] = size(data);
dotPart = [];
dotProduct = [];
% distanceMatrix=zeros(n,n);
% for i=1:n
%     distanceMatrix(:,i) = sum((repmat(data(i,:),n,1) - data).^2,2);
% end
% distanceMatrix = distanceMatrix';
if (n*d > 50000)
    numrows = ceil(50000/d);
    remrows = rem(n,numrows);
    itercount = ((n - remrows)/numrows) + 1;
    for i = 1:itercount
        dotPart = [];
        for j = 1:itercount
            if ((i==itercount)&&(j~=itercount))
                a = (i-1)*numrows + 1;
                b = (i-1)*numrows + remrows;
                c = (j-1)*numrows + 1;
                d = j*numrows;
                dotP = data(a:b,:)*data(c:d,:)';
            elseif ((i~=itercount)&&(j==itercount))
                a = (i-1)*numrows + 1;
                b = i*numrows;
                c = (j-1)*numrows + 1;
                d = (j-1)*numrows + remrows;
                dotP = data(a:b,:)*data(c:d,:)';
            elseif ((i==itercount)&&(j==itercount))
                a = (i-1)*numrows + 1;
                b = (i-1)*numrows + remrows;
                c = (j-1)*numrows + 1;
                d = (j-1)*numrows + remrows;
                dotP = data(a:b,:)*data(c:d,:)';
            else
                a = (i-1)*numrows + 1;
                b = i*numrows;
                c = (j-1)*numrows + 1;
                d = j*numrows;
                dotP = data(a:b,:)*data(c:d,:)';
            end
            dotPart = [dotPart, dotP];
        end
        dotProduct = [dotProduct; dotPart];
    end
    if (n*d > 1000000)
        data = []; %remove the dataset if it is high-dimensional
    end
else
    dotProduct = data*data';
end
diag1 = diag(dotProduct);
diag1 = diag1(:); %converting to a column vector
distanceSqMatrix = repmat(abs(diag1),1,n) - 2*dotProduct + repmat(abs(diag1)',n,1);
distanceMatrix = sqrt(distanceSqMatrix);
fprintf('Finished calculating the Dot Product and Distance Matrices.\n');

%% Creating the partitions for k-fold cross-validation (should not be applied to any dataset having less than 'kpart' pts. in any of the classes)
uniq = unique(labels);
cls_num = length(uniq);
sIze = zeros(1,cls_num);
leftover = zeros(1,cls_num);
for j = 1:cls_num
    sIze(j) = floor(sum(labels==uniq(j))/kpart);
    leftover(j) = sum(labels==uniq(j)) - (kpart * floor(sum(labels==uniq(j))/kpart));
end
sIze = repmat(sIze,kpart,1);
flag = 0;
for j = 1:cls_num
    if flag==0
        sIze_idx = 1;
    else
        sIze_idx = kpart;
    end
    while leftover(j) > 0
        sIze(sIze_idx,j) = sIze(sIze_idx,j) + 1;
        leftover(j) = leftover(j) - 1;
        if flag==0
            sIze_idx = sIze_idx + 1;
        else
            sIze_idx = sIze_idx - 1;
        end
    end
    flag = ~flag;
end
sIze_cum = cumsum(sIze);
sIze_cum = circshift(sIze_cum,1);

rand_IDX = randperm(length(labels));
yy = labels(rand_IDX);

Y_train = cell(kpart,1);
Y_test = cell(kpart,1);
part_idx = cell(kpart,2);

for i = 1:kpart
    
    % extracting the training and test sets for a particular partition
    y_test = []; y_train = []; part_idx_test = []; part_idx_train = [];
    for j = 1:cls_num
        temp_idx = rand_IDX(yy==uniq(j));
        temp_idx = circshift(temp_idx,[0 -1*sIze_cum(i,j)]);
        part_idx_test = [part_idx_test temp_idx(1:sIze(i,j))];
        part_idx_train = [part_idx_train temp_idx((sIze(i,j)+1):end)];
        y_temp = uniq(j) * ones(sum(yy==uniq(j)),1);
        y_test = [y_test; y_temp(1:sIze(i,j))];
        y_train = [y_train; y_temp((sIze(i,j)+1):end)];
    end
    
    %storing x_train, y_train, x_test, y_test, D_train, and D_test here for later use
    part_idx{i,1} = part_idx_train;
    part_idx{i,2} = part_idx_test;
    Y_train{i}=y_train;
    Y_test{i}=y_test;
    
end
fprintf('Finished creating partitions for %d-fold cross-validation.\n',kpart);
clearvars -except data labels kpart X_train Y_train X_test Y_test rand_IDX part_idx distanceSqMatrix distanceMatrix dotProduct d cls_num

%% RBI-LP dual
sigma_gm = [-1, 0.1, 0.5, 1, 5, 10, 50, 100];

ip = [0, 0, 0];
numfuncs = 3;
% declare the number of subproblems
numdirs = 7;
numpoints = numdirs;
maxErrCoeff = 1;
errCoeffs = 0:(maxErrCoeff/(numdirs-1)):maxErrCoeff;
marginCoeffs = 1 + eps - errCoeffs;
squares = errCoeffs.^2 + marginCoeffs.^2;
errCoeffs = errCoeffs./(sqrt(squares)*sqrt(cls_num));
marginCoeffs = marginCoeffs./sqrt(squares);
L1 = [marginCoeffs', repmat(errCoeffs',1,cls_num)]; %directions generated 
%plotting the directions
figure(1)
scatter3(L1(:,1),L1(:,2),L1(:,3),'b*');

tp = zeros(1,length(sigma_gm)*numdirs);
fp = zeros(1,length(sigma_gm)*numdirs);
tn = zeros(1,length(sigma_gm)*numdirs);
fn = zeros(1,length(sigma_gm)*numdirs);
gmeans = zeros(1,length(sigma_gm)*numdirs);
auc = zeros(1,length(sigma_gm)*numdirs);
fmeasure = zeros(1,length(sigma_gm)*numdirs);

% Main Loop
for j=1:length(sigma_gm)
    sigma = sigma_gm(j);
    for i=1:numpoints
        for krep = 1:kpart
            % randomly choose the training and testing datasets here
            partIdx = krep;
            train_y = labels(part_idx{partIdx,1});
            test_y = labels(part_idx{partIdx,2});

            if (sigma==-1)
                K = dotProduct(part_idx{partIdx,1},part_idx{partIdx,1});                           %kernel matrix between training points
                K_tr_ts = dotProduct(part_idx{partIdx,1},part_idx{partIdx,2});                   %kernel matrix between the training and testing points
            else
                dotProdTr = dotProduct(part_idx{partIdx,1},part_idx{partIdx,1});
                dotProdTrTs = dotProduct(part_idx{partIdx,1},part_idx{partIdx,2});      
                dotProdTs = dotProduct(part_idx{partIdx,2},part_idx{partIdx,2});
                distTr = repmat(diag(dotProdTr),1,length(train_y)) - 2*dotProdTr + repmat(diag(dotProdTr)',length(train_y),1);
                distTrTs = repmat(diag(dotProdTr),1,length(test_y)) - 2*dotProdTrTs + repmat(diag(dotProdTs)',length(train_y),1);
                K = exp(-distTr/(2*sigma*sigma));               %kernel matrix between training points
                K_tr_ts = exp(-distTrTs/(2*sigma*sigma));       %kernel matrix between the training and testing points
            end

            LB = [zeros(length(train_y),1); -inf; zeros(length(train_y),1)];
            UB = [inf*ones(length(train_y),1); inf; inf*ones(length(train_y),1)];

            L = L1(i,:);                                %random direction vector
            Lt = repmat(L(1),length(train_y),1);
            L3 = zeros(length(train_y),1);
            L3(train_y==1) = L(2);
            L3(train_y==-1) = L(3);
            L2 = zeros(1,1);
            f = [Lt; L2; L3];

            k1 = eye(length(train_y));
            k2 = (train_y*train_y').*K;
            k3 = train_y;
            k4 = -1*[k2, k3, k1];
            l0 = eye(3)-(L*L');
            l1 = repmat(l0(:,1),1,length(train_y));
            l2 = zeros(3,length(train_y));
            l2(:,(train_y==1)) = (1/sum(train_y==1)).*repmat(l0(:,2),1,sum(train_y==1)); %normalized errors
            l2(:,(train_y==-1)) = (1/sum(train_y==-1)).*repmat(l0(:,3),1,sum(train_y==-1));
%             l2(:,(train_y==1)) = repmat(l0(:,2),1,sum(train_y==1));
%             l2(:,(train_y==-1)) = repmat(l0(:,3),1,sum(train_y==-1));
            l3 = zeros(3,1);
            l4 = [l1, l3, l2];
            A = [k4; l4];

            m1 = -1*ones(length(train_y),1);
            m2 = zeros(3,1);
            b = [m1; m2];

            % running the linear program
            opts = optimset('MaxIter',500);
            problem.f = f;
            problem.Aineq = A;
            problem.bineq = b;
            problem.Aeq = [];
            problem.beq = [];
            problem.lb = LB;
            problem.ub = UB;
            problem.options = opts;
            problem.solver = 'linprog';
            x = linprog(problem);
%             H = 0*eye(length(f));            
%             x = quadprog(H,f,A,b,[],[],LB,UB);
            try
                beta = x(end);
                alpha = x(1:length(train_y));
            catch
                x = zeros(size(LB))';
                beta = x(end);
                alpha = x(1:length(train_y));
            end

            % finding classification results
            alpha = alpha(:);
            pred_y = sign(alpha'.*train_y'*K_tr_ts + beta);
            posMask = (test_y==1)'; negMask = ~posMask;
            out_posMask = (pred_y==1); out_negMask = ~out_posMask;
            truePos = (posMask==1) & (posMask==out_posMask);
            falsePos = (negMask==1) & (negMask~=out_negMask);
            trueNeg = (negMask==1) & (negMask==out_negMask);
            falseNeg = (posMask==1) & (posMask~=out_posMask);
            tp((j-1)*numpoints+i) = tp((j-1)*numpoints+i) + (sum(truePos)/kpart);
            fp((j-1)*numpoints+i) = fp((j-1)*numpoints+i) + (sum(falsePos)/kpart);
            tn((j-1)*numpoints+i) = tn((j-1)*numpoints+i) + (sum(trueNeg)/kpart);
            fn((j-1)*numpoints+i) = fn((j-1)*numpoints+i) + (sum(falseNeg)/kpart);
            tpr = sum(truePos)/(sum(truePos)+sum(falseNeg));
            tnr = sum(trueNeg)/(sum(trueNeg)+sum(falsePos));
            prec = sum(truePos)/(sum(truePos)+sum(falsePos)+eps);
            gmeans((j-1)*numpoints+i) = gmeans((j-1)*numpoints+i) + (sqrt(tpr.*tnr)/kpart);
            auc((j-1)*numpoints+i) = auc((j-1)*numpoints+i) + ((tpr+tnr)/(2*kpart));
            fmeasure((j-1)*numpoints+i) = fmeasure((j-1)*numpoints+i) + ((2*prec.*tpr)./((tpr+prec+eps)*kpart));
            % errorPos((j-1)*numpoints+i) = sum(falseNeg);
            % errorNeg((j-1)*numpoints+i) = sum(falsePos);
        end
    end
    fprintf('Finished running RBI-LP Dual for parameter %d.\n',j);
end
perfLPdual = [tp; fp; tn; fn; gmeans; auc; fmeasure];
clearvars -except data labels kpart Y_train Y_test rand_IDX part_idx distanceSqMatrix distanceMatrix dotProduct d details_svm sigma_gm ip numfuncs numdirs... 
    numpoints perfLPdual 

%% Display Best Results
fprintf('Performance:   gm=%d   auc=%d  fm=%d.\n',max(perfLPdual(5,:)),max(perfLPdual(6,:)),max(perfLPdual(7,:)));
