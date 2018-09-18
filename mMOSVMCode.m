%% Required Inputs:
    % data -> matrix with rows corresponding to data points
    % labels -> column vector of class labels
    
%% Finding the number of classes in the dataset
Uc = unique(labels); Uc = Uc(:);
numCls = length(Uc);

%% Converting the dataset to compatable format
kpart = 5; %number of partitions to be made for cross-validation
clearvars -except data labels kpart Uc numCls

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
    clearvars data %remove the dataset if it is high-dimensional
else
    dotProduct = data*data';
end
diag1 = diag(dotProduct);
diag1 = diag1(:); %converting to a column vector
distanceSqMatrix = repmat(abs(diag1),1,n) - 2*dotProduct + repmat(abs(diag1)',n,1);
distanceMatrix = sqrt(distanceSqMatrix);
fprintf('Finished calculating the Dot Product and Distance Matrices.\n');

%% Creating the partitions for k-fold cross-validation (should not be applied to any dataset having less than 'kpart' pts. in any of the classes)
sIze = zeros(1,numCls);
leftover = zeros(1,numCls);
for j = 1:numCls
    sIze(j) = floor(sum(labels==Uc(j))/kpart);
    leftover(j) = sum(labels==Uc(j)) - (kpart * floor(sum(labels==Uc(j))/kpart));
end
sIze = repmat(sIze,kpart,1);
flag = 0;
for j = 1:numCls
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
    for j = 1:numCls
        temp_idx = rand_IDX(yy==Uc(j));
        temp_idx = circshift(temp_idx,[0 -1*sIze_cum(i,j)]);
        part_idx_test = [part_idx_test temp_idx(1:sIze(i,j))];
        part_idx_train = [part_idx_train temp_idx((sIze(i,j)+1):end)];
        y_temp = Uc(j) * ones(sum(yy==Uc(j)),1);
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
clearvars -except data labels kpart X_train Y_train X_test Y_test rand_IDX part_idx distanceSqMatrix distanceMatrix dotProduct numCls Uc

%% Run RBI-LP dual
sigma_gm = [-1, 0.1, 0.5, 1, 5, 10, 50, 100];

numfuncs = numCls + 1;
ip = zeros(1,numfuncs); %ideal point
% declare the number of subproblems
numdirs = 7;
numpoints = numdirs;
maxErrCoeff = 1;
errCoeffs = 0:(maxErrCoeff/(numdirs-1)):maxErrCoeff;
marginCoeffs = 1 + eps - errCoeffs;
squares = errCoeffs.^2 + marginCoeffs.^2;
errCoeffs = errCoeffs./(sqrt(squares)*sqrt(numCls));
marginCoeffs = marginCoeffs./sqrt(squares);
L1 = [marginCoeffs', repmat(errCoeffs',1,numCls)]; %directions generated

gmeans = zeros(1,length(sigma_gm)*numdirs);
auc = zeros(1,length(sigma_gm)*numdirs);
rp = zeros(1,length(sigma_gm)*numdirs);
tps = zeros(numCls,length(sigma_gm)*numdirs);

% Main Loop
for j=1:length(sigma_gm)
    sigma = sigma_gm(j);
    for i=1:numpoints
        for krep = 1:kpart
            % randomly choose the training and testing datasets here
            partIdx = krep;
            Train_y = labels(part_idx{partIdx,1});
            Test_y = labels(part_idx{partIdx,2});
            n = length(Train_y);
            m = length(Test_y);
            cls3 = zeros(m,numCls);

            if (sigma==-1)
                K = dotProduct(part_idx{partIdx,1},part_idx{partIdx,1});                           %kernel matrix between training points
                K_tr_ts = dotProduct(part_idx{partIdx,1},part_idx{partIdx,2});                   %kernel matrix between the training and testing points
            else
                dotProdTr = dotProduct(part_idx{partIdx,1},part_idx{partIdx,1});
                dotProdTrTs = dotProduct(part_idx{partIdx,1},part_idx{partIdx,2});      
                dotProdTs = dotProduct(part_idx{partIdx,2},part_idx{partIdx,2});
                distTr = repmat(diag(dotProdTr),1,length(Train_y)) - 2*dotProdTr + repmat(diag(dotProdTr)',length(Train_y),1);
                distTrTs = repmat(diag(dotProdTr),1,length(Test_y)) - 2*dotProdTrTs + repmat(diag(dotProdTs)',length(Train_y),1);
                K = exp(-distTr/(2*sigma*sigma));               %kernel matrix between training points
                K_tr_ts = exp(-distTrTs/(2*sigma*sigma));       %kernel matrix between the training and testing points
            end

            LB = [repmat([zeros(length(Train_y),1); -inf],numCls,1); zeros(2*length(Train_y),1)];
            UB = inf*ones(size(LB));

            L = L1(i,:);                                %random direction vector
            Lt = repmat(L(1),length(Train_y),1);
            L2 = zeros(1,1);
            Ltt = repmat([Lt; L2],numCls,1);
            L3 = zeros(length(Train_y),1);
            for ii = 1:numCls
                L3(Train_y==Uc(ii)) = L(ii+1);
            end
            L4 = zeros(length(Train_y),1);
            for ii = 1:numCls
                L4(Train_y==Uc(ii)) = L(ii+1);
            end
            f = [Ltt; L3; L4];

            k4 = []; k8 = [];
            for ii = 1:numCls
                train_y = -1*ones(size(Train_y));
                train_y(Train_y==Uc(ii)) = 1;
                k1 = zeros(sum(train_y==1),numCls*(length(Train_y)+1));
                k1(:,((ii-1)*(length(Train_y)+1)+1):(ii*(length(Train_y)+1))) = -1*[repmat(train_y',sum(train_y==1),1).*K(train_y==1,:), ones(sum(train_y==1),1)];
                I1 = -1*eye(length(Train_y));
                k2 = I1(train_y==1,:);
                k3 = zeros(sum(train_y==1),length(Train_y));
                k4 = [k4; k1,k2,k3];
                k5 = zeros(sum(train_y~=1),numCls*(length(Train_y)+1));
                k5(:,((ii-1)*(length(Train_y)+1)+1):(ii*(length(Train_y)+1))) = [repmat(train_y',sum(train_y~=1),1).*K(train_y~=1,:), ones(sum(train_y~=1),1)];
                k6 = zeros(sum(train_y~=1),length(Train_y));
                k7 = I1(train_y~=1,:);
                k8 = [k8; k5,k6,k7];
            end

            l0 = eye(numfuncs)-(L*L');
            l1 = repmat([repmat(l0(:,1),1,length(Train_y)), zeros(numfuncs,1)],1,numCls);
            l2 = zeros(numfuncs,length(Train_y));
            for ii = 1:numCls
                train_y = -1*ones(size(Train_y));
                train_y(Train_y==Uc(ii)) = 1;
                l2(:,train_y==1) = (1/sum(train_y==1)).*repmat(l0(:,(ii+1)),1,sum(train_y==1));
%                 l2(:,train_y==1) = repmat(l0(:,(ii+1)),1,sum(train_y==1));
            end
            l2 = repmat(l2,1,2);
            l3 = [l1, l2];
            A = [k4; k8; l3];

            m1 = -1*ones(numCls*length(Train_y),1);
            m2 = zeros(numfuncs,1);
            b = [m1; m2];

            % running the linear program
            opts = optimset('MaxIter',200);
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
                h = x(1:((length(Train_y)+1)*numCls));
            catch
                x = zeros(size(LB))';
                h = x(1:((length(Train_y)+1)*numCls));
            end
            h = h(:)';

            % finding classification results
            h_reshape = reshape(h',length(Train_y)+1,numCls);
            alphas = h_reshape(1:length(Train_y),:);
            betas = h_reshape(end,:);

            for ii = 1:numCls
                alpha = alphas(:,ii);
                beta = betas(ii);
                train_y = zeros(size(Train_y));
                test_y = zeros(size(Test_y));
                train_y(Train_y==Uc(ii),1)=1;
                train_y(Train_y~=Uc(ii),1)=-1;
                %test_y(Test_y==Uc(ii),1)=1;
                %test_y(Test_y~=Uc(ii),1)=-1;
                cla3 = ((alpha.*train_y)'*K_tr_ts + beta)'; %predicting fx_test values
                cls3(:,ii)=sign(cla3);
            end

            temp = zeros(m,1);
            for jj=1:m
                tempo=find(cls3(jj,:)==1);
                if isempty(tempo)
                    r = randperm(numCls);
                    temp(jj) = Uc(r(1));
                else
                    siz = size(tempo,1);
                    r = randperm(siz);
                    temp(jj) = tempo(r(1));
                end
            end
            pred_y = temp;

            confMat = zeros(numCls,numCls); %confusion matrix
            for jjj = 1:numCls
                idxx = find(Test_y==Uc(jjj));
                confMat(jjj,:) = sum(repmat(Uc',length(idxx),1) == repmat(pred_y(idxx),1,numCls),1);
            end
            fintpr = diag(confMat)'./sum(confMat,2)';
            tps(:,((j-1)*numpoints+i)) = tps(:,((j-1)*numpoints+i)) + fintpr'/kpart;
            finprec = diag(confMat)'./(sum(confMat,1)+eps);
            gmeans((j-1)*numpoints+i) = gmeans((j-1)*numpoints+i) + (nthroot(prod(fintpr),numCls)/kpart);
            finfpr = zeros((numCls-1),numCls);
            for jjj = 1:numCls
               fpr_tmp = confMat(:,jjj)./sum(confMat,2);
               fpr_tmp(jjj) = [];
               finfpr(:,jjj) = fpr_tmp;
            end
            auc((j-1)*numpoints+i) = auc((j-1)*numpoints+i) + (mean(mean((1+repmat(fintpr,size(finfpr,1),1)-finfpr)/2))/kpart);
            rp((j-1)*numpoints+i) = rp((j-1)*numpoints+i) + (mean((finprec+fintpr)/2)/kpart);
        end
    end
    fprintf('Finished running RBI-LP Dual for parameter %d.\n',j);
end
perfLPdual = [gmeans; auc; rp];
clearvars -except data labels kpart Y_train Y_test rand_IDX part_idx distanceSqMatrix distanceMatrix dotProduct numCls Uc sigma_gm ip numfuncs numdirs numpoints... 
    perfLPdual 

%% Display the results
fprintf('Performance:   gm=%d   auc=%d  rp=%d.\n',max(perfLPdual(1,:)),max(perfLPdual(2,:)),max(perfLPdual(3,:)));
