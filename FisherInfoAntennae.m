function res = FisherInfoAntennae(dataset)

pinit = [0 45 0 -0.3];
x = [-90:45:90];
xx = [-90:90];

tuningdata = vertcat(dataset.fly.mean);
tuningvar = vertcat(dataset.fly.var);

n = size(tuningdata,1);

for i=1:n
    tuningtemp = tuningdata;
    tuningtemp(i,:) = [];
    
    tempvar = tuningvar;
    tempvar(i,:) = [];
    
    res.p(i,:) = nlinfit(x,mean(tuningtemp),'gaussline2',pinit);
    res.tc(i,:) = gaussline2(res.p(i,:),xx);
    res.tcslope(i,:) = diff(res.tc(i,:));
    res.std(i) = sqrt(nanmean(tempvar(:)));
    
    res.FI(i,:) = (res.tcslope(i,:)/res.std(i)).^2;
    
end

    res.FIjkmean = mean(res.FI);
    res.FIjkSE = sqrt(((n-1)/n)*var(res.FI));

