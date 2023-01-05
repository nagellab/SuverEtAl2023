D = dir('/Users/knagel/Documents/Data/SuverRevisionData/2022_12_05_reImported_threeMinusTwo_includesFlight/CS_activate/*vel3*.mat')
for i=1:length(D),
    load([D(i).folder,'/',D(i).name]);
    control.fly(i).data = data;
    for j=1:5,
        ind = find(control.fly(i).data(2,:)==j-1);
        control.fly(i).mean(j) = mean(control.fly(i).data(1,ind));
        control.fly(i).var(j) = var(control.fly(i).data(1,ind));
    end; 
end

D = dir('/Users/knagel/Documents/Data/SuverRevisionData/2022_12_05_reImported_threeMinusTwo_includesFlight/18D07_activate/*vel3*.mat')
for i=1:length(D),
    load([D(i).folder,'/',D(i).name]);
    a18D07.fly(i).data = data;
    for j=1:5,
        ind = find(a18D07.fly(i).data(2,:)==j-1);
        a18D07.fly(i).mean(j) = mean(a18D07.fly(i).data(1,ind));
        a18D07.fly(i).var(j) = var(a18D07.fly(i).data(1,ind));
    end; 
end

D = dir('/Users/knagel/Documents/Data/SuverRevisionData/2022_12_05_reImported_threeMinusTwo_includesFlight/74C10_activate/*vel3*.mat')
for i=1:length(D),
    load([D(i).folder,'/',D(i).name]);
    a74C10.fly(i).data = data;
    for j=1:5,
        ind = find(a74C10.fly(i).data(2,:)==j-1);
        a74C10.fly(i).mean(j) = mean(a74C10.fly(i).data(1,ind));
        a74C10.fly(i).var(j) = var(a74C10.fly(i).data(1,ind));
    end; 
end

D = dir('/Users/knagel/Documents/Data/SuverRevisionData/2022_12_05_reImported_threeMinusTwo_includesFlight/91F02_activate/*vel3*.mat')
for i=1:length(D),
    load([D(i).folder,'/',D(i).name]);
    a91F02.fly(i).data = data;
    for j=1:5,
        ind = find(a91F02.fly(i).data(2,:)==j-1);
        a91F02.fly(i).mean(j) = mean(a91F02.fly(i).data(1,ind));
        a91F02.fly(i).var(j) = var(a91F02.fly(i).data(1,ind));
    end; 
end