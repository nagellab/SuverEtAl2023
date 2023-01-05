function res = plotfiltered2ndsegmentscomparison(datalist,cutoff)

color = 'kg';

for n=1:length(datalist)
    
    % remove bad trials
    data = datalist(n).data';

    for i=1:size(data,2), 
        m(:,i) = data(:,i)-mean(data(1:100,i)); 
        bad(i) = ~isempty(find(isnan(m(:,i))));
    end

    res(n).m = m;

    m(:,bad) = [];

    % filter and get power spectrum
    [bl al] = butter(2,cutoff/60);
    [bh ah] = butter(2,cutoff/60,'high');
    for i=1:size(m,2), 
        mfiltlow(:,i) = filtfilt(bl,al,m(:,i)); 
        mfilthigh(:,i) = filtfilt(bh,ah,m(:,i)); 

        [Ps(:,i) f] = pwelch(m(:,i),256,128,256,60);
    end

    res(n).mfiltlow = mfiltlow;
    res(n).mfilthigh = mfilthigh;
    res(n).Ps = Ps;
    res(n).f = f;
    
    clear data m bad mfilthigh mfiltlow Ps
end


figure; set(gcf,'Position',[-99        1555         887         137]);
subplot(1,4,1); 
length(datalist)
for n=length(datalist)
    plot(res(n).mfiltlow);
end
hold on; plot([120 240],[-28 -28],'r');
box off; set(gca,'TickDir','out');
ylabel('<cutoff all'); set(gca,'Ylim',[-30 10])

subplot(1,4,2); 
for n=1:length(datalist)
    plot(mean(res(n).mfiltlow'),color(n));
    hold on; plot(mean(res(n).mfiltlow')+std(res(n).mfiltlow')/sqrt(size(res(n).mfiltlow,2)),[color(n),':']);
    hold on; plot(mean(res(n).mfiltlow')-std(res(n).mfiltlow')/sqrt(size(res(n).mfiltlow,2)),[color(n),':']);
end
hold on; plot([120 240],[-4 -4],'r');
box off; set(gca,'TickDir','out');
ylabel('<cutoff mean'); set(gca,'Ylim',[-4.5 2])

subplot(1,4,3); plot(res(n).mfilthigh);
hold on; plot([120 240],[-18 -18],'r');
box off; set(gca,'TickDir','out');
ylabel('>cutoff all'); set(gca,'Ylim',[-20 20])

subplot(1,4,4); 
for n=1:length(datalist)
    plot(mean(abs(res(n).mfilthigh)'),color(n));
    hold on; plot(mean(abs(res(n).mfilthigh)')+std(abs(res(n).mfilthigh)')/sqrt(size(res(n).mfilthigh,2)),[color(n),':']);
    hold on; plot(mean(abs(res(n).mfilthigh)')-std(abs(res(n).mfilthigh)')/sqrt(size(res(n).mfilthigh,2)),[color(n),':']);
end
hold on; plot([120 240],[0.2 0.2],'r');
box off; set(gca,'TickDir','out');
ylabel('>cutoff mean abs'); set(gca,'Ylim',[0 2.5])

setX(80,300);

