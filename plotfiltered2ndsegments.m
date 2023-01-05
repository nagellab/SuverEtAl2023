function res = plotfiltered2ndsegments(data,cutoff)

% remove bad trials
data = data';

for i=1:size(data,2), 
    m(:,i) = data(:,i)-mean(data(1:100,i)); 
    bad(i) = ~isempty(find(isnan(m(:,i))));
end

res.m = m;

m(:,bad) = [];

% filter and get power spectrum
[bl al] = butter(2,cutoff/60);
[bh ah] = butter(2,cutoff/60,'high');
for i=1:size(m,2), 
    mfiltlow(:,i) = filtfilt(bl,al,m(:,i)); 
    mfilthigh(:,i) = filtfilt(bh,ah,m(:,i)); 
    
    [Ps(:,i) f] = pwelch(m(:,i),256,128,256,60);
end

res.mfiltlow = mfiltlow;
res.mfilthigh = mfilthigh;
res.Ps = Ps;
res.f = f;

figure; set(gcf,'Position',[-99        1555         887         137]);
subplot(1,4,1); plot(mfiltlow);
hold on; plot([120 240],[-18 -18],'r');
box off; set(gca,'TickDir','out');
ylabel('<cutoff all'); set(gca,'Ylim',[-20 10])

subplot(1,4,2); plot(mean(mfiltlow'),'k');
hold on; plot(mean(mfiltlow')+std(mfiltlow')/sqrt(size(m,2)),'k:');
hold on; plot(mean(mfiltlow')-std(mfiltlow')/sqrt(size(m,2)),'k:');
hold on; plot([120 240],[-3 -3],'r');
box off; set(gca,'TickDir','out');
ylabel('<cutoff mean'); set(gca,'Ylim',[-3.5 1.5])

subplot(1,4,3); plot(mfilthigh);
hold on; plot([120 240],[-8 -8],'r');
box off; set(gca,'TickDir','out');
ylabel('>cutoff all'); set(gca,'Ylim',[-11 11])

subplot(1,4,4); plot(mean(abs(mfilthigh)'),'k');
hold on; plot(mean(abs(mfilthigh)')+std(abs(mfilthigh)')/sqrt(size(m,2)),'k:');
hold on; plot(mean(abs(mfilthigh)')-std(abs(mfilthigh)')/sqrt(size(m,2)),'k:');
hold on; plot([120 240],[0.2 0.2],'r');
box off; set(gca,'TickDir','out');
ylabel('>cutoff mean abs'); set(gca,'Ylim',[0 1.8])

setX(80,300);

figure; set(gcf,'Position',[-102        1307         478         168]);

subplot(1,2,1); plot(m);
hold on; plot([120 240],[-22 -22],'r');
box off; set(gca,'TickDir','out');
ylabel('all'); axis([81,300,-25,10])

subplot(1,2,2); loglog(f,mean(Ps'),'b');
hold on; loglog(f,mean(Ps')+std(Ps')/sqrt(size(m,2)),'b:');
hold on; loglog(f,mean(Ps')-std(Ps')/sqrt(size(m,2)),'b:');
hold on; plot([cutoff cutoff],[10^-3 10^2],'k:');
box off; set(gca,'TickDir','out');
axis([0.2 20 10^-3 10^2])