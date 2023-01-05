function res = plot2ndsegaveragesFig1(res)

color = 'kbcm';

figure; set(gcf,'Position',[-99        1555         887         137]);
for i=1:5
    subplot(1,5,i); 
    for j=1:4
            hold on; plot(nanmean(res(j).speed(i).m'),color(j))
            axis([101,300,-5,1]); box off;
            set(gca,'TickDir','out');
    end
end
set(gcf,'PaperPositionMode','auto')

figure; set(gcf,'Position',[-99        1555         887         137]);
for i=1:5
    subplot(1,5,i); 
    for j=1:4
            hold on; plot(mean(abs(res(j).speed(i).mfilthigh')),color(j));
            axis([101,300,0,2]); box off;
            set(gca,'TickDir','out');
    end
end
set(gcf,'PaperPositionMode','auto')

