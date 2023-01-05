function res = plotFIantennae()

%load FI_Marie control a18D07 a74C10 a91F02
load 'Marie antennal Behavior'/Fig4updates control a18D07 a74C10 a91F02

res(1) = FisherInfoAntennae(control);
res(2) = FisherInfoAntennae(a18D07);
res(3) = FisherInfoAntennae(a74C10);
res(4) = FisherInfoAntennae(a91F02);



figure;
set(gcf,'PaperPositionMode','auto');
set(gcf,'Position',[123   404   806   293]);

x = [-90:45:90];
xx = [-90:90];
xd = [-90:89];

% plot tuning curves and fits
subplot(1,2,1);
plot(x,vertcat(control.fly.mean)','k.'); hold on;
plot(x,vertcat(a18D07.fly.mean)','r.');
plot(x,vertcat(a74C10.fly.mean)','b.');
plot(x,vertcat(a91F02.fly.mean)','m.');

plot(xx,mean(res(1).tc),'k','lineWidth',2);
plot(xx,mean(res(2).tc),'r','lineWidth',2);
plot(xx,mean(res(3).tc),'b','lineWidth',2);
plot(xx,mean(res(4).tc),'m','lineWidth',2);

box off; set(gca,'TickDir','out')
xlabel('wind direction');
ylabel('diff in antennal angles (deg)');
title('Tuning curve fits');

% plot FI with errorbars
subplot(1,2,2);

plot(xd,res(1).FIjkmean,'k','lineWidth',2); hold on;
plot(xd,res(2).FIjkmean,'r','lineWidth',2);
plot(xd,res(3).FIjkmean,'b','lineWidth',2);
plot(xd,res(4).FIjkmean,'m','lineWidth',2);

plot(xd,res(1).FIjkmean+res(1).FIjkSE,'k');
plot(xd,res(1).FIjkmean-res(1).FIjkSE,'k');

plot(xd,res(2).FIjkmean+res(2).FIjkSE,'r');
plot(xd,res(2).FIjkmean-res(2).FIjkSE,'r');

plot(xd,res(3).FIjkmean+res(3).FIjkSE,'b');
plot(xd,res(3).FIjkmean-res(3).FIjkSE,'b');

plot(xd,res(4).FIjkmean+res(4).FIjkSE,'m');
plot(xd,res(4).FIjkmean-res(4).FIjkSE,'m');

set(gca,'Ylim',[0 0.08]);
box off; set(gca,'TickDir','out')
xlabel('wind direction');
ylabel('Fisher information (1/°)^2');
title('Fisher Information');

legend('Canton-S','18D07>Chrimson','74C10>Chrimson','91F02>Chrimson');
