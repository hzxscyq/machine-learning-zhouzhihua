clear;
X=[0.679 0.460;
   0.774 0.376;
   0.634 0.264;
   0.608 0.318;
   0.556 0.215;
   0.403 0.237;
   0.481 0.149;
   0.437 0.211;
   0.666 0.091;
   0.243 0.267;
   0.245 0.057;
   0.343 0.099;
   0.639 0.161;
   0.657 0.198;
   0.360 0.370;
   0.593 0.042;
   0.719 0.103];
y= [1;1;1;1;1;1;1;1;0;0;0;0;0;0;0;0;0];
figure(1);hold on;
for i = 1:17
  if(y(i) == 1)
    plot(X(i,1),X(i,2),'k+','LineWidth',2,'MarkerSize',10);
  else
    plot(X(i,1),X(i,2),'ko','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','y');
  end
end
[m,n]=size(X);
u1 = zeros(1,n);u0 = zeros(1,n);
u1 = [mean(X(1:8,1)),mean(X(1:8,2))];
u0 = [mean(X(9:17,1)),mean(X(9:17,2))];

Sw = X;
temp1 = X(1:8,:);
temp1(1:8,1) = temp1(1:8,1)-ones(8,1)*u1(1);
temp1(1:8,2) = temp1(1:8,2)-ones(8,1)*u1(2);
temp0 = X(9:17,:);
temp0(1:9,1) =  temp0(1:9,1)-ones(9,1)*u0(1);
temp0(1:9,2) =  temp0(1:9,2)-ones(9,1)*u0(2);
Sw = temp1'*temp1 + temp0'*temp0;
w= inv(Sw)*(u0-u1)';

y1 = -w(1)*0.1/w(2);
y2 = -w(1)*0.9/w(2);
plot([0.1 0.9],[y1 y2]);