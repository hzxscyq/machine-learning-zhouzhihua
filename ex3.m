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

m = size(X,1);
X = [ones(m,1) X];
theta = [0;0;1];
J=0;
alpha = 0.01;
for i = 1:2000
  J = -(log(y'*sigmoid(X*theta)+(1-y)'*(1-sigmoid(X*theta))));
  grad = X'*(sigmoid(X*theta)-y);
  theta = theta - alpha*grad;
end
figure(1);hold on;
for i = 1:17
  if(y(i) == 1)
    plot(X(i,2),X(i,3),'k+','LineWidth',2,'MarkerSize',10);
  else
    plot(X(i,2),X(i,3),'ko','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','y');
  end
end
  test_y_1 = -(theta(1)+theta(2)*0.1)/theta(3);
  test_y_2 = -(theta(1)+theta(2)*0.9)/theta(3);
  line([0.1 0.9],[test_y_1 test_y_2]);
 x1=[1 0.650 0.450]; predict(theta,x1);
 x2=[1 0.240 0.220]; predict(theta,x2);
 
 