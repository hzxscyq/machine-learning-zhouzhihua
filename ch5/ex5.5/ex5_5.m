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

[m n] = size(X);
h = 4;    %4�����ؽڵ�
o = 1;    %1������ڵ�
%��������ֵ��4�����ؽڵ㣬һ������ڵ�
v = rand(n,h);  %����㵽���ؽڵ��Ȩֵ���� 2X4
w = rand(h,o);  %���ز㵽������Ȩֵ���� 4X1
bias_h = rand(h,1); %���ز���ֵ 4X1
bias_o = rand(o,1); %�������ֵ 1X1
eta = 0.9; %ѧϰ����
delta3 = zeros(m,1);
it = 20000; %����5000��
J=0;old_J=0;
for t = 1:it
  for i=1:m
    z1 = X(i,:);  %���������ֵ 1X2
    z2 = z1*v;   %���ز�����  1X4
    a2 = sigmoid(z2-bias_h'); %���ز���� 1X4
    %a2 = (bias_o z2);  %������ֵ 1X5
    z3 = a2*w;   %��������� 1X1
    a3 = sigmoid(z3-bias_o);  %���
    delta3(i) = (a3-y(i,:)).^2/2; %��������
    gj = -a3.*(1-a3).*(a3-y(i,:));  %1X1
    eh = (gj*w)'.*a2.*(1-a2);  %1x4
    delta_bias_o = -eta*gj;  %����ڵ���ֵ�޸�ֵ
    delta_bias_h = -eta*eh;  %���ؽڵ���ֵ�޸�ֵ
    delta2 = gj*a2*eta; %w���޸�ֵ 1X4
    delta1 = eta*eh'*z1; %v���޸�ֵ 4X2
    w = w + delta2';
    v = v + delta1';
    bias_h = bias_h + delta_bias_h';
    bias_o = bias_o + delta_bias_o;
  end
  J(t) = sum(delta3)  %�����ۻ���� 
  %if(abs(abs(old_J-J) <= 0.00001)) break; end
  old_J =J;
end
plot(J);
y_t = predict(X,v,w,bias_h,bias_o);