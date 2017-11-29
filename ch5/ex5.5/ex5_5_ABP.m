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
bias_h = rand(h); %���ز���ֵ 4*1
bias_o = rand(o); %�������ֵ 1X1
eta = 1; %ѧϰ����
it = 2000; %����5000��
for t = 1:it
  J=0;
  for i=1:m
    for j=1:h
      ca = 0;
      for k=1:n
        ca = ca + v(k,j)*X(i,k);
      end
      a2(i,j) = 1/(1+exp(-ca + bias_h(j))); 
    end
    
    for j=1:o
      cb = 0;
      for k=1:h
        cb = cb+ w(k,j)*a2(i,k);
      end
      a3(i,j) = 1/(1+exp(-cb + bias_o(j)));
    end
  %{
    z1 = X;  %���������ֵ 17*2
    z2 = z1*v;   %���ز�����  17*4
    a2 = sigmoid(z2-bias_h'); %���ز���� 17*4
    z3 = a2*w;   %��������� 17*1
    a3 = sigmoid(z3-bias_o);  %���17*3
    %}
  end
    delta1 = zeros(n,h);
    delta2 = zeros(h,o);
    delta_bias_h = zeros(h);
    delta_bias_o = zeros(o);
    for i =1:m
      for j=1:o
        J =J + (a3(i,j)-y(i,j))^2/2; %��������
      end
      for j=1:o
        gj(j) = -a3(i,j)*(1-a3(i,j))*(a3(i,j)-y(i));  %17*1
      end
      
      for j=1:h
        temp = 0;
        for k = 1:o
          temp = temp+w(j,k)*gj(k);
        end

        eh(j) = temp*a2(i,j)*(1-a2(i,j));  %4*1
      end
      
      %�������ز���ֵ����ֵ������㵽���ز�Ȩֵ����ֵ
      for j = 1:h
        delta_bias_h(j) = delta_bias_h(j) + (-1)*eh(j);
        for k=1:n
          delta1(k,j) =delta1(k,j)+ eh(j)*X(i,k);
        end
      end
      
      %���������ֵ����ֵ�����ز㵽�����Ȩֵ����ֵ
      for j=1:o
        delta_bias_o(j) = delta_bias_o(j) + (-1)*gj(j);
        for k = 1:h
          delta2(k,j)=delta2(k,j)+gj(j)*a2(i,k);
        end
      end
          
   end
        w = w+eta*delta2;
      v = v+eta*delta1; 
      bias_h = bias_h + eta*delta_bias_h;
      bias_o = bias_o + eta*delta_bias_o;

  %J(t) = sum(delta3)  %�����ۻ���� 
end
%plot(J);
  %if(abs(abs(old_J-J) <= 0.00001)) break; end
  %old_J =J;
%end
%plot(J);
%y_t = predict(X,v,w,bias_h,bias_o);