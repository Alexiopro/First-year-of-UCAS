function  p =  Parzen_Gauss(x,h,N)
%  ��˹����Parzen ��  ͳ������parzen���ڵĹ��Ƹ���
f = x(1:50);
f=sort(f);
b=0;
h1 = h;
for i=1:50
    for j=1:N
    b= b+ exp(((x(j)-f(i))/h1).^2/(-2))/sqrt(2*pi)/h1;
    end
    p(i) =  b/N;
    b=0;
end
