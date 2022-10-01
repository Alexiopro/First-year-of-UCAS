function p=Parzen(x,h,N)
% ·½´°  parzen ´°   
f = x(1:50);
f=sort(f);
b=0;
for i=1:50
    for j=1:N
        if abs((x(j)-f(i))/h) <= 1/2
            q=1;
        else
            q=0;
        end
        b= q+ b;
    end
    a(i)=b;
    b=0;
end
for i=1:50
    p(i) = 1/(N*h)*a(i);
end
end