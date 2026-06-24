function y = cxk(rr,sint)
bandera = 800;
n = length(rr);
y = zeros(bandera+1,1);

for k=0:bandera
    inn = rr(1:n-k);
    inn1 = sint(k+1:n);
    y(k+1) = (sum(inn.*inn1))/(n-k);
end

end