kernel_blur = zeros(39,39,181,'single');
for i=1:39
    for j=1:39
        center = 20;
        rx = i-center;
        ry = j-center;
        r = sqrt(rx^2 + ry^2);
        count=1;
        for r_a=1:1:3
            for h=1:1:5
                for b=0:1
                   for sigma=1:1:3
                      for a=0:1
                        Lorentzian = lorentz(r, h, b);
                        Airy = airy_1(r, r_a);
                        Gaussian = gaussian(a, r,sigma);
                        total = Lorentzian + Airy + Gaussian;
                        disp(count);
                        kernel_blur(i,j,count) = total;
                        count=count+1;
                      end
                   end
                end
            end
        end
    end
end

PSF = zeros(39,39,'single');
PSF(20,20) = 1;
kernel_blur(:,:,181) = PSF;

for k=1:180
   normal = kernel_blur(:,:,k);
   kernel_blur(:,:,k) = normal / sum(sum(normal));
end

save('kernel_blur.mat', 'kernel_blur');
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    