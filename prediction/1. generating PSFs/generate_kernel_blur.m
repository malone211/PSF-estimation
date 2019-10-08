for img_count=1:10000
    kernel_blur = zeros(39,39,181,'single');
    for kernel_blur_count=1:180
        r_a = rand(1)*2+1;
        h = rand(1)*4+1;
        sigma = rand(1)*2+1;
        a = rand(1);
        b = rand(1);
        for row=1:39
            for column=1:39
                centre = 20;
                rx = row-centre;
                ry = column-centre; 
                r = sqrt(rx^2+ry^2);
                Airy = airy_1(r,r_a);
                Lorentz =  lorentz(r,h,b);
                Gaussian = gaussian(a,r,sigma);
                total = Airy + Lorentz + Gaussian;
                kernel_blur(row, column, kernel_blur_count)=total;
            end
        end
        %kernel_blur mean 
        normal = kernel_blur(:,:,kernel_blur_count);
        kernel_blur(:,:,kernel_blur_count) = normal / sum(sum(normal));
    end
    %add PSF
    PSF = zeros(39,39,'single');
    PSF(20,20) = 1;
    kernel_blur(:,:,181) = PSF;
    file_name = sprintf('%d.mat',img_count);
    save(file_name, 'kernel_blur');
end



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    