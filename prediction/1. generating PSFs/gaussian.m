function [Gaussian] = gaussian(a,r,sigma)
Gaussian = a * exp(-(r^2/(2*sigma^2)));
end