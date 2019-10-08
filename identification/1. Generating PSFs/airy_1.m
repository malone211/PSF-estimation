function [Airy] = airy_1(r, r_a)
beta = 10^(-16);
Z = (pi * r + beta)/(2 * r_a);
Airy = (2 *  besselj(1,Z) / Z)^2;
end