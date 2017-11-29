## Copyright (C) 2017 63254
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} predict (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 63254 <63254@DESKTOP-N7OQKM7>
## Created: 2017-11-27

function [y] = predict (X,v,w,bias_h,bias_o)
  [m,n] = size(X);
  for i=1:m
   z1 = X(i,:); % ‰»Î 1X2
   z2 = z1*v; %“˛≤ÿ≤„ ‰»Î1X4
   a2 = sigmoid(z2' - bias_h); %“˛≤ÿ≤„ ‰≥ˆ 4X1
   z3 = a2'*w;   % ‰»Î≤„ ‰»Î 1X1
   a3 = sigmoid(z3' - bias_o); % ‰≥ˆ≤„ ‰≥ˆ
   y(i) = a3;   
  end
endfunction
