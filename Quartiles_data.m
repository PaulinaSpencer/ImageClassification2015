function [Q1, Q2, Q3] = Quartiles_data(SCORES) 
%compute the quartiles of the current score data set.  This will allow us
%to use the quartiles (Q3) to sort data into binary classes of good vs bad

%make sure this function is in your current working folder, along with
%BINARY_Reddit_Runscript.m

X = SCORES ; 

xs = sort(X) ; 
medx = median(xs) ; 

Q1 = median(xs(find(xs<median(xs)))) ;
Q2 = median(xs) ; 
Q3 = median(xs(find(xs>median(xs)))); 

IQR = Q3 - Q1 ; 

%Q1 outliers
Q1_O = find(xs<Q1-1.5*IQR) ; 
Q1_O = xs(Q1_O) ;
%Q3 outliers
Q3_O = find(xs> Q3 + 1.5*IQR)  ;
Q3_O = xs(Q3_O) ;


end
