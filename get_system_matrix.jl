function get_system_matrix()
A1 = 28
A3 = 28
A2 = 32
A4 = 32

k1_negative = 3.33
k2_negative = 3.35
k1_positive = 3.14
k2_positive = 3.29

g1_negative = 0.7
g2_negative = 0.6
g1_positive = 0.43
g2_positive = 0.34

T1_negative = 62
T2_negative = 90
T3_negative = 23
T4_negative = 30
T1_positive = 63
T2_positive = 91
T3_positive = 39
T4_positive = 56


Ac = [
-1 / T1_negative 0 A3 / (A1 * T3_negative) 0;
0 -1 / T2_negative 0 A4 / (A2 * T4_negative);
0 0 -1 / T3_negative 0;
0 0 0 -1 / T4_negative]

Bc = [
g1_negative * k1_negative / A1 0;
0 g2_negative * k2_negative / A2;
0 (1 - g2_negative) * k2_negative / A3;
(1 - g1_negative) * k1_negative/A4 0]


P = [Bc[:, 1] Ac*Bc[:, 1] Bc[:, 2] Ac*Bc[:, 2]]
S = P^(-1)
S = [S[2,:]' ; S[2,:]'*Ac ; S[4,:]' ; S[4,:]'*Ac]


G = S * Ac * (S)^(-1)
H = S * Bc
for i in eachindex(G)
    if abs(G[i])<10^(-10)
        G[i]=0
    end
end
for i in eachindex(H)
    if abs(H[i])<10^(-10)
        H[i]=0
    end
end

return Ac,Bc,G,S,H
end