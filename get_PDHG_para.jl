## load our solver parameter ##################################
Nineq = 2*(m+n) #number of poly ineq constraints
Neq = 2*N*(d+1)+(N-1)*n+n
Mep=(d+1)*N*Nineq #number of eq constraints, only from poly ineq
Me=Mep+Neq #number of eq constraints in total

global D=zeros(d+1,d+1) #  求导矩阵，用于右乘
for i in 1:d
    D[i+1,i]=d+1-i
end

############################# 计算不等式约束所需要的L #############################
Lx=zeros(d+1,n*(d+1),2*n) # 这里c是把yflat拉成向量，向量从上到下有d+1段，每段长度为n,不同段对应的L和g是一样的
gx=zeros(d+1,2*n)

Id1=Matrix(1.0I,d+1,d+1)
for i in 1:n
    Lx[:,:,i]=kron((S^(-1))[i,:]',Id1)
    gx[:,i]=-x_min[i,:]
end
for i in 1:n
    Lx[:,:,n+i]=-kron((S^(-1))[i,:]',Id1)
    gx[:,n+i]=x_max[i,:]
end

Lu=zeros(d+1,n*(d+1),2*m)
gu=zeros(d+1,2*m)

Lu[:,d+2:2*(d+1),1]=D
Lu[:,:,1]-=kron((pinv(H)*Gflat)[1,:]',Id1)

Lu[:,3*(d+1)+1:4*(d+1),2]=D
Lu[:,:,2]-=kron((pinv(H)*Gflat)[2,:]',Id1)

Lu[:,d+2:2*(d+1),3]=-D
Lu[:,:,3]+=kron((pinv(H)*Gflat)[1,:]',Id1)

Lu[:,3*(d+1)+1:4*(d+1),4]=-D
Lu[:,:,4]+=kron((pinv(H)*Gflat)[2,:]',Id1)

for i in 1:m
    gu[:,i]=-u_min[i,:]
    gu[:,m+i]=u_max[i,:]
end

Lxcol=zeros(n*(d+1)*N, n*(d+1)*N)
Lucol=zeros(m*(d+1)*N, n*(d+1)*N)
for l in 1:N
    base_ind=(l-1)*n*(d+1)
    for j in 1:n
        Lxcol[base_ind+(j-1)*(d+1)+1:base_ind+j*(d+1),base_ind+1:base_ind+n*(d+1)]=Lx[:,:,j]
    end
end
for l in 1:N
    base_rind=(l-1)*m*(d+1)
    base_cind=(l-1)*n*(d+1)
    for j in 1:m
        Lucol[base_rind+(j-1)*(d+1)+1:base_rind+j*(d+1),base_cind+1:base_cind+n*(d+1)]=Lu[:,:,j]
    end
end 

L=zeros((d+1)*2(m+n)*N, n*(d+1)*N)
g=zeros((d+1)*2(m+n)*N)

for l in 1:N
for j in 1:2(m+n)
for i in 1:d+1
    row=(l-1)*(d+1)*2(m+n)+(j-1)*(d+1)+i
    col=(l-1)*n*(d+1)+1:l*n*(d+1)
    if j<=2n
        L[row,col]=Lx[i,:,j]
        g[row]=gx[i,j]
    else
        L[row,col]=Lu[i,:,j-2n]
        g[row]=gu[i,j-2n]
    end

end
end
end

############################# 等式约束所需要的h,r #############################
h=zeros(Neq,n*(d+1)*N)
r=zeros(Neq)
for l in 1:N # 对应r全为0
    r_seg_ind=(l-1)*2*(d+1)
    c_seg_ind=(l-1)*4*(d+1)
    h[r_seg_ind+1:r_seg_ind+d+1,c_seg_ind+1:c_seg_ind+d+1]=-D
    h[r_seg_ind+1:r_seg_ind+d+1,c_seg_ind+d+2:c_seg_ind+2*(d+1)]=Matrix(1.0I,d+1,d+1)
    h[r_seg_ind+d+2:r_seg_ind+2*(d+1),c_seg_ind+2*(d+1)+1:c_seg_ind+3*(d+1)]=-D
    h[r_seg_ind+d+2:r_seg_ind+2*(d+1),c_seg_ind+3*(d+1)+1:c_seg_ind+4*(d+1)]=Matrix(1.0I,d+1,d+1)
end
############################### 段间连续性要求有 （N-1)*n 个##################################
basis=zeros(d+1)
for i in 1:d+1
    basis[i]=(T/N)^(d+1-i)
end

for l in 1:N-1
    seg_ind=2*N*(d+1)+(l-1)*n
    for j in 1:n
        for i in 1:d+1 
            h[seg_ind+j,(l-1)*n*(d+1)+(j-1)*(d+1)+i]=basis[i]
        end
        h[seg_ind+j,l*n*(d+1)+(j-1)*(d+1)+d+1]=-1
    end
end
############################ 初值约束有 n个 #################################################
for j in 1:n
    seg_ind=2*N*(d+1)+(N-1)*n
    h[seg_ind+j,1:n*(d+1)]=Lx[end,:,j]
    r[seg_ind+j]=x0[j]
end

############  calc 目标函数 P ################################################
Qx=zeros(d+1,d+1)

for ii in d:-1:0
    for jj in d:-1:0
        Qx[d+1-ii,d+1-jj]=(1)^(ii+jj+1)/(ii+jj+1)
    end
end
# obj_dim=m+n # all entries has REF
# I_objdn=Matrix(1.0I,obj_dim*(d+1)*N,obj_dim*(d+1)*N)
I_objn=[Matrix(1.0I,n*N,n*N) zeros(n*N,m*N);zeros(m*N,n*N) Matrix(0.1I,m*N,m*N)]
P=kron(I_objn,Qx)

P2=[Lxcol;Lucol]'*P*[Lxcol;Lucol] # 二次项系数的变换
P2=(P2+P2')/2 # 重新对称化

## 使用固定reference
xref1=zeros(d+1,N)
xref2=zeros(d+1,N)
xref3=zeros(d+1,N)
xref4=zeros(d+1,N)
for j in 1:N
    xref1[end,j]=19.9
    xref2[end,j]=19.9
    xref3[end,j]=2.4
    xref4[end,j]=2.4
end
xref=zeros(d+1,n,N)
xref[:,1,:]=xref1
xref[:,2,:]=xref2
xref[:,3,:]=xref3
xref[:,4,:]=xref4
xref_col=zeros(n*(d+1)*N)
for l in 1:N
    xref_col[(l-1)*n*(d+1)+1:l*n*(d+1)]=(xref[:,:,l])[:]
    #注意这里把三维向量抻长的过程，要按行展开，即先xref[1,:,1]第一行,再xref[2,:,1]
end
q=-([xref_col; zeros(m*(d+1)*N)]'*P*[Lxcol;Lucol])'

################ calculate F,G ##################################
function calc_FG(seg_T,d)
    d2=Int((d-1)/2+1)
    Fconst = zeros(d2,d2,d)

    for ix in CartesianIndices(Fconst)
        i, j, k = ix.I
        if i + j == k + 1
            Fconst[ix] = 1.
        end
    end
    Fout=zeros(d2,d2,d+1)
    Gout=zeros(d2,d2,d+1)
    for i in 1:d+1
        if i==1
            Fout[:,:,i]=Fconst[:,:,i]
            Gout[:,:,i]=-Fconst[:,:,i]
        elseif i==d+1
            Fout[:,:,i]=0*Fconst[:,:,d]
            Gout[:,:,i]=seg_T.*Fconst[:,:,d]
        else
            Fout[:,:,i]=Fconst[:,:,i]
            Gout[:,:,i]=-Fconst[:,:,i]+seg_T.*Fconst[:,:,i-1]
        end

    end

    return Fout,Gout
end

F,G=calc_FG(T/N,d)
# A=[F[:,:,1][:]' G[:,:,1][:]';
# F[:,:,2][:]' G[:,:,2][:]';
# F[:,:,3][:]' G[:,:,3][:]';
# F[:,:,4][:]' G[:,:,4][:]';
# F[:,:,5][:]' G[:,:,5][:]';
# F[:,:,6][:]' G[:,:,6][:]']

left=[2*P2 zeros(n*(d+1)*N, (d+1)*2(m+n)*N) L' h';
zeros( (d+1)*2(m+n)*N, n*(d+1)*N) d_step_size*I -I zeros((d+1)*2(m+n)*N, Neq);
L -I zeros(size(L,1),size(L,1)) zeros(size(L,1), Neq);
h zeros(Neq, (d+1)*2(m+n)*N) zeros(Neq, size(L,1)) zeros(Neq, Neq)]

global inv_left=inv(left)

global FF=inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N,size(q,1)+1:size(q,1)+2(m+n)*(d+1)*N]
global μμ=inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N,1:size(q,1)]*(-2*q)+inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N, size(q,1)+2(m+n)*(d+1)*N+1:end]*[-g; r]

