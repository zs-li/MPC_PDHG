using TensorOperations
using CUDA


function eigenval_recover_batched!(A, D, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(D)
        if D[i] < 0.
	        @inbounds D[i] = 0.
		end
        for j = 1:m
            A[(i-1) * m + j] *= D[i]
        end
    end
    return
end

function projection_batched(X_d, m::Int64)
    D, A = CUSOLVER.syevjBatched!('V', 'U', X_d)
    A_c = deepcopy(A)
    @cuda threads=THREADS_PER_BLOCK blocks=BLOCK eigenval_recover_batched!(A, D, m)
    X_d = CUBLAS.gemm_strided_batched('N', 'T', 1., A, A_c)
    return X_d
end


function interpret_solution(Xf,Xg)
    our_y=zeros(2(m+n)*(d+1)*N)
    xt=zeros(n,d+1,N)
    ut=zeros(m,d+1,N)
    # xt_col=zeros(n*(d+1)*N)
    # ut_col=zeros(m*(d+1)*N)

    # Px=kron(Matrix(1.0I,n*N,n*N),Qx)   
    # Pu=kron(Matrix(0.1I,m*N,m*N),Qx)  

    Xf = Array{Float32,4}(reshape(Xf, d2, d2, 2(m+n), N))
    Xg = Array{Float32,4}(reshape(Xg, d2, d2, 2(m+n), N))

    for l in 1:N
        for i in 1:d+1 
            seg=(l-1)*2(m+n)*(d+1)
            for j in 1:2(m+n)
                our_y[seg+(j-1)*(d+1)+i]=tr(F[:,:,i]*Xf[:,:,j,l])+tr(G[:,:,i]*Xg[:,:,j,l])
            end
            for j in 1:n
                xt[j,i,l]=((our_y-g)[seg+(j-1)*(d+1)+i]-(our_y-g)[seg+(j+n-1)*(d+1)+i])/2 #/2 to get average
                # xt_col[(l-1)*n*(d+1)+(j-1)*(d+1)+i]=xt[j,i,l]
            end
            for j in 1:m
                ut[j,i,l]=((our_y-g)[seg+(j+2n-1)*(d+1)+i]-(our_y-g)[seg+(j+2n+m-1)*(d+1)+i])/2 #/2 to get average
                # ut_col[(l-1)*m*(d+1)+(j-1)*(d+1)+i]=ut[j,i,l]
            end        
        end
    end
    # obj=(xt_col-xref_col)'*Px*(xt_col-xref_col)+ut_col'*Pu*ut_col
    # innerp=u'*(our_y)

    return xt,ut#,obj,innerp

end


const BLOCK = 128
const THREADS_PER_BLOCK = Int(ceil(N * 2(m+n) / BLOCK))
I_FF = cu(I(2(m+n)*(d+1)*N))

function PDHG_solver(N,d,m,n,α,β,FF,μμ,F,G, max_iter=20000,if_warm_start=false,last_Xf=nothing,last_Xg=nothing,last_u=nothing)

d2=Int((d-1)/2+1)


if if_warm_start 
    # primal variables, 
    Xf=last_Xf
    Xg=last_Xg
    # # dual variables 
    u=last_u
else
    # primal variables, 
    Xf=rand(d2,d2,2(m+n)*N).-0.5
    Xg=rand(d2,d2,2(m+n)*N).-0.5
    # # dual variables 
    u=zeros(d+1,2(n+m)*N).-0.5
end


FF = cu(FF)
μμ = cu(μμ)
β = cu(β)
α = cu(α)
IminusβFF = CuArray{Float32}(I_FF-FF*β)
FG = cu(cat(F, G, dims=3))

X = cu(cat(Xf, Xg, dims=3))
dX = cu(zeros(d2,d2,2Nineq*N))

u_large = zeros(2(d+1), 4(m+n) * N)
u_large[1:d+1, 1:2(m+n) * N] = u
u_large[d+2:end, 2(m+n) * N + 1:end] = u
u = cu(u)
u_large = cu(u_large)
u_large_tmp = cu(deepcopy(u_large))



@time for k in 1:max_iter
    oldX = deepcopy(X)
    u_large[1:d+1, 1:2(m+n) * N] = u
    u_large[d+2:end, 2(m+n) * N + 1:end] = u
    @tensor begin
        X[a, b, j] -= α * u_large[i, j] * FG[a, b, i]
    end
    X = projection_batched(X, d2)
    @tensor begin
        dX = 2 * X - oldX
        u_large_tmp[i, j] = β * FG[a, b, i] * dX[b, a, j]
    end
    u += u_large_tmp[1:d+1, 1:2(m+n) * N] + u_large_tmp[d+2:end, 2(m+n) * N + 1:end]
    tmp = deepcopy(μμ)
    u = reshape(u, 2(m+n) * (d+1) * N)
    CUBLAS.symv!('U', 1., IminusβFF, u, -β, tmp)
    u = reshape(tmp, d+1, 2(m+n)*N)
end

xt,ut=interpret_solution(X[:,:,1:Nineq*N],X[:,:,Nineq*N+1:end])

return xt,ut,Xf,Xg,u
end




