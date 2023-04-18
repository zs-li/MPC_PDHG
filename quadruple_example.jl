using LinearAlgebra, NumericalIntegration
using DynamicPolynomials, Plots


include("./get_system_matrix.jl")

Ac,Bc,Gflat,S,H=get_system_matrix()

global d = 3
n = 4 # dim x
m = 2 # dim u
Nineq = (m+n)*2
global d2=Int((d-1)/2+1)

global T = 20
global N = 20


x_min = zeros(n, d+1)
x_max = zeros(n, d+1)
x_max[:,end] .= 20

u_min = [zeros(1,d) 0; zeros(1,d) 0] 
u_max = [zeros(1,d) 8; zeros(1,d) 8]

x0=[10; 19; 19; 1]


xrefs=zeros(d+1,n,N)
for l in 1:N
    xrefs[end,1,l]=19.9
    xrefs[end,2,l]=19.9
    xrefs[end,3,l]=2.4
    xrefs[end,4,l]=2.4
end


p_step_size=0.2
d_step_size=0.4


include("./get_PDHG_para.jl")

include("./PDHG_GPU_solver.jl")



function apply_poly_u(ut_coef_in,xk,apply_T)
    Δt = 0.0001
    Ad=exp(Ac*Δt)
    apply_steps=Int(apply_T/Δt)
    x_test=zeros(n,apply_steps+1)
    x_test[:,1]=xk
    for k in 1:apply_steps
        basis=zeros(d+1)
        for i in 1:d+1
            basis[d+2-i]=(k*Δt)^(i)/(i)-((k-1)*Δt)^(i)/(i)
        end
        delta_Bu=Bc*ut_coef_in*basis
        x_test[:,k+1]=Ad*x_test[:,k]+delta_Bu
    end
    return x_test[:,end]
end


function shift_warm_start(Xf_in,Xg_in,u_in)
    Xf_in=Array{Float32,4}(reshape(Xf_in, d2, d2, 2(m+n), N))
    Xg_in=Array{Float32,4}(reshape(Xg_in, d2, d2, 2(m+n), N))
    u_in=Array{Float32,3}(reshape(u_in, d+1, 2(m+n), N))
    for l in 2:N
        Xf_in[:,:,:,l]=Xf_in[:,:,:,l-1]
        Xg_in[:,:,:,l]=Xg_in[:,:,:,l-1]
        u_in[:,:,l]=u_in[:,:,l-1]
    end
    Xf_in[:,:,:,1]=rand(d2, d2, 2(m+n)).-0.5
    Xg_in[:,:,:,1]=rand(d2, d2, 2(m+n)).-0.5
    u_in[:,:,1]=rand(d+1, 2(m+n)).-0.5
    Xf_in=Array{Float32,3}(reshape(Xf_in, d2, d2, 2(m+n)*N))
    Xg_in=Array{Float32,3}(reshape(Xg_in, d2, d2, 2(m+n)*N))
    u_in=Array{Float32,2}(reshape(u_in, d+1, 2(m+n)*N))
    return Xf_in,Xg_in,u_in
end


global k=1


max_k=101
x=zeros(n,max_k)
ut_coef=zeros(m,d+1,N,max_k)
xt_coef=zeros(n,d+1,N,max_k)


x[:,1]=x0   
apply_T=1

F,G=calc_FG(T/N,d)

while true
    global k, last_Xf, last_Xg, last_dual, inv_left, FF, μμ

    @show k
   

    if k==1
        xt_coef[:,:,:,k],ut_coef[:,:,:,k],last_Xf,last_Xg,last_dual=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ, F,G, 1000)

    else
        for j in 1:n
            seg_ind=2*N*(d+1)+(N-1)*n
            r[seg_ind+j]=x[j,k]
        end
        μμ=inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N,1:size(q,1)]*(-2*q)+inv_left[n*(d+1)*N+1: n*(d+1)*N+(d+1)*2(m+n)*N, size(q,1)+2(m+n)*(d+1)*N+1:end]*[-g; r]

        last_Xf, last_Xg, last_dual =shift_warm_start(last_Xf, last_Xg, last_dual)

        xt_coef[:,:,:,k],ut_coef[:,:,:,k],last_Xf,last_Xg,last_dual=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ,F,G, 500,true,last_Xf,last_Xg,last_dual)
    end


    x[:,k+1]=apply_poly_u(ut_coef[:,:,1,k],x[:,k],apply_T)
    k+=1
    if k>=max_k
        break
    end
end



plot(x',title="MPC states N=$N, T=$T, d=$d",label="",xlabel="time / s", ylabel="Liquid level height/cm")
savefig("x.png")


# visualize polynomials
@polyvar t
n_points=101 # how many points in a segment
seg_T=Int(T/N)
parallel_u=zeros(m,n_points*N,max_k)
parallel_x=zeros(n,n_points*N,max_k)
for k in 1:max_k
    for i in 1:m # inputs
    for j = 1:N # record how many segments
        pu=dot(monomials([t], 0:d),(ut_coef[i,:,j,k]))
        x_list=0:1/(n_points-1):1
        for kk in 1:n_points
            parallel_u[i,(j-1)*n_points+kk,k]=pu(t=>x_list[kk])
        end
    end
    end
    for i in 1:n # states
    for j = 1:N # record how many segments
        px=dot(monomials([t], 0:d),(xt_coef[i,:,j,k]))
        x_list=0:1/(n_points-1):1
        for kk in 1:n_points
            parallel_x[i,(j-1)*n_points+kk,k]=px(t=>x_list[kk])
        end
    end
    end
end
## plot inputs
plot()
for k in 1:max_k
    for i in 1:1 # plot how many segments
        time_axis=(k-1)+(i-1)*1:1/(n_points-1):(k-1)+i*1
        for j = 1:m
            plot!(time_axis,parallel_u[j,(i-1)*n_points+1:i*n_points,k],label="")
        end
    end
end
plot!()
savefig("u_poly.png")
## plot states
plot()
for k in 1:max_k
    for i in 1:1 # plot how many segments
        time_axis=(k-1)+(i-1)*1:1/(n_points-1):(k-1)+i*1
        for j = 1:n
            plot!(time_axis,parallel_x[j,(i-1)*n_points+1:i*n_points,k],label="")
        end
    end
end
plot!()
savefig("x_poly.png")
# plot!(xlims=[45,55],ylims=[19.9995,20.0005])
