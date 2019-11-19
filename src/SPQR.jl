module SPQR

export spqr

using LinearAlgebra
using Krylov
using SparseArrays

function spqr(R ::SparseMatrixCSC)
    m = R.m
    n = R.n
    Pr = AbstractArray{Int}(1:m)
    Pc = AbstractArray{Int}(1:n)
    rank = min(m,n)
    Q = SparseMatrixCSC{Float64}(I,m,m)

   for i = 1:rank
        Pc0 = AbstractArray{Int}(1:n)
        Pr0 = AbstractArray{Int}(1:m)

        while i <= rank && R[i, i:n].nzind == []
            aux = Pr[rank]
            Pr[rank] = Pr[i]
            Pr[i] = aux
            aux = Pr0[rank]
            Pr0[rank] = Pr0[i]
            Pr0[i] = aux
            R = permute(R, Pr0, AbstractArray{Int}(1:n))
            rank -= 1
        end

        mx = R[i,i]
        j = i
        for l in R[i,1:n].nzind
            if R[i,l] > mx && l > i
                j = l
                mx = R[i,l]
            end
        end
        aux = Pc[i]
        Pc[i] = Pc[j]
        Pc[j] = aux
        aux = Pc0[i]
        Pc0[i] = Pc0[j]
        Pc0[j] = aux
        R = permute(R, AbstractArray{Int}(1:m), Pc0)

        for l in R[:,i].nzind
            if l > i
                G = SparseMatrixCSC{Float64}(I,m,m)
                Gt = SparseMatrixCSC{Float64}(I,m,m)
                c = R[i,i]
                s = R[l,i]
                r = sqrt(c^2 + s^2)
                G[i,i] = c/r
                G[l,i] = -s/r
                G[i,l] = -G[l,i]
                G[l,l] = G[i,i]
                Gt[i,i] = G[i,i]
                Gt[l,i] = G[i,l]
                Gt[i,l] = G[l,i]
                Gt[l,l] = G[l,l]
                R = G*R
                R[l,i] = 0.0
                R = dropzeros(R)
                Gt = permute(Gt, Pr0, AbstractArray{Int}(1:m))
                Q = Q*Gt
                Q = dropzeros(Q)
            end
        end
    end

    return rank, Q, R, Pr, Pc
end

end # module
