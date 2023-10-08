import numpy as np
cimport numpy as np

def rebuild(const double[:,:] eri, int n_orb):
    '''
    n_orb = 2*n_orb is the number of spin orbitals
    eri, (ab|ij), shape=(N,N), N=n_orb*(n_orb+1)//2
    h2e, <ab||ij> = <ab|ij> - <ab|ji> = (ai|bj) - (aj|bi), shape=(n_orb,n_orb,n_orb,n_orb)
    <ab|ij> = <a_ b|i_ j> = <a b_|i j_> = <a_ b_|i_ j_>
    otherwise, <ab|ij>=0 because of spin
    '''
    h2e = np.zeros((n_orb*2,)*4, dtype=np.float64)
    cdef double[:,:,:,:] h2e_view = h2e

    cdef int a,b,i,j
    cdef int a0,b0,i0,j0

    # i<n_orb; b<n_orb; i<n_orb; j<n_orb
    for a in range(n_orb):
        for b in range(n_orb):
            for i in range(n_orb):
                for j in range(n_orb):
                    # <ab|ij> = (ai|bj)
                    linx = [a,i]
                    rinx = [b,j]
                    linx.sort()
                    rinx.sort()
                    a0, i0 = linx
                    b0, j0 = rinx
                    h2e_view[a,b,i,j] = eri[i0*(i0+1)//2+a0,j0*(j0+1)//2+b0]

    # i<n_orb; b>n_orb; i<n_orb; j>n_orb
    for a in range(n_orb):
        for b in range(n_orb,n_orb*2):
            for i in range(n_orb):
                for j in range(n_orb,n_orb*2):
                    # <a b_|i j_> = (a i|b_ j_)
                    linx = [a,i]
                    rinx = [b,j]
                    linx.sort()
                    rinx.sort()
                    a0, i0 = linx
                    b0, j0 = rinx
                    h2e_view[a,b,i,j] = eri[i0*(i0+1)//2+a0,j0*(j0+1)//2+b0]

    # i>n_orb; b<n_orb; i>n_orb; j<n_orb
    for a in range(n_orb,n_orb*2):
        for b in range(n_orb):
            for i in range(n_orb,n_orb*2):
                for j in range(n_orb):
                    # <ab|ij> = (ai|bj)
                    linx = [a,i]
                    rinx = [b,j]
                    linx.sort()
                    rinx.sort()
                    a0, i0 = linx
                    b0, j0 = rinx
                    h2e_view[a,b,i,j] = eri[i0*(i0+1)//2+a0,j0*(j0+1)//2+b0]

    # i>n_orb; b>n_orb; i>n_orb; j>n_orb
    for a in range(n_orb,n_orb*2):
        for b in range(n_orb,n_orb*2):
            for i in range(n_orb,n_orb*2):
                for j in range(n_orb,n_orb*2):
                    # <ab|ij> = (ai|bj)
                    linx = [a,i]
                    rinx = [b,j]
                    linx.sort()
                    rinx.sort()
                    a0, i0 = linx
                    b0, j0 = rinx
                    h2e_view[a,b,i,j] = eri[i0*(i0+1)//2+a0,j0*(j0+1)//2+b0]

    return h2e

def bisearch(np.ndarray[np.int8_t, ndim=2] states):
    cdef int left, index, right
    left = 0
    right = states.shape[0] - 1
    index = (left + right) // 2
    while left != right:
        if np.all(states[index]==states[index+1]):
            right = index
        else:
            left = index + 1
        index = (left + right) // 2
    return index + 1

def unique_combine(np.ndarray[np.int8_t, ndim=3] states):
    cdef int i, index
    unique_states = [states[i,:bisearch(states[i]),:] for i in range(states.shape[0])]
    return np.vstack(unique_states)