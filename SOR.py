
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('load_ext', 'cython')


# In[33]:


get_ipython().run_cell_magic('cython', '-a', 'import cython\nimport numpy as np\n\nA_py = [[102.5, 0., 2.5, 0., 0.],\n    [3.5, 104.88, 1.05, 0., 0.33],\n    [0., 0., 100., 0., 0.], \n    [0., 1.3, 0.0, 101.3, 0.0], \n    [0.73, 0.0, 0., 1.5, 102.23]]\nA_np = np.asarray(A_py)\n\nb_np = np.asarray([6., 7., 8., 9., 1.])\n\ninitial_guess = np.asarray([0., 0., 0., 0., 0.])\n\ncdef double w = 1.\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef solve_sor(double[:, :] a, double[:] b, double[:] sol, double w):\n    cdef int m, n, p, i # a are m linii si n coloane\n    cdef double sum_1, sum_2\n    \n    m = a.shape[0]\n    n = a.shape[1]\n    p = b.shape[0]\n    \n    assert (n == p)\n    \n    with cython.nogil:\n        for i in range(0, m):\n            sum_1 = 0\n            sum_2 = 0\n            for j in range(0, i):\n                sum_1 += a[i, j] * sol[j] # folososesc solutii updatate\n            for j in range(i + 1, m):\n                sum_2 += a[i, j] * sol[j]\n            sol[i] += (w / a[i, i]) * (b[i] - sum_1 - sum_2)\n    \n    print("sol =", sol)\n    return sol\nx = solve_sor(A_np, b_np, initial_guess, w)')

