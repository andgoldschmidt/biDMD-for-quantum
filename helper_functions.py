import numpy as np
from itertools import combinations


def u_idft(t, args):
    '''
        Analytic expression for the inverse discrete fourier transform.
        
        Args:
            t (int): The time at which to evaluate the function.
            
            args (dict): A dictionary of function arguments requiring:
                'u_hat': A list of frequency coefficients--the DFT of a signal.
                'dt': The time step between the discrete spatial measurements of the DFT.
                 
        Returns:
            (`ndarray` of float): The evaluation of the IDFT at time t.

        Raises:
            ValueError: If t is a time series, make sure the timesteps can be divided by
                args['dt'] without a remainder. This is to catch timesteps that have a 
                persistent error.
    ''' 
    if not np.isscalar(t):
        test = round((t[1]-t[0]) / args['dt'], 4) % 1
        if not test == 0:
            raise ValueError("args['dt']={:.5} is not an integer divisor ".format(args['dt']) + \
                             "of the timesteps in t={:.5}.".format(t[1]-t[0]) + \
                             "\n Result: {}".format(test))

    n = len(args['u_hat'])
    # compute the dft manually
    res = np.zeros_like(t)
    for k in np.nonzero(args['u_hat'])[0]:
        res = res + args['u_hat'][k] * np.exp(2j * np.pi / n * k * (t / args['dt']))
    return res.real/n


def resample_u_dft(args, resample):
    '''
        Args:
            args (dict): A dictionary of function arguments requiring:
                 'u_hat': A list of frequency coefficients--the DFT of a signal.
                 'dt': The time step between the discrete spatial measurements of the DFT.
                 
            resample (int): The factor to use to resample the signal if greater spatial resolution 
                            is needed e.g. for numerical integration
            
        Returns:
            (dict): a new resampled args dictionary
    '''
    n = len(args['u_hat'])
    dt = args['dt']/resample
    u_hat = np.zeros(resample*n, dtype=complex)
    if n % 2 == 0:
        # Frequencies [1, ..., n/2-1]
        u_hat[:n//2] = args['u_hat'][:n//2]
        # Frequenies [-n/2 + 1, ..., -1] <-- should we include n//2?
        u_hat[(-n//2+1):] = args['u_hat'][(-n//2+1):]
        u_hat[len(u_hat)//2] = args['u_hat'][n//2]
    else:
        # Frequenies [1, ..., (n-1)/2]
        u_hat[:(n+1)//2] = args['u_hat'][:(n+1)//2]
        # Frequenies [-(n-1)/2, ..., -1]
        u_hat[-(n-1)//2:] = args['u_hat'][-(n-1)//2:]
    u_hat = resample*u_hat
    return {'u_hat': u_hat, 'dt': dt}


def dft_coef_rand(n_pts, n_components, positive=False):
    # Generate random dft coefficients
    # Everything is easier if you pick odd n_pts (unsure about that middle value when even)
    u_hat = np.zeros(n_pts, dtype=complex)
    if n_components == 0:
        return u_hat
    if n_pts%2 == 0:
        m = n_pts//2-1
        if positive:
            coeffs = np.random.rand(m) + 1j*np.random.rand(m)
        else:
            coeffs = np.random.randn(m) + 1j*np.random.randn(m)
        ii = np.random.choice(np.arange(m), n_components, replace=False)
        # Frequencies [1, ...,   n/2-1]
        u_hat[1:n_pts//2][ii] = coeffs[ii]
        # Frequenies [-n/2 + 1, ...,   -1]
        u_hat[-n_pts//2+1:] = u_hat[1:n_pts//2][::-1].conj()
        # Real-valued frequencies
        u_hat[0] = 0
        u_hat[n_pts//2] = 0 # TODO ???
    else:
        m = n_pts//2
        if positive:
            coeffs = np.random.rand(m) + 1j*np.random.rand(m)
        else:
            coeffs = np.random.randn(m) + 1j*np.random.randn(m)
        ii = np.random.choice(np.arange(m), n_components, replace=False)
        # Frequenies [1, ...,   (n-1)/2]
        u_hat[1:(n_pts+1)//2][ii] = coeffs[ii]
        # Frequenies [-(n-1)/2, ...,   -1]
        u_hat[-(n_pts-1)//2:] = u_hat[1:(n_pts+1)//2][::-1].conj()
        # Real-valued frequencies
        u_hat[0] = 0
    return u_hat


def dft_coef_span(n_pts, noise=False, noise_ampl=1e-6):
    # Generate all single frequency dft coefficients
    if n_pts%2 == 0:
        m = n_pts//2-1
        u_hat_arr = np.zeros([2*m + 1, n_pts], dtype=complex)
        if noise:
            u_hat_arr[:, 1:n_pts//2] += noise_ampl*(np.random.rand(2*m + 1, n_pts//2-1) + (-1)**np.random.choice([1,2])*1j*np.random.rand(2*m + 1, n_pts//2-1))
        for loc in range(m):
            # Frequencies [1, ...,   n/2-1]
            # 1. Cosine
            u_hat_arr[1 + loc, 1 + loc] += n_pts/2
            # 2. Sine
            u_hat_arr[1 + m + loc, 1 + loc] += -1j*n_pts/2
        # Conjugate for frequenies [-n/2 + 1, ...,   -1]
        u_hat_arr[:, -n_pts//2+1:] = u_hat_arr[:, 1:n_pts//2][:, ::-1].conj()
    else:
        m = n_pts//2
        u_hat_arr = np.zeros([2*m + 1, n_pts], dtype=complex)
        if noise:
            u_hat_arr[:, 1:(n_pts+1)//2] += noise_ampl*(np.random.rand(2*m + 1, (n_pts+1)//2-1) + (-1)**np.random.choice([1,2])*1j*np.random.rand(2*m + 1, (n_pts+1)//2-1))
        for loc in range(m):
            # Frequenies [1, ...,   (n-1)/2]
            # 1. Cosine
            u_hat_arr[1+loc, 1 + loc] += n_pts/2
            # 2. Sine
            u_hat_arr[1+m + loc, 1 + loc] += -1j*n_pts/2
        # Conjugate for frequenies [-(n-1)/2, ...,   -1]
        u_hat_arr[:, -(n_pts-1)//2:] = u_hat_arr[:, 1:(n_pts+1)//2][:, ::-1].conj()
    return u_hat_arr


def multinomial_powers(n, k):
    '''
    Returns all combinations of powers of the expansion (x_1+x_2+...+x_k)^n.
    The motivation for the algorithm is to use dots and bars: 
    e.g.    For (x1+x2+x3)^3, count n=3 dots and k-1=2 bars. 
            ..|.| = [x1^2, x2^1, x3^0]

    Note: Add 1 to k to include a constant term, (1+x+y+z)^n, to get all
    groups of powers less than or equal to n (just ignore elem[0])
    
    Emphasis is on preserving yield behavior of combinatorial iterator.
        
    Arguments:
        n: the order of the multinomial_powers
        k: the number of variables {x_i}

    Yields:
        list: a list of multinomial powers
    '''
    for elem in combinations(np.arange(n+k-1), k - 1):
        elem = np.array([-1] + list(elem) + [n+k-1])
        yield elem[1:] - elem[:-1] - 1


def make_control_library(u_hat, order):
    '''
    Constructs a library of multinomials based on the coefficients in u_hat.

    Arguments:
        u_hat: discrete Fourier transform coefficients
        order: the maximum value of the multinomial exponent

    Returns:
        `ndarray`: a column vector for the control multinomial library up to the given order        
    '''
    n = len(u_hat)
    # Q: If n is even, do we do anything with that middle value?
    n_real = n//2 if n%2==0 else (n+1)//2
    # 1. Get the unique real and imaginary coefficients
    u_list = np.hstack([u_hat[1:n_real].real, u_hat[1:n_real].imag])
    # 2. Construct a library of the multinomials exponents
    u_multinomials = np.vstack([list(multinomial_powers(j,  len(u_list))) for j in range(1,order+1)])
    # 3. Combine the coefficients and the multinomial exponents
    result = []
    for row in u_multinomials:
        u_row = np.product(u_list.real**row)
        result.append([u_row])
    return np.vstack(result)