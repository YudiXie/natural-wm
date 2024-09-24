import numpy as np

from scipy import stats
from scipy import special
from scipy import optimize


def fit_vonmises_uniform(data, range=180):

    scale = range / np.pi
    data = np.array(data) / scale
    # print(np.abs(data).mean())

    def negative_loglikelihood(parameters):
        pm, mu, kappa = parameters
        prob = (1 - pm) / 2 / np.pi + pm * stats.vonmises.pdf(data, 1 / kappa, loc=mu)
        # print(stats.vonmises.pdf([0, -1, 1], 1 / kappa, loc=mu))
        neg_log_prob = - np.log(prob).sum()
        # print(parameters, neg_log_prob)
        return neg_log_prob

    kappa0 = stats.circstd(data, low=-np.pi, high=np.pi)
    
    mle_model = optimize.minimize(
        negative_loglikelihood,
        (0.1, 0, 1), 
        bounds=((0, 1), (-np.pi, np.pi), (0.01, None)),
        method='L-BFGS-B'
    )

    # print(mle_model)
    return (mle_model.x[0], np.sqrt(mle_model.x[2]) * scale, kappa0 * scale)


def fit_power_law(x, y):

    def power_law(x, c, p):
        return x ** -p * c

    popt, pcov = optimize.curve_fit(power_law, x, y)

    x_axis = np.linspace(min(x), max(x), 100)
    return x_axis, power_law(x_axis, *popt), popt


def von_mises_pdf(kappa, mu, theta):
    """von Mises pdf,
    kappa is a scalar, kappa > 0,
    mu is a scalar in (-pi, pi),
    theta is a scalar or a vector in (-pi, pi)"""
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * special.iv(0, kappa))


def von_mises_uniform_mixture_pdf(w, kappa, mu, theta):
    """
    pdf of a mixture of a von Mises and a uniform distribution
    w, is the weighting factor in [0, 1]
    kappa is a scalar, kappa > 0,
    mu is a scalar in (-pi, pi),
    x is a scalar or a vector in (-pi, pi)"""
    return w * von_mises_pdf(kappa, mu, theta) + (1 - w) * (1 / (2 * np.pi))


def neg_logllhd_func(ds):
    """return negative log-likelihood function
    for a mixture of a von Mises and a uniform distribution
    given a set of data points centered at 0"""
    def neg_logllhd_vmunifmixture(x):
        # x is ndarray of shape (2,)
        # x[0] is w, x[1] is kappa
        return - np.sum(np.log(von_mises_uniform_mixture_pdf(x[0], x[1], 0.0, ds)))
    return neg_logllhd_vmunifmixture


def center_angle(angles):
    """
    return an array of angles with center at 0,
    in the range of (-pi, pi)
    :param angles: ndarray of angles, in range of (-pi, pi)
    :return: ndarray of angles, in range of (-pi, pi)
    """
    angles = angles - stats.circmean(angles, high=np.pi, low=-np.pi)
    return (angles + np.pi) % (2 * np.pi) - np.pi


def fit_von_mises_uniform_mixture(errors_centered):
    """
        fit von Mises + uniform distribution to the error distribution
    :param errors_centered: 1d ndarray of centered errors
    :return: w, kappa, csd
    """
    res = optimize.minimize(neg_logllhd_func(errors_centered),
                            np.array((0.5, 5)),
                            method='Nelder-Mead',
                            bounds=((0.0, 1.0), (0.0, None)))
    w, kappa = res.x
    csd = np.sqrt(1 - (special.iv(1, kappa) / special.iv(0, kappa)))

    if not res.success:
        print(f'Optimization success: {res.success}')
        print(f'message: {res.message}')
        print(f'w: {w}, kappa: {kappa}')

    return w, kappa, csd


def get_density(errors, x_axis):
    density = []
    interval = x_axis[1] - x_axis[0]
    for x in x_axis:
        density.append(((errors > x - interval / 2) &
                        (errors <= x + interval / 2)).mean() / interval)

    return np.array(density)


def get_density_fit_and_residual(errors_centered, x_axis, w, kappa):
    density = get_density(errors_centered * 180 / np.pi, x_axis)
    mixture_fit = von_mises_uniform_mixture_pdf(w, kappa, 0.0, x_axis * np.pi / 180) * np.pi / 180
    residuals = density - mixture_fit
    return density, mixture_fit, residuals
