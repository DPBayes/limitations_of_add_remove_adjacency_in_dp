import numpy as np
from numpy import power as pow
from numpy import sqrt
from prv_accountant.privacy_random_variables.abstract_privacy_random_variable import (
    PrivacyRandomVariable,
)
from prv_accountant.privacy_random_variables.poisson_subsampled_gaussian_mechanism import (
    _compute_log_a,
    log,
)
from scipy import special
from scipy.special import erfc
import scipy.stats as stats

# adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/privacy_random_variables/poisson_subsampled_gaussian_mechanism.py

M_SQRT2 = sqrt(np.longdouble(2))
M_PI = np.pi
M_LOG2 = log(np.longdouble(2))


class PoissonSubsampledGaussianMechanismSubstitute(PrivacyRandomVariable):
    def __init__(self, sampling_probability: float, noise_multiplier: float) -> None:
        """
        Following Koskela et al. 2020 (NOTE denoting the argument for inverse PL function as t instead of s),
        we have:
            L^{-1}_{X/Y}(t) = sigma^2 { - log(2) - log(c) + log[-(1-p)(1-e^t) + sqrt((1-p)^2 (1-exp(t))^2 + 4c^2 exp(t))] }
        where
            c = p exp(-1 / (2 sigma^2)) => log c = log p - 1/(2 sigma^2)

        Now, we want to form the CDF for the PRV, i.e. we are interested in
            Pr(L_{X/Y}(T) < t), T ~ q N(1, sigma^2) + (1-q) N(0, sigma^2)
        Hence the CDF can be written as
            Pr(L_{X/Y}(T) < t) = Pr(T < L^{-1}_{X/Y}(t))
                               = q Pr(sigma Z + 1 < L^{-1}_{X/Y}(t)) + (1-q) Pr(sigma Z < L^{-1}_{X/Y}(t)), Z ~ N(0,1)
                               = q Phi(L^{-1}_{X/Y}(t) / sigma - 1 / sigma) + (1-q) Phi(L^{-1}_{X/Y}(t) / sigma)

        We have
            L^{-1}_{X/Y}(t) / sigma = sigma { - log(2) - log(c) + log[-(1-p)(1-e^t) + sqrt((1-p)^2 (1-exp(t))^2 + 4c^2 exp(t))] }
                                    = sigma { - log(2) - log(p) + 1 / (2 sigma^2) + log[-(1-p)(1-e^t) + sqrt((1-p)^2 (1-exp(t))^2 + 4c^2 exp(t))] }
                                    = 1 / (2 sigma) + sigma { - log(2) - log(p) + log[-(1-p)(1-e^t) + sqrt((1-p)^2 (1-exp(t))^2 + 4c^2 exp(t))] }

        Denoting
            r := - log(2) - log(p) + log[-(1-p)(1-e^t) + sqrt((1-p)^2 (1-exp(t))^2 + 4c^2 exp(t))],
        we have
            L^{-1}_{X/Y}(t) / sigma = sigma r + 1 / (2 sigma)
        and
            Pr(T < L^{-1}_{X/Y}(t)) = q Phi(sigma r + 1 / (2 sigma) - 1 / sigma) + (1-q) Phi(sigma r + 1 / (2 sigma))
                                    = q Phi([ sigma^2 r - 1 / 2 ] / sigma) + (1-q) Phi([ sigma^2 r + 1 / 2 ] / sigma)

        We use the error function to implement the CDF: Phi(x) = 1/2 [1 + erf(x / sqrt(2))] = 1/2 [2 - erfc(x / sqrt(2))] = 1 - erfc(x / sqrt(2)) / 2
            Pr(T < L^{-1}_{X/Y}(t)) = 1 - (q / 2) erfc(1/sqrt(2 sigma^2) [sigma^2 r - 1 / 2]) - ((1-q) / 2) erfc( 1/sqrt(2 sigma^2) [sigma^2 r + 1 / 2])
        """
        self.p = np.longdouble(sampling_probability)
        self.sigma = np.longdouble(noise_multiplier)

    def pdf(self, t: float) -> float:
        sigma = self.sigma
        p = self.p

        log_p = log(p)

        r_term3 = (1.0 - p) * (1.0 - np.exp(t))
        r = (
            -M_LOG2
            - log_p
            + log(
                -r_term3 + np.sqrt(r_term3**2 + 4.0 * p**2 * np.exp(t - 1.0 / sigma**2))
            )
        )

        comp0_pdf = stats.norm(0.0, 1.0).pdf(sigma * r - 0.5 / sigma)
        comp1_pdf = stats.norm(0.0, 1.0).pdf(sigma * r + 0.5 / sigma)


    def cdf(self, t: float) -> float:
        sigma = self.sigma  # DP noise-scale
        p = self.p  # subsampling prob.
        log_p = log(p)

        r_term3 = (1.0 - p) * (1.0 - np.exp(t))
        r = (
            -M_LOG2
            - log_p
            + log(
                -r_term3 + np.sqrt(r_term3**2 + 4.0 * p**2 * np.exp(t - 1.0 / sigma**2))
            )
        )

        # erfc_arg0 = np.double(sigma**2 * r  / (M_SQRT2 * sigma))
        erfc_arg0 = np.double(sigma * r / M_SQRT2)
        erfc_arg1 = np.double(0.5 / (M_SQRT2 * sigma))
        cdf_summand0 = 1
        cdf_summand1 = -0.5 * p * erfc(erfc_arg0 - erfc_arg1)
        cdf_summand2 = -0.5 * (1.0 - p) * erfc(erfc_arg0 + erfc_arg1)

        return cdf_summand0 + cdf_summand1 + cdf_summand2

    def mean(self):
        raise NotImplementedError("Mean computation not implemented")

    def rdp(self, alpha: float) -> float:
        """
        This is now an overapproximation of the RDP using add remove with half the sigma
        """
        if self.p == 0:
            return 0

        if self.p == 1.0:
            return alpha / (2 * (self.sigma / 2.0) ** 2)

        if np.isinf(alpha):
            return np.inf

        return _compute_log_a(np.double(self.p), np.double(self.sigma / 2.0), alpha) / (
            alpha - 1
        )

    def rdp_wor(self, alpha: float) -> float:
        """
        Compute RDP of this mechanism of order alpha.

        Following Theorem 9 from http://proceedings.mlr.press/v89/wang19b/wang19b.pdf.
        This result applies for WOR sampling, while we are interested in Poisson subsampling.
        However, the RDP bounds for the WOR should dominate the ones from Poisson subsampling,
        and hence it is appropriate to use the WOR results to find the domain for the corresponding
        PRV.
        """

        def epsilon(order):
            """
            RDP epsilon for M(X) = f(X) + C * sigma * N(0, I)
            where ||f(X)|| < C, under the substitute relation
            """
            return alpha * 2 / self.sigma**2

        def higher_order_terms(j):
            return (
                M_LOG2 + j * np.log(self.p) + log_binom(alpha, j) + (j - 1) * epsilon(j)
            )

        # compute the logartihms of the summands
        first_term = 0
        exp_epsilon2 = np.exp(epsilon(2))
        second_term = (
            2 * np.log(self.p)
            + log_binom(alpha, 2)
            + min(2 * M_LOG2 + np.log(exp_epsilon2 - 1), M_LOG2 + epsilon(2))
        )
        rest = [higher_order_terms(j) for j in range(3, alpha + 1)]

        return special.logsumexp([first_term, second_term] + rest) / (alpha - 1)


def log_binom(n, k):
    return (
        special.loggamma(n + 1) - special.loggamma(k + 1) - special.loggamma(n - k + 1)
    )
