
class GaussianCopula(Bivariate):
    """Class for Gaussian copula model."""

    copula_type = CopulaTypes.GAUSSIAN
    theta_interval = [-1, 1]  # Correlation coefficient range
    invalid_thetas = []

    def generator(self, t):
        """Gaussian copulas do not have a generator function."""
        raise NotImplementedError("Gaussian copulas do not use a generator function.")

    def probability_density(self, X):
        r"""Compute the probability density function (PDF) for the Gaussian copula.

        The PDF for the Gaussian copula is given by:

        .. math:: c(u, v) = \frac{\phi_{\Sigma}(\Phi^{-1}(u), \Phi^{-1}(v))}{\phi(\Phi^{-1}(u)) \phi(\Phi^{-1}(v))}

        where:
        - :math:`\phi_{\Sigma}` is the multivariate normal PDF with correlation matrix :math:`\Sigma`.
        - :math:`\Phi^{-1}` is the inverse CDF (quantile function) of the standard normal distribution.

        Args:
            X (numpy.ndarray): Input data (U, V) in the unit hypercube.

        Returns:
            numpy.ndarray: Probability density for the input values.
        """
        self.check_fit()

        U, V = split_matrix(X)
        U_inv = norm.ppf(U)
        V_inv = norm.ppf(V)

        # Multivariate normal PDF
        mvn_pdf = multivariate_normal.pdf(
            np.column_stack((U_inv, V_inv)), mean=[0, 0], cov=self.theta
        )

        # Marginal normal PDFs
        marginal_pdf_u = norm.pdf(U_inv)
        marginal_pdf_v = norm.pdf(V_inv)

        return mvn_pdf / (marginal_pdf_u * marginal_pdf_v)

    def cumulative_distribution(self, X):
        r"""Compute the cumulative distribution function (CDF) for the Gaussian copula.

        The CDF for the Gaussian copula is given by:

        .. math:: C(u, v) = \Phi_{\Sigma}(\Phi^{-1}(u), \Phi^{-1}(v))

        where:
        - :math:`\Phi_{\Sigma}` is the multivariate normal CDF with correlation matrix :math:`\Sigma`.
        - :math:`\Phi^{-1}` is the inverse CDF (quantile function) of the standard normal distribution.

        Args:
            X (numpy.ndarray): Input data (U, V) in the unit hypercube.

        Returns:
            numpy.ndarray: Cumulative probability for the input values.
        """
        self.check_fit()

        U, V = split_matrix(X)
        U_inv = norm.ppf(U)
        V_inv = norm.ppf(V)

        return multivariate_normal.cdf(
            np.column_stack((U_inv, V_inv)), mean=[0, 0], cov=self.theta
        )

    def percent_point(self, y, V):
        """Compute the inverse of the conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y (numpy.ndarray): Value of :math:`C(u|v)`.
            V (numpy.ndarray): Given value of v.

        Returns:
            numpy.ndarray: Inverse conditional CDF values.
        """
        self.check_fit()

        # Transform V to the standard normal scale
        V_inv = norm.ppf(V)

        # Compute the conditional mean and variance
        rho = self.theta[0, 1]
        conditional_mean = rho * V_inv
        conditional_variance = 1 - rho**2

        # Transform y to the standard normal scale
        y_inv = norm.ppf(y)

        # Compute the inverse conditional CDF
        return norm.cdf((y_inv - conditional_mean) / np.sqrt(conditional_variance))

    def partial_derivative(self, X):
        r"""Compute the partial derivative of the cumulative distribution.

        The partial derivative of the Gaussian copula is the conditional CDF:

        .. math:: F(v|u) = \Phi\left(\frac{\Phi^{-1}(v) - \rho \Phi^{-1}(u)}{\sqrt{1 - \rho^2}}\right)

        Args:
            X (numpy.ndarray): Input data (U, V) in the unit hypercube.

        Returns:
            numpy.ndarray: Partial derivatives (conditional CDF values).
        """
        self.check_fit()

        U, V = split_matrix(X)
        U_inv = norm.ppf(U)
        V_inv = norm.ppf(V)

        rho = self.theta[0, 1]
        conditional_mean = rho * U_inv
        conditional_variance = 1 - rho**2

        return norm.cdf((V_inv - conditional_mean) / np.sqrt(conditional_variance))

    def compute_theta(self):
        r"""Compute the correlation parameter (theta) using Kendall's tau.

        For the Gaussian copula, the relationship between Kendall's tau and the correlation
        coefficient (rho) is given by:

        .. math:: \tau = \frac{2}{\pi} \arcsin(\rho)

        Solving for rho:

        .. math:: \rho = \sin\left(\frac{\pi}{2} \tau\right)

        Returns:
            float: Correlation coefficient (rho).
        """
        return np.sin(np.pi / 2 * self.tau)