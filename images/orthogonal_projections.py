import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def find_perpendicular_point(p_first,p_second,outer_point):
    v_x = p_second[0] - p_first[0]
    v_y = p_second[1] - p_first[1]
    u_x = outer_point[0] - p_first[0]
    u_y = outer_point[1] - p_first[1]

    if v_x==0 and v_y==0:
        return p_first
    else:
        per_point = [p_first[0],p_first[1]]
        inner_product = v_x * u_x + v_y * u_y
        lenSquare_v = v_x * v_x + v_y * v_y
        projected_v_x = inner_product / lenSquare_v * v_x
        projected_v_y = inner_product / lenSquare_v * v_y
        per_point[0] = projected_v_x + p_first[0]
        per_point[1] = projected_v_y + p_first[1]
        return per_point
 

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

#draw a vector
def draw_arrow(ax,start_point,end_point,annotation):
    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]
    ax.arrow(start_point[0],start_point[0],x_delta,y_delta, head_width=0.01, head_length=0.001, fc='k', ec='k')

np.random.seed(0)

PARAMETERS = {
    'Positive correlation': np.array([[0.15, 0.65],
                                      [0.65, 0.35]]),
    'Negative correlation': np.array([[0.9, -0.4],
                                      [0.1, -0.6]]),
    'Weak correlation': np.array([[1, 0],
                                  [0, 1]]),
    'Orthogonal Projection': np.array([[0.15, 0.65],
                                      [0.65, 0.35]]),
}

mu = -2,3
scale = 1, 1


fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
title = "Orthogonal Projection"

#measurement line
measurement_space = [(-3, -2.3), (2, 7)]
(measurement_xs, measurement_ys) = zip(*measurement_space)
ax.add_line(Line2D(measurement_xs, measurement_ys, linewidth=2, color='red'))
tex = r'$\Omega = \left\{ x|Hx=z \right\}$'
ax.text(0.7, 4, tex, fontsize=15, va='bottom')
#predict with noise
dependency = PARAMETERS.get(title)
x, y = get_correlated_dataset(800, dependency, mu, scale)
ax.scatter(x, y, s=0.5)
ax.axvline(c='grey', lw=1)
ax.axhline(c='grey', lw=1)
confidence_ellipse(x, y, ax, edgecolor='blue')
ax.scatter(mu[0], mu[1], c='red', s=5)
tex = r'$\hat x_{t| t-1}$'
ax.text(-2.5, 3, tex, fontsize=20, va='bottom')
tex = r'$Covariance : Q$'
ax.text(-2.5, 5.5, tex, fontsize=20, va='bottom')
#title of the plot
ax.set_title(title)

plt.axis('off')
plt.show()