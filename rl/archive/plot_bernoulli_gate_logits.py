#%%
"""Plot the Bernoulli gate's on/off probabilities versus its logit."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')

def main() -> None:
    logits = np.linspace(-5, 5, 401)
    p_on = 1.0 / (1.0 + np.exp(-logits))
    _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True, dpi=200)
    ax.plot(logits, p_on, label=r"$P(z=1\mid\ell)=\sigma(\ell)$")
    ax.plot(logits, 1 - p_on, label=r"$P(z=0\mid\ell)=1-\sigma(\ell)$")
    ax.axvline(0, color="0.7", lw=1)
    ax.set(xlabel=r"Gate logit $\ell$", ylabel="Probability",
           title="Bernoulli gate probabilities", xlim=(-5, 5), ylim=(0, 1.05))
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()


#%%
"""Plot softplus and its ReLU limit for the coupling-strength action."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_softplus() -> None:
    x = np.linspace(-8, 8, 501)
    softplus = np.logaddexp(0, x)
    _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True, dpi=200)
    ax.plot(x, softplus, label=r"$\mathrm{softplus}(x)=\log(1+e^x)$")
    ax.plot(x, np.maximum(x, 0), "--", color="0.5", label=r"$\max(0,x)$ (ReLU)")
    # ax.scatter([0], [np.log(2)], color="C0", zorder=3)
    # ax.annotate(r"$\mathrm{softplus}(0)=\log 2$", (0, np.log(2)),
    #             xytext=(0.6, 1.5), arrowprops={"arrowstyle": "->"})
    ax.axvline(0, color="0.8", lw=1)
    ax.set(xlabel=r"Raw action $x$", ylabel="Output", title="Softplus",
           xlim=(-8, 8), ylim=(-0.5, 8.5))
    ax.legend(loc="upper left")
    plt.grid(alpha=0.2)
    plt.show()


if __name__ == "__main__":
    plot_softplus()
