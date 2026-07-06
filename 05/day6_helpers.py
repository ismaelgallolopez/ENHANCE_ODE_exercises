"""Shared helper code for the Day 6 regression notebook.

The notebooks keep the model definitions and student tasks local to the
exercise sections. This module contains only provided infrastructure: data
creation, training, evaluation, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random

# Keep figures readable in both Jupyter and Quarto-rendered versions.
plt.rcParams.update({"figure.figsize": (7.2, 3.8), "axes.grid": True})

# Day 6 is about regression models and overfitting, not optimizer behaviour.
# Set this to True if you want to inspect Adam loss histories while debugging.
SHOW_OPTIMIZATION_PLOTS = False

# Week 1 mass-spring-damper parameters. With these values the oscillator is
# underdamped, so the displacement is a decaying oscillation.
mass = 1.0
stiffness = 9.0
damping = 0.7
y0 = 1.0
v0 = 0.0
T = 5.0

# Constants for the closed-form underdamped solution.
gamma = damping/(2.0*mass)
omega0 = np.sqrt(stiffness/mass)
omega_d = np.sqrt(omega0**2 - gamma**2)


def mass_spring_damper_displacement(t):
    """Closed-form displacement for the underdamped Week 1 oscillator.

    The exact formula is used only to make data. During the regression tasks,
    the models only see samples of this curve.
    """
    t = np.asarray(t)
    amplitude_cos = y0
    amplitude_sin = (v0 + gamma*y0)/omega_d
    oscillation = amplitude_cos*np.cos(omega_d*t) + amplitude_sin*np.sin(omega_d*t)
    return np.exp(-gamma*t)*oscillation


def scale_time(t):
    """Map physical time t in [0, T] to the model input x in [-1, 1].

    This simple scaling makes polynomial powers and neural-network weights
    better conditioned than if we trained directly on t in [0, 5].
    """
    return 2.0*np.asarray(t)/T - 1.0


def make_regression_data(signal_fn, noise_level=0.0, seed=0, n_train=34, n_test=180, n_plot=500):
    """Sample a one-dimensional signal and add optional observation noise.

    The returned dictionary keeps both the physical time `t` and the scaled
    model input `x`. The model trains on `y_train`, which may be noisy. The
    test and plotting targets stay clean so that the diagnostics measure how
    well a model recovers the underlying displacement, not just the samples.
    """
    local_rng = np.random.default_rng(seed)

    # The training set is deliberately sparse. The denser arrays are only used
    # for clean test errors and smooth plots.
    t_train = np.linspace(0.0, T, n_train)
    t_test = np.linspace(0.025*T, 0.975*T, n_test)
    t_plot = np.linspace(0.0, T, n_plot)

    y_train_clean = signal_fn(t_train)
    noise = noise_level*local_rng.standard_normal(y_train_clean.shape)
    y_train = y_train_clean + noise

    return {
        "t_train": t_train,
        "x_train": scale_time(t_train),
        "y_train": y_train,
        "y_train_clean": y_train_clean,
        "t_test": t_test,
        "x_test": scale_time(t_test),
        "y_test": signal_fn(t_test),
        "t_plot": t_plot,
        "x_plot": scale_time(t_plot),
        "y_plot": signal_fn(t_plot),
        "noise_level": noise_level,
    }


clean_data = make_regression_data(mass_spring_damper_displacement, noise_level=0.0, seed=20260706)
medium_noise_data = make_regression_data(mass_spring_damper_displacement, noise_level=0.06, seed=20260706)
# The high-noise case is a deliberate stress test for flexible models.
high_noise_data = make_regression_data(mass_spring_damper_displacement, noise_level=0.40, seed=3)


def rmse(prediction, target):
    """Root-mean-square error used for all model comparisons below."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    return np.sqrt(np.mean((prediction - target)**2))


def mean_squared_error_loss(params, apply_model, x, y):
    """Default regression loss: one half times the mean squared residual."""
    residual = apply_model(params, x) - y
    return 0.5*jnp.mean(residual**2)


def make_experiment(
    name,
    data,
    init_model,
    apply_model,
    loss=mean_squared_error_loss,
    seed=0,
    steps=1000,
    learning_rate=2e-2,
):
    """Bundle the modelling choices and training options for one run.

    The first four entries are the pieces we compare in the exercises: data,
    initialization, forward model, and loss. Everything else is shared
    bookkeeping for reproducible training.
    """
    return {
        "name": name,
        "data": data,
        "init_model": init_model,
        "apply_model": apply_model,
        "loss": loss,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
    }


def train_parameters(init_model, apply_model, loss, data, seed, steps, learning_rate, record_every=25):
    """Train one parameter pytree with Adam.

    `apply_model(params, x)` should map an input array of shape (n, 1) to a
    prediction array of shape (n,). The optimizer itself is intentionally
    shared across all exercises; only the model and loss change.
    """
    # Convert the training data once. NumPy remains convenient for plotting;
    # JAX arrays are needed for automatic differentiation.
    x = jnp.asarray(data["x_train"][:, None])
    y = jnp.asarray(data["y_train"])
    params = init_model(random.PRNGKey(seed))

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Adam keeps two moment estimates with the same pytree structure as params.
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    # JAX differentiates the scalar loss with respect to every array in params.
    value_and_grad = jax.value_and_grad(lambda current_params: loss(current_params, apply_model, x, y))

    @jax.jit
    def adam_step(params, m, v, step_index):
        loss_value, grads = value_and_grad(params)
        m = jax.tree_util.tree_map(lambda old_m, g: beta1*old_m + (1.0 - beta1)*g, m, grads)
        v = jax.tree_util.tree_map(lambda old_v, g: beta2*old_v + (1.0 - beta2)*(g*g), v, grads)
        m_hat = jax.tree_util.tree_map(lambda value: value/(1.0 - beta1**step_index), m)
        v_hat = jax.tree_util.tree_map(lambda value: value/(1.0 - beta2**step_index), v)
        params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - learning_rate*mh/(jnp.sqrt(vh) + eps),
            params,
            m_hat,
            v_hat,
        )
        return params, m, v, loss_value

    history_steps = [0]
    history_losses = [float(loss(params, apply_model, x, y))]

    for step in range(1, steps + 1):
        params, m, v, _ = adam_step(params, m, v, jnp.asarray(step))
        if step % record_every == 0 or step == steps:
            history_steps.append(step)
            history_losses.append(float(loss(params, apply_model, x, y)))

    history = {"step": np.array(history_steps), "loss": np.array(history_losses)}
    return params, history


def run_experiment(experiment):
    """Train one experiment and attach a common prediction interface."""
    params, history = train_parameters(
        experiment["init_model"],
        experiment["apply_model"],
        experiment["loss"],
        experiment["data"],
        experiment["seed"],
        experiment["steps"],
        experiment["learning_rate"],
    )

    def predict(x_values, params=params, apply_model=experiment["apply_model"]):
        # Every model in this notebook expects a column input with shape (n, 1).
        x_column = jnp.asarray(np.asarray(x_values)[:, None])
        return np.asarray(apply_model(params, x_column))

    trained = dict(experiment)
    trained.update({"params": params, "history": history, "predict": predict})
    return trained


def run_experiments(experiments):
    """Train several experiments with the same optimizer."""
    return [run_experiment(experiment) for experiment in experiments]


def model_predict(model, x_values):
    """Evaluate a trained model using its shared prediction wrapper."""
    return np.asarray(model["predict"](x_values))


def model_errors(model, data=None):
    """Return RMSE on observations and on the clean test signal."""
    if data is None:
        data = model["data"]
    train_error = rmse(model_predict(model, data["x_train"]), data["y_train"])
    test_error = rmse(model_predict(model, data["x_test"]), data["y_test"])
    return train_error, test_error


def print_model_table(models, data=None):
    """Print a compact train/test error table for trained models."""
    print(f"{'model':<42} | {'train RMSE':>10} | {'test RMSE':>9}")
    print("-"*70)
    for model in models:
        train_error, test_error = model_errors(model, data)
        print(f"{model['name']:<42} | {train_error:10.4f} | {test_error:9.4f}")


def plot_dataset(data, ax=None):
    """Draw the clean reference curve and the training observations."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(data["t_plot"], data["y_plot"], "k--", linewidth=2, label="clean reference")
    noisy = data["noise_level"] > 0.0
    color = "tab:red" if noisy else "tab:blue"
    label = "noisy training data" if noisy else "clean training data"
    ax.scatter(data["t_train"], data["y_train"], s=28, color=color, zorder=3, label=label)
    ax.set_xlabel("time t")
    ax.set_ylabel("displacement y(t)")
    return ax


def plot_datasets(datasets, titles, title=None):
    """Plot several regression datasets with one consistent layout."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8*len(datasets), 3.8), squeeze=False)
    for ax, data, panel_title in zip(axes[0], datasets, titles):
        plot_dataset(data, ax=ax)
        ax.set_title(panel_title)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_model_fits(models, data, title):
    """Overlay trained models on the same dataset and clean reference curve."""
    fig, ax = plt.subplots()
    plot_dataset(data, ax=ax)
    for model in models:
        ax.plot(data["t_plot"], model_predict(model, data["x_plot"]), label=model["name"])
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_training_histories(models, title="Training histories", show=None):
    """Optionally plot Adam loss histories.

    These plots are useful for debugging and for connecting to the Day 7
    optimization lecture, but they are not central to this regression lab.
    By default they are hidden through `SHOW_OPTIMIZATION_PLOTS`.
    """
    if show is None:
        show = SHOW_OPTIMIZATION_PLOTS
    if not show:
        return

    fig, ax = plt.subplots()
    for model in models:
        history = model["history"]
        ax.semilogy(history["step"], history["loss"], label=model["name"])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    plt.show()
