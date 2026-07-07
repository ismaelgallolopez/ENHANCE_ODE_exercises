"""Provided helper infrastructure for the Day 7 neural-network training lab.

Students may inspect this file, but the core exercises do not require reading
or editing it. The notebook keeps the actual student implementation tasks in
exercise-local code cells.
"""

import math

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

try:
    from jax.flatten_util import ravel_pytree
except ImportError:
    def ravel_pytree(pytree):
        """Small fallback for older or stripped-down JAX installations."""
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        shapes = [leaf.shape for leaf in leaves]
        sizes = [leaf.size for leaf in leaves]
        flat = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

        def unravel(flat_vector):
            chunks = []
            start = 0
            for shape, size in zip(shapes, sizes):
                chunks.append(jnp.reshape(flat_vector[start:start + size], shape))
                start += size
            return jax.tree_util.tree_unflatten(treedef, chunks)

        return flat, unravel

# -----------------------------------------------------------------------------
# Data and diagnostic frequency basis
# -----------------------------------------------------------------------------

def target_parts(x):
    """Return the three frequency components of the target."""
    low = 0.75 * jnp.sin(jnp.pi * x)
    medium = 0.35 * jnp.sin(4.0 * jnp.pi * x - 0.20)
    high = 0.18 * jnp.sin(9.0 * jnp.pi * x + 0.35)
    return jnp.stack([low, medium, high], axis=0)


def target_function(x):
    return jnp.sum(target_parts(x), axis=0)


def frequency_basis(x):
    return jnp.column_stack(
        [
            jnp.sin(jnp.pi * x),
            jnp.sin(4.0 * jnp.pi * x - 0.20),
            jnp.sin(9.0 * jnp.pi * x + 0.35),
        ]
    )


def make_training_data(key, n_train=96, n_test=401, noise_level=0.02):
    """Create one noisy training set and one clean test grid."""
    x_train = jnp.linspace(-1.0, 1.0, n_train)
    y_clean = target_function(x_train)
    x_test = jnp.linspace(-1.0, 1.0, n_test)
    return {
        "x_train": x_train,
        "y_train": y_clean + noise_level * jax.random.normal(key, x_train.shape),
        "y_clean_train": y_clean,
        "x_test": x_test,
        "y_test": target_function(x_test),
        "x_plot": x_test,
        "y_plot": target_function(x_test),
        "noise_level": noise_level,
    }

# -----------------------------------------------------------------------------
# ReLU networks
# -----------------------------------------------------------------------------

def relu(z):
    return jnp.maximum(z, 0.0)


def init_relu_mlp(key, layer_sizes, hidden_weight_scale=1.0, hidden_bias_scale=0.2, output_scale=0.2):
    """Initialize a small ReLU MLP as a list of layer dictionaries."""
    keys = jax.random.split(key, 2 * (len(layer_sizes) - 1))
    layers = []
    for layer_id, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        is_output = layer_id == len(layer_sizes) - 2
        scale = output_scale if is_output else hidden_weight_scale
        weight = scale * jax.random.normal(keys[2 * layer_id], (in_dim, out_dim)) / math.sqrt(in_dim)
        if is_output:
            bias = jnp.zeros((out_dim,))
        else:
            # This scale controls where many initial ReLU breakpoints lie.
            bias = hidden_bias_scale * jax.random.uniform(
                keys[2 * layer_id + 1], (out_dim,), minval=-1.0, maxval=1.0
            )
        layers.append({"W": weight, "b": bias})
    return {"layers": layers}


def apply_relu_mlp(params, x):
    """Evaluate the network on one-dimensional inputs of shape (n,)."""
    values = jnp.reshape(x, (-1, 1))
    for layer in params["layers"][:-1]:
        values = relu(values @ layer["W"] + layer["b"])
    final = params["layers"][-1]
    return jnp.ravel(values @ final["W"] + final["b"])


def standard_deep_init(key):
    """Default two-hidden-layer network used in Exercises 1b and 3."""
    return init_relu_mlp(key, (1, 48, 48, 1), 1.0, 0.2, 0.2)

# -----------------------------------------------------------------------------
# Loss, batches, and histories
# -----------------------------------------------------------------------------

def mse_loss(params, batch):
    residual = apply_relu_mlp(params, batch["x_train"]) - batch["y_train"]
    return jnp.mean(residual**2)


def batch_indices(key, num_samples, batch_size):
    """Return all indices for full-batch training, or a random mini-batch."""
    if batch_size is None or batch_size >= num_samples:
        return jnp.arange(num_samples)
    return jax.random.permutation(key, num_samples)[:batch_size]


def select_batch(data, indices):
    return {"x_train": data["x_train"][indices], "y_train": data["y_train"][indices]}


def parameter_count(params):
    flat, _ = ravel_pytree(params)
    return int(flat.size)


def frequency_coefficients(values, x_grid):
    """Project a curve onto the three sine modes used to build the target."""
    basis = frequency_basis(x_grid)
    return jnp.linalg.solve(basis.T @ basis + 1.0e-12 * jnp.eye(basis.shape[1]), basis.T @ values)


def compute_ntk_diagnostics_for_params(params, data):
    """Compute finite-width NTK eigenmodes for a fixed parameter tree.

    The NTK is evaluated on the training inputs. The frequency Rayleigh
    quotients are effective eigenvalues for the three sine modes used in the
    target. For a finite network these sine modes are not exact eigenvectors,
    but the quotients show which frequencies are favored by this NTK.
    """
    flat_params, unravel = ravel_pytree(params)
    x_train = data["x_train"]

    def model_values(flat_values):
        return apply_relu_mlp(unravel(flat_values), x_train)

    jacobian = jax.jacobian(model_values)(flat_params)
    kernel = (jacobian @ jacobian.T) / jacobian.shape[1]
    kernel = 0.5 * (kernel + kernel.T)

    eigenvalues, eigenvectors = jnp.linalg.eigh(kernel)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    basis = frequency_basis(x_train)
    rayleigh = []
    top_five_energy = []
    for mode in range(basis.shape[1]):
        vector = basis[:, mode]
        vector = vector / (jnp.linalg.norm(vector) + 1.0e-12)
        weights = (eigenvectors.T @ vector) ** 2
        rayleigh.append(float(jnp.sum(eigenvalues * weights)))
        top_five_energy.append(float(jnp.sum(weights[:5])))

    return {
        "params": params,
        "kernel": kernel,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "frequency_rayleigh": np.asarray(rayleigh),
        "frequency_top_five_energy": np.asarray(top_five_energy),
    }


def compute_ntk_diagnostics(experiment, data):
    """Initialize one experiment and compute its finite-width NTK."""
    key = jax.random.PRNGKey(experiment["config"].get("seed", 0))
    params0 = experiment["init_fn"](key)
    diagnostics = compute_ntk_diagnostics_for_params(params0, data)
    diagnostics["params0"] = params0
    return diagnostics


def print_ntk_frequency_table(ntk, label=None):
    """Print effective NTK eigenvalues for the target frequency modes."""
    if label is not None:
        print(label)
    labels = ["low", "medium", "high"]
    print(f"{'frequency':<12} {'effective eigenvalue':>22} {'energy in first 5 modes':>24}")
    print("-" * 62)
    for label, value, energy in zip(
        labels,
        ntk["frequency_rayleigh"],
        ntk["frequency_top_five_energy"],
    ):
        print(f"{label:<12} {value:>22.4e} {energy:>24.4f}")
def new_history():
    keys = ["step", "loss", "train_mse", "test_mse", "prediction"]
    keys += [f"mode_{mode}_coeff" for mode in range(3)]
    keys += [f"mode_{mode}_residual" for mode in range(3)]
    return {key: [] for key in keys}


def record(history, step, params, data, loss_value=None):
    """Store scalar diagnostics and the prediction curve at one step."""
    loss_value = mse_loss(params, data) if loss_value is None else loss_value
    prediction = apply_relu_mlp(params, data["x_plot"])
    coeffs = frequency_coefficients(prediction, data["x_plot"])
    target_coeffs = frequency_coefficients(data["y_plot"], data["x_plot"])
    residual_coeffs = target_coeffs - coeffs

    history["step"].append(int(step))
    history["loss"].append(float(loss_value))
    history["train_mse"].append(float(mse_loss(params, data)))
    history["test_mse"].append(float(jnp.mean((apply_relu_mlp(params, data["x_test"]) - data["y_test"]) ** 2)))
    history["prediction"].append(np.asarray(prediction))
    for mode, coeff in enumerate(coeffs):
        history[f"mode_{mode}_coeff"].append(float(coeff))
        history[f"mode_{mode}_residual"].append(float(jnp.abs(residual_coeffs[mode])))

# -----------------------------------------------------------------------------
# Provided optimizers
# -----------------------------------------------------------------------------

def train_first_order(params, data, config, method):
    """Shared implementation for gradient descent and Adam.

    If ``early_stopping_patience`` is provided, the full training loss is
    monitored whenever diagnostics are recorded. The returned parameters are the
    best recorded parameters, not necessarily the last iterate.
    """
    steps = config.get("steps", 1000)
    learning_rate = config.get("learning_rate", 1.0e-2)
    batch_size = config.get("batch_size")
    record_every = config.get("record_every", max(1, steps // 40))
    patience = config.get("early_stopping_patience")
    min_delta = config.get("early_stopping_min_delta", 0.0)
    key = jax.random.PRNGKey(config.get("batch_seed", 0))
    value_and_grad = jax.value_and_grad(mse_loss)
    history = new_history()

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    beta1, beta2, eps = config.get("beta1", 0.9), config.get("beta2", 0.999), config.get("eps", 1.0e-8)

    best_params = params
    best_loss = math.inf
    best_step = 0
    checks_without_improvement = 0
    stopped_step = None

    def record_and_check(step, params_now):
        nonlocal best_params, best_loss, best_step, checks_without_improvement
        loss_value = mse_loss(params_now, data)
        record(history, step, params_now, data, loss_value)
        if patience is None:
            return False

        current_loss = float(loss_value)
        if current_loss < best_loss - min_delta:
            best_params = params_now
            best_loss = current_loss
            best_step = int(step)
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        return checks_without_improvement >= patience

    record_and_check(0, params)
    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        batch = select_batch(data, batch_indices(subkey, data["x_train"].shape[0], batch_size))
        _, grads = value_and_grad(params, batch)

        if method == "gd":
            updates = jax.tree_util.tree_map(lambda g: -learning_rate * g, grads)
        else:
            m = jax.tree_util.tree_map(lambda old, g: beta1 * old + (1.0 - beta1) * g, m, grads)
            v = jax.tree_util.tree_map(lambda old, g: beta2 * old + (1.0 - beta2) * (g**2), v, grads)
            m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**step), m)
            v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**step), v)
            updates = jax.tree_util.tree_map(
                lambda m_leaf, v_leaf: -learning_rate * m_leaf / (jnp.sqrt(v_leaf) + eps),
                m_hat,
                v_hat,
            )
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)

        if step % record_every == 0:
            if record_and_check(step, params):
                stopped_step = step
                break

    final_step = stopped_step if stopped_step is not None else steps
    if history["step"][-1] != final_step:
        record_and_check(final_step, params)

    if patience is not None:
        params = best_params
        history["early_stopped"] = stopped_step is not None
        history["stopped_step"] = int(final_step)
        history["best_step"] = int(best_step)
        history["best_loss"] = float(best_loss)
    return params, history

def train_gd(params, data, config):
    return train_first_order(params, data, config, method="gd")


def train_adam(params, data, config):
    return train_first_order(params, data, config, method="adam")


def lbfgs_direction(grad, s_history, y_history):
    """Limited-memory BFGS two-loop recursion."""
    if not s_history:
        return -grad
    q, alphas, rhos = grad, [], []
    for s_vec, y_vec in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / (jnp.dot(y_vec, s_vec) + 1.0e-12)
        alpha = rho * jnp.dot(s_vec, q)
        q = q - alpha * y_vec
        alphas.append(alpha)
        rhos.append(rho)
    gamma = jnp.dot(s_history[-1], y_history[-1]) / (jnp.dot(y_history[-1], y_history[-1]) + 1.0e-12)
    r = gamma * q
    for (s_vec, y_vec), alpha, rho in zip(zip(s_history, y_history), reversed(alphas), reversed(rhos)):
        r = r + s_vec * (alpha - rho * jnp.dot(y_vec, r))
    return -r


def train_lbfgs(params, data, config):
    """Compact full-batch L-BFGS with backtracking line search."""
    steps = config.get("steps", 80)
    memory = config.get("memory", 8)
    initial_step = config.get("learning_rate", 1.0)
    record_every = config.get("record_every", 1)
    flat, unravel = ravel_pytree(params)

    def flat_loss(flat_params):
        return mse_loss(unravel(flat_params), data)

    value_and_grad = jax.value_and_grad(flat_loss)
    history, s_history, y_history = new_history(), [], []

    for step in range(steps):
        loss_value, grad = value_and_grad(flat)
        if step % record_every == 0:
            record(history, step, unravel(flat), data, loss_value)

        direction = lbfgs_direction(grad, s_history, y_history)
        if float(jnp.dot(grad, direction)) >= 0.0:
            direction = -grad

        step_length = initial_step
        for _ in range(12):
            trial = flat + step_length * direction
            trial_loss, trial_grad = value_and_grad(trial)
            decrease = loss_value + 1.0e-4 * step_length * jnp.dot(grad, direction)
            if float(trial_loss) <= float(decrease):
                break
            step_length *= 0.5

        s_vec, y_vec = trial - flat, trial_grad - grad
        if float(jnp.dot(s_vec, y_vec)) > 1.0e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            s_history, y_history = s_history[-memory:], y_history[-memory:]
        flat = trial

    params = unravel(flat)
    record(history, steps, params, data)
    return params, history


def train_gauss_newton(params, data, config):
    """Damped Gauss--Newton; damping plays the Levenberg--Marquardt role."""
    steps = config.get("steps", 25)
    damping = config.get("damping", 1.0e-2)
    record_every = config.get("record_every", 1)
    flat, unravel = ravel_pytree(params)

    def residual(flat_params):
        params_now = unravel(flat_params)
        return apply_relu_mlp(params_now, data["x_train"]) - data["y_train"]

    history = new_history()
    for step in range(steps):
        res = residual(flat)
        jacobian = jax.jacobian(residual)(flat)
        normal_matrix = jacobian.T @ jacobian
        step_vector = jnp.linalg.solve(
            normal_matrix + damping * jnp.eye(flat.size),
            jacobian.T @ res,
        )
        if step % record_every == 0:
            record(history, step, unravel(flat), data, jnp.mean(res**2))
        flat = flat - step_vector

    params = unravel(flat)
    record(history, steps, params, data)
    return params, history


OPTIMIZERS = {
    "gd": train_gd,
    "adam": train_adam,
    "lbfgs": train_lbfgs,
    "gauss_newton": train_gauss_newton,
}

# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def make_experiment(
    name,
    init_fn,
    optimizer,
    learning_rate=None,
    batch_size=None,
    steps=1000,
    seed=0,
    batch_seed=0,
    record_every=None,
    **optimizer_options,
):
    config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "steps": steps,
        "seed": seed,
        "batch_seed": batch_seed,
    }
    config.update(optimizer_options)
    if record_every is not None:
        config["record_every"] = record_every
    return {"name": name, "init_fn": init_fn, "optimizer": optimizer, "config": config}


def run_experiment(experiment, data):
    key = jax.random.PRNGKey(experiment["config"].get("seed", 0))
    params0 = experiment["init_fn"](key)
    params, history = OPTIMIZERS[experiment["optimizer"]](params0, data, experiment["config"])
    return {
        **experiment,
        "params0": params0,
        "params": params,
        "history": history,
        "n_params": parameter_count(params),
    }


def run_experiments(experiments, data):
    results = []
    for experiment in experiments:
        print(f"Training {experiment['name']} ...")
        results.append(run_experiment(experiment, data))
    return results

# -----------------------------------------------------------------------------
# Shared diagnostics
# -----------------------------------------------------------------------------

def prediction_errors(result, data):
    params = result["params"]
    train = jnp.mean((apply_relu_mlp(params, data["x_train"]) - data["y_train"]) ** 2)
    test = jnp.mean((apply_relu_mlp(params, data["x_test"]) - data["y_test"]) ** 2)
    return float(train), float(test)


def print_training_table(results, data):
    header = f"{'model':<28} {'opt':<13} {'lr':>9} {'batch':>8} {'params':>7} {'train mse':>12} {'test mse':>12}"
    print(header)
    print("-" * len(header))
    for result in results:
        train_mse, test_mse = prediction_errors(result, data)
        cfg = result["config"]
        lr = "-" if cfg.get("learning_rate") is None else f"{cfg['learning_rate']:.1e}"
        batch = "full" if cfg.get("batch_size") is None else str(cfg["batch_size"])
        print(f"{result['name']:<28} {result['optimizer']:<13} {lr:>9} {batch:>8} {result['n_params']:>7d} {train_mse:>12.4e} {test_mse:>12.4e}")


def print_early_stopping_summary(results):
    """Print where patience-based early stopping stopped each run."""
    rows = [result for result in results if result["config"].get("early_stopping_patience") is not None]
    if not rows:
        return
    header = f"{'model':<28} {'stopped?':>9} {'last step':>10} {'best step':>10} {'best loss':>12}"
    print(header)
    print("-" * len(header))
    for result in rows:
        history = result["history"]
        stopped = "yes" if history.get("early_stopped", False) else "no"
        last_step = history.get("stopped_step", result["config"].get("steps"))
        best_step = history.get("best_step", last_step)
        best_loss = history.get("best_loss", history["loss"][-1])
        print(f"{result['name']:<28} {stopped:>9} {last_step:>10d} {best_step:>10d} {best_loss:>12.4e}")


def draw_data(ax, data, show_samples=True):
    ax.plot(np.asarray(data["x_plot"]), np.asarray(data["y_plot"]), color="black", lw=2.2, label="clean target")
    if show_samples:
        ax.scatter(np.asarray(data["x_train"]), np.asarray(data["y_train"]), s=14, color="0.6", alpha=0.65, label="training data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)


def plot_training_data(data):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    draw_data(ax, data)
    ax.set_title("Mixed-frequency regression data")
    ax.legend(loc="best")
    plt.show()


def plot_model_fits(results, data, title):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    draw_data(ax, data)
    for result in results:
        y = apply_relu_mlp(result["params"], data["x_plot"])
        ax.plot(np.asarray(data["x_plot"]), np.asarray(y), lw=1.8, label=result["name"])
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.show()




def plot_loss_curves(results, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for result in results:
        history = result["history"]
        steps = np.asarray(history["step"])
        losses = np.asarray(history["loss"])
        label = result["name"]
        if result["optimizer"] == "gauss_newton":
            # Raw damped Gauss--Newton losses can oscillate because the local
            # least-squares model is only approximate. For the lab diagnostic we
            # plot the best-so-far envelope, which shows progress without the
            # distracting zig-zag trace.
            losses = np.minimum.accumulate(losses)
            label = f"{label} (best so far)"
        ax.semilogy(steps, losses, lw=1.8, label=label)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("full training loss")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.show()


def plot_frequency_learning(results, data, title):
    target_coeffs = frequency_coefficients(data["y_plot"], data["x_plot"])
    labels = ["low-frequency coefficient", "medium-frequency coefficient", "high-frequency coefficient"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for result in results:
        history = result["history"]
        steps = np.asarray(history["step"])
        for mode, ax in enumerate(axes):
            denominator = float(target_coeffs[mode])
            if abs(denominator) < 1.0e-12:
                denominator = 1.0
            values = np.asarray(history[f"mode_{mode}_coeff"]) / denominator
            ax.plot(steps, values, lw=1.8, label=result["name"])
    for ax, title_part in zip(axes, labels):
        ax.axhline(1.0, color="black", lw=1.0, ls="--")
        ax.set_title(title_part)
        ax.set_xlabel("optimization step")
        ax.set_ylabel("coefficient / target")
        ax.grid(True, alpha=0.25)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_ntk_eigenmode_comparison(
    ntk_specs,
    data,
    title,
    num_largest_modes=5,
    num_smallest_modes=5,
):
    """Lecture-style NTK comparison: fast modes, slow modes, and spectrum."""
    if isinstance(ntk_specs, dict):
        specs = [("NN at initialization", ntk_specs)]
    else:
        specs = list(ntk_specs)
    if not specs:
        raise ValueError("ntk_specs must contain at least one NTK diagnostic.")

    x = np.asarray(data["x_train"])
    all_values = [np.maximum(np.asarray(ntk["eigenvalues"]).reshape(-1), 1.0e-14) for _, ntk in specs]
    eig_min = min(float(values.min()) for values in all_values)
    eig_max = max(float(values.max()) for values in all_values)

    fig, axes = plt.subplots(len(specs), 3, figsize=(10.0, 2.55 * len(specs) + 1.0), sharex="col", squeeze=False)
    axes[0, 0].set_title(f"Largest {num_largest_modes} eigenmodes")
    axes[0, 1].set_title(f"Smallest {num_smallest_modes} eigenmodes")
    axes[0, 2].set_title("Eigenvalue distribution")

    for row, (row_label, ntk) in enumerate(specs):
        eigenvectors = np.asarray(ntk["eigenvectors"])
        eigenvalues = np.asarray(ntk["eigenvalues"]).reshape(-1)
        eigenvalues_clipped = np.maximum(eigenvalues, 1.0e-14)
        n_modes = eigenvectors.shape[1]
        largest = list(range(min(num_largest_modes, n_modes)))
        smallest = list(range(max(0, n_modes - num_smallest_modes), n_modes))

        def normalized_mode(mode_index):
            """Fix the arbitrary eigenvector sign and scale modes for comparison."""
            vector = eigenvectors[:, mode_index]
            sign = np.sign(vector[np.argmax(np.abs(vector))]) or 1.0
            return sign * vector / (np.max(np.abs(vector)) + 1.0e-12)

        for mode_index in largest:
            axes[row, 0].plot(
                x,
                normalized_mode(mode_index),
                lw=1.0,
                label=fr"$q_{{{mode_index + 1}}}$",
            )
        for mode_index in smallest:
            axes[row, 1].plot(
                x,
                normalized_mode(mode_index),
                lw=1.0,
                label=fr"$q_{{{mode_index + 1}}}$",
            )
        axes[row, 2].semilogy(
            np.arange(1, eigenvalues_clipped.shape[0] + 1),
            eigenvalues_clipped,
            color="black",
            lw=1.2,
        )
        axes[row, 2].set_ylim(0.9 * eig_min, 1.1 * eig_max)

        axes[row, 0].set_ylabel(f"{row_label}\n" + r"$q_i(x)$")
        axes[row, 1].set_ylabel(r"$q_i(x)$")
        axes[row, 2].set_ylabel(r"$\lambda_i$")
        for col in range(3):
            axes[row, col].grid(True, alpha=0.25)
        if row == 0:
            axes[row, 0].legend(fontsize=6, ncol=2, frameon=False, loc="upper right")
            axes[row, 1].legend(fontsize=6, ncol=2, frameon=False, loc="upper right")

    axes[-1, 0].set_xlabel(r"$x$")
    axes[-1, 1].set_xlabel(r"$x$")
    axes[-1, 2].set_xlabel("mode index $i$")
    fig.suptitle(title)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.90, bottom=0.10, hspace=0.38, wspace=0.28)
    plt.show()


def plot_spectral_bias_slide_diagnostics(
    result,
    data,
    title,
    snapshot_steps=(0, 100, 500, 1600),
):
    """Plot spectral-bias snapshots and frequency-wise residual reduction."""
    history = result["history"]
    steps = np.asarray(history["step"])
    predictions = np.asarray(history["prediction"])
    requested = np.asarray(snapshot_steps)
    snapshot_indices = []
    for step in requested:
        nearest = int(np.argmin(np.abs(steps - step)))
        if nearest not in snapshot_indices:
            snapshot_indices.append(nearest)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    target_line, = axes[0].plot(
        np.asarray(data["x_plot"]),
        np.asarray(data["y_plot"]),
        color="black",
        lw=2.6,
        label="target",
        zorder=1,
    )
    try:
        import matplotlib.patheffects as pe
        target_line.set_path_effects([pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()])
    except Exception:
        pass

    cmap = plt.get_cmap("viridis")
    for i, index in enumerate(snapshot_indices):
        color = cmap(i / max(1, len(snapshot_indices) - 1))
        axes[0].plot(
            np.asarray(data["x_plot"]),
            predictions[index],
            lw=1.25,
            color=color,
            alpha=0.78,
            label=f"step {int(steps[index])}",
            zorder=3,
        )
    axes[0].set_title("Prediction snapshots")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$f_\theta(x)$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2, frameon=True, framealpha=0.92, edgecolor="none")

    mode_labels = ["low", "medium", "high"]
    markers = ["o", "s", "^"]
    colors = plt.get_cmap("tab10")
    for mode, label in enumerate(mode_labels):
        residual = np.asarray(history[f"mode_{mode}_residual"])
        relative = residual / max(float(residual[0]), 1.0e-14)
        axes[1].semilogy(
            steps,
            relative,
            marker=markers[mode],
            markevery=max(1, len(steps) // 10),
            lw=1.8,
            color=colors(mode),
            label=f"{label} frequency",
        )
    axes[1].set_title("Residual reduction by frequency")
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel("relative amplitude")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
