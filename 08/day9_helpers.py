"""Provided helper infrastructure for the Day9 PINN lab.

Students may inspect this file, but the core exercises do not require reading
or editing it. The notebook keeps the actual student implementation tasks in
exercise-local code cells.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FIGSIZE = (10.5, 3.6)


def init_tanh_mlp(key, input_dim, hidden_layers=(24, 24), output_dim=1):
    """Initialize a small fully connected tanh network."""
    layer_dims = [input_dim, *hidden_layers, output_dim]
    layer_keys = random.split(key, len(layer_dims) - 1)
    params = []
    for layer_key, in_dim, out_dim in zip(layer_keys, layer_dims[:-1], layer_dims[1:]):
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        weight = random.uniform(layer_key, (in_dim, out_dim), minval=-limit, maxval=limit)
        bias = jnp.zeros((out_dim,), dtype=jnp.float64)
        params.append({"W": weight, "b": bias})
    return tuple(params)


def apply_tanh_mlp(params, points):
    """Evaluate the tanh MLP on a batch of points."""
    values = points
    for layer in params[:-1]:
        values = jnp.tanh(values @ layer["W"] + layer["b"])
    return values @ params[-1]["W"] + params[-1]["b"]


def mlp_scalar(params, point):
    """Evaluate a scalar-output MLP on one point of shape (input_dim,)."""
    return apply_tanh_mlp(params, point.reshape(1, -1))[0, 0]


def ode_soft_trial(params, point):
    return mlp_scalar(params, point)


def poisson_soft_trial(params, point):
    return mlp_scalar(params, point)


def heat_soft_trial(params, point):
    return mlp_scalar(params, point)


def exact_ode(t_points, kappa):
    t = t_points.reshape(-1, 1)
    return jnp.sin(kappa * jnp.pi * t)


def make_ode_collocation(num_points):
    return jnp.linspace(0.0, 1.0, num_points).reshape(-1, 1)


def exact_poisson_2d(points):
    return (jnp.sin(jnp.pi * points[:, 0]) * jnp.sin(jnp.pi * points[:, 1])).reshape(-1, 1)


def poisson_rhs(points):
    return (
        2.0
        * jnp.pi**2
        * jnp.sin(jnp.pi * points[:, 0])
        * jnp.sin(jnp.pi * points[:, 1])
    ).reshape(-1, 1)


def make_square_interior_grid(num_axis):
    x = jnp.linspace(0.0, 1.0, num_axis + 2)[1:-1]
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    return jnp.stack([X.ravel(), Y.ravel()], axis=1)


def make_square_boundary_points(num_edge):
    s = jnp.linspace(0.0, 1.0, num_edge)
    left = jnp.stack([jnp.zeros_like(s), s], axis=1)
    right = jnp.stack([jnp.ones_like(s), s], axis=1)
    bottom = jnp.stack([s, jnp.zeros_like(s)], axis=1)
    top = jnp.stack([s, jnp.ones_like(s)], axis=1)
    return jnp.concatenate([left, right, bottom, top], axis=0)


def exact_heat(points, nu=0.1):
    t = points[:, 0]
    x = points[:, 1]
    return (jnp.exp(-nu * jnp.pi**2 * t) * jnp.sin(jnp.pi * x)).reshape(-1, 1)


def make_heat_points(num_t_axis=7, num_x_axis=9, num_ic=21, num_bc=15):
    t = jnp.linspace(0.0, 1.0, num_t_axis)
    x = jnp.linspace(0.0, 1.0, num_x_axis)
    T, X = jnp.meshgrid(t[1:], x[1:-1], indexing="ij")
    interior = jnp.stack([T.ravel(), X.ravel()], axis=1)

    x_ic = jnp.linspace(0.0, 1.0, num_ic)
    initial = jnp.stack([jnp.zeros_like(x_ic), x_ic], axis=1)

    t_bc = jnp.linspace(0.0, 1.0, num_bc)
    left = jnp.stack([t_bc, jnp.zeros_like(t_bc)], axis=1)
    right = jnp.stack([t_bc, jnp.ones_like(t_bc)], axis=1)
    boundary = jnp.concatenate([left, right], axis=0)
    return {"interior": interior, "initial": initial, "boundary": boundary}

def batch_values(params, points, trial_fn):
    values = jax.vmap(lambda point: trial_fn(params, point))(points)
    return values.reshape(-1, 1)


def value_mse(params, points, targets, trial_fn):
    predictions = batch_values(params, points, trial_fn)
    return jnp.mean((predictions - targets) ** 2)


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def adam_update(params, grads, state, step, learning_rate, beta1=0.9, beta2=0.999, eps=1.0e-8):
    first_moment, second_moment = state
    first_moment = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g, first_moment, grads
    )
    second_moment = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), second_moment, grads
    )
    first_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - beta1**step), first_moment)
    second_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - beta2**step), second_moment)
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + eps),
        params,
        first_hat,
        second_hat,
    )
    return params, (first_moment, second_moment)


def train_pinn(params, loss_fn, steps=600, learning_rate=1.0e-3, log_every=None):
    """Train a PINN with Adam and return trained parameters plus a compact history."""
    if log_every is None:
        log_every = max(1, steps // 6)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    value_fn = jax.jit(loss_fn)
    state = (tree_zeros_like(params), tree_zeros_like(params))
    history = []
    for step in range(1, steps + 1):
        (loss_value, components), grads = value_and_grad(params)
        params, state = adam_update(params, grads, state, step, learning_rate)
        if step == 1 or step % log_every == 0 or step == steps:
            loss_value, components = value_fn(params)
            row = {"step": step, "total": float(loss_value)}
            row.update({name: float(value) for name, value in components.items()})
            history.append(row)
    return params, history


def make_ode_soft_loss(collocation, kappa, ic_weight, residual_fn, ic_loss_fn, trial_fn=ode_soft_trial):
    def loss_fn(params):
        residual = residual_fn(params, collocation, kappa, trial_fn)
        pde_loss = jnp.mean(residual**2)
        ic_loss = ic_loss_fn(params, trial_fn, y0=0.0)
        total = pde_loss + ic_weight * ic_loss
        return total, {"pde": pde_loss, "ic": ic_loss, "bc": jnp.array(0.0)}

    return loss_fn


def make_ode_residual_only_loss(collocation, kappa, residual_fn, trial_fn):
    def loss_fn(params):
        residual = residual_fn(params, collocation, kappa, trial_fn)
        pde_loss = jnp.mean(residual**2)
        return pde_loss, {"pde": pde_loss, "ic": jnp.array(0.0), "bc": jnp.array(0.0)}

    return loss_fn


def make_poisson_soft_loss(interior, boundary, bc_weight, residual_fn, trial_fn=poisson_soft_trial):
    boundary_targets = jnp.zeros((boundary.shape[0], 1), dtype=jnp.float64)

    def loss_fn(params):
        residual = residual_fn(params, interior, trial_fn)
        pde_loss = jnp.mean(residual**2)
        bc_loss = value_mse(params, boundary, boundary_targets, trial_fn)
        total = pde_loss + bc_weight * bc_loss
        return total, {"pde": pde_loss, "ic": jnp.array(0.0), "bc": bc_loss}

    return loss_fn


def make_poisson_residual_only_loss(interior, residual_fn, trial_fn):
    def loss_fn(params):
        residual = residual_fn(params, interior, trial_fn)
        pde_loss = jnp.mean(residual**2)
        return pde_loss, {"pde": pde_loss, "ic": jnp.array(0.0), "bc": jnp.array(0.0)}

    return loss_fn


def make_poisson_deep_ritz_loss(interior, boundary, bc_weight, energy_density_fn, trial_fn=poisson_soft_trial):
    boundary_targets = jnp.zeros((boundary.shape[0], 1), dtype=jnp.float64)

    def loss_fn(params):
        energy_density = energy_density_fn(params, interior, trial_fn)
        energy_loss = jnp.mean(energy_density)
        bc_loss = value_mse(params, boundary, boundary_targets, trial_fn)
        total = energy_loss + bc_weight * bc_loss
        return total, {"energy": energy_loss, "ic": jnp.array(0.0), "bc": bc_loss}

    return loss_fn


def make_heat_soft_loss(points, nu, ic_weight, bc_weight, residual_fn, trial_fn=heat_soft_trial):
    initial_targets = exact_heat(points["initial"], nu)
    boundary_targets = jnp.zeros((points["boundary"].shape[0], 1), dtype=jnp.float64)

    def loss_fn(params):
        residual = residual_fn(params, points["interior"], nu, trial_fn)
        pde_loss = jnp.mean(residual**2)
        ic_loss = value_mse(params, points["initial"], initial_targets, trial_fn)
        bc_loss = value_mse(params, points["boundary"], boundary_targets, trial_fn)
        total = pde_loss + ic_weight * ic_loss + bc_weight * bc_loss
        return total, {"pde": pde_loss, "ic": ic_loss, "bc": bc_loss}

    return loss_fn


def make_heat_residual_only_loss(points, nu, residual_fn, trial_fn):
    def loss_fn(params):
        residual = residual_fn(params, points["interior"], nu, trial_fn)
        pde_loss = jnp.mean(residual**2)
        return pde_loss, {"pde": pde_loss, "ic": jnp.array(0.0), "bc": jnp.array(0.0)}

    return loss_fn

def evaluate_ode_model(params, trial_fn, residual_fn, kappa, num_eval=201):
    points = make_ode_collocation(num_eval)
    predictions = batch_values(params, points, trial_fn)
    exact = exact_ode(points, kappa)
    residual = residual_fn(params, points, kappa, trial_fn)
    ic_error = float(jnp.abs(trial_fn(params, jnp.array([0.0]))))
    return {
        "validation_l2": float(jnp.sqrt(jnp.mean((predictions - exact) ** 2))),
        "validation_residual": float(jnp.sqrt(jnp.mean(residual**2))),
        "condition_error": ic_error,
    }


def evaluate_poisson_model(params, trial_fn, residual_fn, num_axis=31):
    grid = make_square_interior_grid(num_axis)
    boundary = make_square_boundary_points(num_axis)
    predictions = batch_values(params, grid, trial_fn)
    exact = exact_poisson_2d(grid)
    residual = residual_fn(params, grid, trial_fn)
    boundary_error = jnp.sqrt(value_mse(params, boundary, jnp.zeros((boundary.shape[0], 1)), trial_fn))
    return {
        "validation_l2": float(jnp.sqrt(jnp.mean((predictions - exact) ** 2))),
        "validation_residual": float(jnp.sqrt(jnp.mean(residual**2))),
        "condition_error": float(boundary_error),
    }


def evaluate_heat_model(params, trial_fn, residual_fn, nu, num_t=21, num_x=31):
    t = jnp.linspace(0.0, 1.0, num_t)
    x = jnp.linspace(0.0, 1.0, num_x)
    T, X = jnp.meshgrid(t, x, indexing="ij")
    points = jnp.stack([T.ravel(), X.ravel()], axis=1)
    predictions = batch_values(params, points, trial_fn)
    exact = exact_heat(points, nu)

    interior = points[(points[:, 0] > 0.0) & (points[:, 1] > 0.0) & (points[:, 1] < 1.0)]
    residual = residual_fn(params, interior, nu, trial_fn)
    initial = jnp.stack([jnp.zeros_like(x), x], axis=1)
    boundary = jnp.concatenate(
        [jnp.stack([t, jnp.zeros_like(t)], axis=1), jnp.stack([t, jnp.ones_like(t)], axis=1)],
        axis=0,
    )
    ic_error = jnp.sqrt(value_mse(params, initial, exact_heat(initial, nu), trial_fn))
    bc_error = jnp.sqrt(value_mse(params, boundary, jnp.zeros((boundary.shape[0], 1)), trial_fn))
    return {
        "validation_l2": float(jnp.sqrt(jnp.mean((predictions - exact) ** 2))),
        "validation_residual": float(jnp.sqrt(jnp.mean(residual**2))),
        "condition_error": float(jnp.maximum(ic_error, bc_error)),
    }


def run_ode_experiment(config, residual_fn, ic_loss_fn, trial_fn=ode_soft_trial):
    collocation = make_ode_collocation(config["num_collocation"])
    params = init_tanh_mlp(random.PRNGKey(config["seed"]), 1, config["hidden_layers"])
    loss_fn = make_ode_soft_loss(collocation, config["kappa"], config["ic_weight"], residual_fn, ic_loss_fn, trial_fn)
    trained_params, history = train_pinn(params, loss_fn, steps=config["steps"], learning_rate=config["learning_rate"])
    diagnostics = evaluate_ode_model(trained_params, trial_fn, residual_fn, config["kappa"])
    return {"name": config["name"], "config": config, "params": trained_params, "history": history, **diagnostics}


def run_ode_hard_experiment(config, residual_fn, trial_fn):
    collocation = make_ode_collocation(config["num_collocation"])
    params = init_tanh_mlp(random.PRNGKey(config["seed"]), 1, config["hidden_layers"])
    loss_fn = make_ode_residual_only_loss(collocation, config["kappa"], residual_fn, trial_fn)
    trained_params, history = train_pinn(params, loss_fn, steps=config["steps"], learning_rate=config["learning_rate"])
    diagnostics = evaluate_ode_model(trained_params, trial_fn, residual_fn, config["kappa"])
    return {"name": config["name"], "config": config, "params": trained_params, "history": history, **diagnostics}


def run_poisson_soft_experiment(config, residual_fn, trial_fn=poisson_soft_trial):
    interior = make_square_interior_grid(config["num_interior_axis"])
    boundary = make_square_boundary_points(config["num_boundary_edge"])
    params = init_tanh_mlp(random.PRNGKey(config["seed"]), 2, config["hidden_layers"])
    loss_fn = make_poisson_soft_loss(interior, boundary, config["bc_weight"], residual_fn, trial_fn)
    trained_params, history = train_pinn(params, loss_fn, steps=config["steps"], learning_rate=config["learning_rate"])
    diagnostics = evaluate_poisson_model(trained_params, trial_fn, residual_fn)
    return {"name": config["name"], "config": config, "params": trained_params, "history": history, **diagnostics}


def run_poisson_deep_ritz_experiment(config, energy_density_fn, residual_fn, trial_fn=poisson_soft_trial):
    interior = make_square_interior_grid(config["num_interior_axis"])
    boundary = make_square_boundary_points(config["num_boundary_edge"])
    params = init_tanh_mlp(random.PRNGKey(config["seed"]), 2, config["hidden_layers"])
    loss_fn = make_poisson_deep_ritz_loss(interior, boundary, config["bc_weight"], energy_density_fn, trial_fn)
    trained_params, history = train_pinn(params, loss_fn, steps=config["steps"], learning_rate=config["learning_rate"])
    diagnostics = evaluate_poisson_model(trained_params, trial_fn, residual_fn)
    return {"name": config["name"], "config": config, "params": trained_params, "history": history, **diagnostics}


def run_poisson_hard_experiment(config, residual_fn, trial_fn):
    interior = make_square_interior_grid(config["num_interior_axis"])
    params = init_tanh_mlp(random.PRNGKey(config["seed"]), 2, config["hidden_layers"])
    loss_fn = make_poisson_residual_only_loss(interior, residual_fn, trial_fn)
    trained_params, history = train_pinn(params, loss_fn, steps=config["steps"], learning_rate=config["learning_rate"])
    diagnostics = evaluate_poisson_model(trained_params, trial_fn, residual_fn)
    return {"name": config["name"], "config": config, "params": trained_params, "history": history, **diagnostics}


def run_heat_soft_experiment(config, residual_fn, trial_fn=heat_soft_trial):
    points = make_heat_points(config["num_t_axis"], config["num_x_axis"], config["num_ic"], config["num_bc"])
    params = init_tanh_mlp(random.PRNGKey(config["seed"]), 2, config["hidden_layers"])
    loss_fn = make_heat_soft_loss(points, config["nu"], config["ic_weight"], config["bc_weight"], residual_fn, trial_fn)
    trained_params, history = train_pinn(params, loss_fn, steps=config["steps"], learning_rate=config["learning_rate"])
    diagnostics = evaluate_heat_model(trained_params, trial_fn, residual_fn, config["nu"])
    return {"name": config["name"], "config": config, "params": trained_params, "history": history, **diagnostics}


def run_heat_hard_experiment(config, residual_fn, trial_fn):
    points = make_heat_points(config["num_t_axis"], config["num_x_axis"], config["num_ic"], config["num_bc"])
    params = init_tanh_mlp(random.PRNGKey(config["seed"]), 2, config["hidden_layers"])
    loss_fn = make_heat_residual_only_loss(points, config["nu"], residual_fn, trial_fn)
    trained_params, history = train_pinn(params, loss_fn, steps=config["steps"], learning_rate=config["learning_rate"])
    diagnostics = evaluate_heat_model(trained_params, trial_fn, residual_fn, config["nu"])
    return {"name": config["name"], "config": config, "params": trained_params, "history": history, **diagnostics}

def print_result_table(results, title="Results"):
    print(title)
    header = f"{'name':<28} {'val L2':>10} {'val residual':>14} {'condition':>12} {'last total':>12}"
    print(header)
    print("-" * len(header))
    for result in results:
        last_total = result["history"][-1]["total"]
        print(
            f"{result['name']:<28} "
            f"{result['validation_l2']:>10.3e} "
            f"{result['validation_residual']:>14.3e} "
            f"{result['condition_error']:>12.3e} "
            f"{last_total:>12.3e}"
        )


def plot_loss_histories(results, title="Loss histories"):
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    all_totals = [row["total"] for result in results for row in result["history"]]
    use_log_scale = all(total > 0.0 for total in all_totals)
    for result in results:
        steps = [row["step"] for row in result["history"]]
        totals = [row["total"] for row in result["history"]]
        plotter = ax.semilogy if use_log_scale else ax.plot
        plotter(steps, totals, marker="o", label=result["name"])
    ax.set_xlabel("Adam step")
    ax.set_ylabel("total objective")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_ode_fit_and_residual(result, residual_fn, trial_fn, title=None):
    kappa = result["config"]["kappa"]
    points = make_ode_collocation(201)
    predictions = np.asarray(batch_values(result["params"], points, trial_fn)).ravel()
    exact = np.asarray(exact_ode(points, kappa)).ravel()
    residual = np.asarray(residual_fn(result["params"], points, kappa, trial_fn)).ravel()
    t = np.asarray(points).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    axes[0].plot(t, exact, color="black", linewidth=2.0, label="exact")
    axes[0].plot(t, predictions, color="tab:blue", linestyle="--", label="PINN")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y(t)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, residual, color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("residual")
    axes[1].grid(True, alpha=0.3)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_poisson_fields(result, residual_fn, trial_fn, title=None, num_axis=35):
    x = jnp.linspace(0.0, 1.0, num_axis)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    prediction = np.asarray(batch_values(result["params"], points, trial_fn)).reshape(num_axis, num_axis)
    exact = np.asarray(exact_poisson_2d(points)).reshape(num_axis, num_axis)
    error = prediction - exact
    residual = np.asarray(residual_fn(result["params"], points, trial_fn)).reshape(num_axis, num_axis)

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2), constrained_layout=True)
    fields = [(prediction, "PINN"), (exact, "exact"), (error, "error"), (residual, "residual")]
    for ax, (field, label) in zip(axes, fields):
        image = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), aspect="equal")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.78)
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_heat_slices(result, residual_fn, trial_fn, title=None, nu=None):
    if nu is None:
        nu = result["config"].get("nu", 0.1)
    x = jnp.linspace(0.0, 1.0, 101)
    times = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for t_value in times:
        points = jnp.stack([jnp.full_like(x, t_value), x], axis=1)
        prediction = np.asarray(batch_values(result["params"], points, trial_fn)).ravel()
        exact = np.asarray(exact_heat(points, nu)).ravel()
        axes[0].plot(np.asarray(x), exact, color="black", alpha=0.45)
        axes[0].plot(np.asarray(x), prediction, linestyle="--", label=f"t={t_value:.1f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(t,x)")
    axes[0].set_title("time slices")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    t_grid = jnp.linspace(0.0, 1.0, 31)
    x_grid = jnp.linspace(0.0, 1.0, 41)
    T, X = jnp.meshgrid(t_grid[1:], x_grid[1:-1], indexing="ij")
    residual_points = jnp.stack([T.ravel(), X.ravel()], axis=1)
    residual = np.asarray(residual_fn(result["params"], residual_points, nu, trial_fn)).reshape(30, 39)
    image = axes[1].imshow(residual.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    axes[1].set_title("validation residual")
    fig.colorbar(image, ax=axes[1], shrink=0.85)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
