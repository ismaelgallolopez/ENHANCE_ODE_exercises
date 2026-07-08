"""Provided helper infrastructure for the Day8 backpropagation lab.

Students may inspect this file, but the core exercises do not require reading
or editing it. The notebook keeps the actual student implementation tasks in
exercise-local code cells.
"""

import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

SHALLOW_ORDER = ("W1", "b1", "W2", "b2")
DEEP_ORDER = ("W1", "b1", "W2", "b2", "W3", "b3")
SHALLOW_CACHE_KEYS = ("x", "z1", "h1", "y_hat")
DEEP_CACHE_KEYS = ("x", "z1", "h1", "z2", "h2", "y_hat")


# Data and fixed parameters used throughout the notebook.
# The arrays are intentionally tiny so that shapes and cached values are easy to inspect.
def make_tiny_batch():
    """Return a deterministic one-dimensional regression batch.

    Shapes:
    - x: (n_samples, 1)
    - y: (n_samples, 1)
    """
    x = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y = jnp.sin(jnp.pi * x) + 0.25 * x
    return x, y


def init_shallow_network():
    """Fixed one-hidden-layer ReLU network with small inspectable shapes."""
    return {
        "W1": jnp.array([[1.2, -0.7, 0.5]], dtype=jnp.float64),
        "b1": jnp.array([[0.10, -0.20, 0.05]], dtype=jnp.float64),
        "W2": jnp.array([[0.7], [-0.5], [0.4]], dtype=jnp.float64),
        "b2": jnp.array([[0.05]], dtype=jnp.float64),
    }


def init_deep_network():
    """Fixed two-hidden-layer ReLU network with small inspectable shapes."""
    return {
        "W1": jnp.array([[1.2, -0.7, 0.5]], dtype=jnp.float64),
        "b1": jnp.array([[0.10, -0.20, 0.05]], dtype=jnp.float64),
        "W2": jnp.array(
            [[0.6, -1.0], [0.4, 0.3], [-0.8, 0.9]], dtype=jnp.float64
        ),
        "b2": jnp.array([[0.00, 0.15]], dtype=jnp.float64),
        "W3": jnp.array([[0.7], [-0.5]], dtype=jnp.float64),
        "b3": jnp.array([[0.05]], dtype=jnp.float64),
    }


# JAX reference models. These are not the student implementation; they are used
# as trusted comparisons for the manual gradients in Exercise 2.

def reference_relu(z):
    # This convention has derivative 0 at the ReLU kink when used with JAX.
    return jnp.where(z > 0.0, z, 0.0)


def shallow_reference_forward(params, x):
    h1 = reference_relu(x @ params["W1"] + params["b1"])
    return h1 @ params["W2"] + params["b2"]


def deep_reference_forward(params, x):
    h1 = reference_relu(x @ params["W1"] + params["b1"])
    h2 = reference_relu(h1 @ params["W2"] + params["b2"])
    return h2 @ params["W3"] + params["b3"]


def loss_from_predictions(y_hat, y):
    return jnp.mean((y_hat - y) ** 2)


def shallow_reference_loss(params, x, y):
    return loss_from_predictions(shallow_reference_forward(params, x), y)


def deep_reference_loss(params, x, y):
    return loss_from_predictions(deep_reference_forward(params, x), y)


def shallow_jax_gradients(params, x, y):
    return jax.grad(shallow_reference_loss)(params, x, y)


def deep_jax_gradients(params, x, y):
    return jax.grad(deep_reference_loss)(params, x, y)

# Shape checks and gradient diagnostics. These helpers keep the exercise cells
# short while still giving immediate feedback when a cached value or gradient
# has the wrong shape.
def count_parameters(params, order):
    return int(sum(np.prod(params[name].shape) for name in order))


def print_parameter_shapes(title, params, order):
    print(title)
    print("-" * len(title))
    for name in order:
        print(f"{name:>2}: {str(params[name].shape):>10}  values={params[name].size:2d}")
    print(f"Total parameters: {count_parameters(params, order)}\n")


def check_activation_and_loss_rules(relu_fn, relu_prime_fn, mse_rule):
    probe = jnp.array([[-2.0, 0.0, 3.0]])
    np.testing.assert_allclose(np.asarray(relu_fn(probe)), np.array([[0.0, 0.0, 3.0]]))
    np.testing.assert_allclose(np.asarray(relu_prime_fn(probe)), np.array([[0.0, 0.0, 1.0]]))

    y_hat = jnp.array([[1.0], [3.0]], dtype=jnp.float64)
    y = jnp.array([[0.0], [1.0]], dtype=jnp.float64)
    loss, dy_hat = mse_rule(y_hat, y)
    np.testing.assert_allclose(float(loss), 2.5)
    np.testing.assert_allclose(np.asarray(dy_hat), np.array([[1.0], [2.0]]))


def check_cache(cache, expected_shapes, required_keys):
    missing = set(required_keys).difference(cache)
    assert not missing, f"Missing cache entries: {sorted(missing)}"
    for name, shape in expected_shapes.items():
        assert cache[name].shape == shape, f"{name} has shape {cache[name].shape}, expected {shape}"


def check_shallow_forward_cache(params, x, y_hat, cache):
    expected_shapes = {
        "x": x.shape,
        "z1": (x.shape[0], params["W1"].shape[1]),
        "h1": (x.shape[0], params["W1"].shape[1]),
        "y_hat": (x.shape[0], params["W2"].shape[1]),
    }
    check_cache(cache, expected_shapes, SHALLOW_CACHE_KEYS)
    assert y_hat.shape == expected_shapes["y_hat"]
    np.testing.assert_allclose(np.asarray(y_hat), np.asarray(shallow_reference_forward(params, x)), atol=1e-12)


def check_deep_forward_cache(params, x, y_hat, cache):
    expected_shapes = {
        "x": x.shape,
        "z1": (x.shape[0], params["W1"].shape[1]),
        "h1": (x.shape[0], params["W1"].shape[1]),
        "z2": (x.shape[0], params["W2"].shape[1]),
        "h2": (x.shape[0], params["W2"].shape[1]),
        "y_hat": (x.shape[0], params["W3"].shape[1]),
    }
    check_cache(cache, expected_shapes, DEEP_CACHE_KEYS)
    assert y_hat.shape == expected_shapes["y_hat"]
    np.testing.assert_allclose(np.asarray(y_hat), np.asarray(deep_reference_forward(params, x)), atol=1e-12)


def print_cache_shapes(title, keys, cache):
    print(title)
    print("-" * len(title))
    for name in keys:
        print(f"{name:>5}: {str(cache[name].shape):>10}")
    print()


def print_gradient_shapes(title, grads, order):
    print(title)
    print("-" * len(title))
    for name in order:
        print(f"d{name:>2}: {str(grads[name].shape):>10}")
    print()


def assert_matching_tree_shapes(candidate, reference, order):
    for name in order:
        assert name in candidate, f"Missing gradient entry {name}"
        assert candidate[name].shape == reference[name].shape, (
            f"Shape mismatch for {name}: {candidate[name].shape} != {reference[name].shape}"
        )


def tree_max_abs_error(candidate, reference, order):
    return max(float(jnp.max(jnp.abs(candidate[name] - reference[name]))) for name in order)


def tree_l2_error(candidate, reference, order):
    squared = [jnp.sum((candidate[name] - reference[name]) ** 2) for name in order]
    return float(jnp.sqrt(sum(squared)))


def print_gradient_comparison(title, manual_grads, reference_grads, order):
    print(title)
    print("-" * len(title))
    print(f"{'name':>3}  {'shape':>10}  {'max abs error':>14}  {'relative error':>14}")
    for name in order:
        diff = manual_grads[name] - reference_grads[name]
        max_abs = float(jnp.max(jnp.abs(diff)))
        scale = max(float(jnp.max(jnp.abs(reference_grads[name]))), 1e-12)
        print(f"{name:>3}  {str(manual_grads[name].shape):>10}  {max_abs:14.3e}  {max_abs / scale:14.3e}")
    print(f"Tree max abs error: {tree_max_abs_error(manual_grads, reference_grads, order):.3e}")
    print(f"Tree L2 error:      {tree_l2_error(manual_grads, reference_grads, order):.3e}\n")


def assert_gradient_agreement(manual_grads, reference_grads, order, atol=1e-10):
    assert_matching_tree_shapes(manual_grads, reference_grads, order)
    for name in order:
        np.testing.assert_allclose(
            np.asarray(manual_grads[name]), np.asarray(reference_grads[name]), atol=atol, rtol=atol
        )


def apply_gradient_step(params, grads, order, learning_rate):
    return {name: params[name] - learning_rate * grads[name] for name in order}


def print_loss_step(title, loss_fn, params, grads, order, x, y, learning_rate=0.05):
    base_loss = float(loss_fn(params, x, y))
    trial_params = apply_gradient_step(params, grads, order, learning_rate)
    trial_loss = float(loss_fn(trial_params, x, y))
    print(title)
    print("-" * len(title))
    print(f"Loss before one manual-gradient step: {base_loss:.8f}")
    print(f"Loss after one manual-gradient step:  {trial_loss:.8f}\n")
    assert trial_loss < base_loss


# Manual-gradient-descent and plotting diagnostics for Exercise 1.

# The optimizer is provided; the teaching focus is the gradient supplied to it.
def run_manual_gradient_descent(params, forward_with_cache, backward_fn, order, x, y, learning_rate, steps):
    """Optimize the batch MSE using gradients from a manual backprop function."""
    current_params = {name: params[name] for name in order}
    loss_history = []
    for step in range(steps + 1):
        y_hat, cache = forward_with_cache(current_params, x)
        loss_history.append(float(loss_from_predictions(y_hat, y)))
        if step == steps:
            break
        grads = backward_fn(current_params, cache, y)
        current_params = apply_gradient_step(current_params, grads, order, learning_rate)
    return current_params, np.asarray(loss_history)


def print_training_summary(title, loss_history):
    print(title)
    print("-" * len(title))
    print(f"Initial MSE: {loss_history[0]:.8f}")
    print(f"Final MSE:   {loss_history[-1]:.8f}")
    print(f"Reduction:   {loss_history[0] - loss_history[-1]:.8f}\n")


def plot_loss_histories(histories):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for label, history in histories.items():
        ax.plot(np.arange(len(history)), history, label=label)
    ax.set_xlabel("gradient-descent step")
    ax.set_ylabel("batch MSE")
    ax.set_title("Manual-gradient descent on the same MSE loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()


def plot_batch_and_predictions(x, y, shallow_y_hat=None, deep_y_hat=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(np.asarray(x).ravel(), np.asarray(y).ravel(), "o", label="target batch")
    if shallow_y_hat is not None:
        ax.plot(np.asarray(x).ravel(), np.asarray(shallow_y_hat).ravel(), "s--", label="shallow output")
    if deep_y_hat is not None:
        ax.plot(np.asarray(x).ravel(), np.asarray(deep_y_hat).ravel(), "^--", label="deep output")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Tiny batch for manual backpropagation checks")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def run_manual_backpropagation_diagnostics(
    shallow_params,
    deep_params,
    x,
    y,
    shallow_forward_with_cache,
    shallow_manual_backward,
    deep_forward_with_cache,
    deep_manual_backward,
    activation_fn,
    activation_prime_fn,
    mse_output_adjoint_fn,
    shallow_learning_rate=0.05,
    deep_learning_rate=0.03,
    steps=120,
):
    """Run the Exercise 1 checks, short training runs, and plots."""
    check_activation_and_loss_rules(activation_fn, activation_prime_fn, mse_output_adjoint_fn)

    shallow_y_hat, shallow_cache = shallow_forward_with_cache(shallow_params, x)
    check_shallow_forward_cache(shallow_params, x, shallow_y_hat, shallow_cache)
    shallow_manual_grads = shallow_manual_backward(shallow_params, shallow_cache, y)
    assert_matching_tree_shapes(shallow_manual_grads, shallow_params, SHALLOW_ORDER)

    print_cache_shapes("Part a: shallow cache shapes", SHALLOW_CACHE_KEYS, shallow_cache)
    print_gradient_shapes("Part a: shallow manual gradient shapes", shallow_manual_grads, SHALLOW_ORDER)

    deep_y_hat, deep_cache = deep_forward_with_cache(deep_params, x)
    check_deep_forward_cache(deep_params, x, deep_y_hat, deep_cache)
    deep_manual_grads = deep_manual_backward(deep_params, deep_cache, y)
    assert_matching_tree_shapes(deep_manual_grads, deep_params, DEEP_ORDER)

    print_cache_shapes("Part b: deep cache shapes", DEEP_CACHE_KEYS, deep_cache)
    print_gradient_shapes("Part b: deep manual gradient shapes", deep_manual_grads, DEEP_ORDER)
    plot_batch_and_predictions(x, y, shallow_y_hat, deep_y_hat)

    shallow_trained_params, shallow_loss_history = run_manual_gradient_descent(
        shallow_params,
        shallow_forward_with_cache,
        shallow_manual_backward,
        SHALLOW_ORDER,
        x,
        y,
        learning_rate=shallow_learning_rate,
        steps=steps,
    )
    deep_trained_params, deep_loss_history = run_manual_gradient_descent(
        deep_params,
        deep_forward_with_cache,
        deep_manual_backward,
        DEEP_ORDER,
        x,
        y,
        learning_rate=deep_learning_rate,
        steps=steps,
    )

    print_training_summary("Part a: shallow manual-gradient descent", shallow_loss_history)
    print_training_summary("Part b: deep manual-gradient descent", deep_loss_history)
    assert shallow_loss_history[-1] < shallow_loss_history[0]
    assert deep_loss_history[-1] < deep_loss_history[0]
    plot_loss_histories({"shallow": shallow_loss_history, "deep": deep_loss_history})

    shallow_trained_y_hat = shallow_reference_forward(shallow_trained_params, x)
    deep_trained_y_hat = deep_reference_forward(deep_trained_params, x)
    plot_batch_and_predictions(x, y, shallow_trained_y_hat, deep_trained_y_hat)

    return {
        "shallow_cache": shallow_cache,
        "deep_cache": deep_cache,
        "shallow_grads": shallow_manual_grads,
        "deep_grads": deep_manual_grads,
        "shallow_trained_params": shallow_trained_params,
        "deep_trained_params": deep_trained_params,
        "shallow_loss_history": shallow_loss_history,
        "deep_loss_history": deep_loss_history,
    }


# Forward-mode versus reverse-mode efficiency helpers for Exercise 3.

# These estimate and time how the cost scales with the number of parameters.
def count_dense_parameters(layer_widths):
    return int(
        sum((layer_widths[i] + 1) * layer_widths[i + 1] for i in range(len(layer_widths) - 1))
    )


def estimate_cached_scalars(layer_widths, batch_size):
    hidden_widths = layer_widths[1:-1]
    return int(batch_size * (layer_widths[0] + 2 * sum(hidden_widths) + layer_widths[-1]))


def check_efficiency_table(rows, widths, batch_size=6):
    required = {
        "architecture", "width", "parameters", "cached_scalars",
        "forward_mode_sweeps", "reverse_mode_sweeps", "sweep_ratio"
    }
    widths = list(widths)
    expected_pairs = {(architecture, width) for width in widths for architecture in ("shallow", "deep")}
    seen_pairs = set()
    assert len(rows) == 2 * len(widths), "Use one shallow and one deep row per width"
    for row in rows:
        missing = required.difference(row)
        assert not missing, f"Missing efficiency-table columns: {sorted(missing)}"
        architecture = row["architecture"]
        width = row["width"]
        pair = (architecture, width)
        assert pair in expected_pairs, f"Unexpected architecture/width pair: {pair}"
        assert pair not in seen_pairs, f"Duplicate efficiency-table row for {pair}"
        seen_pairs.add(pair)

        layer_widths = [1, width, 1] if architecture == "shallow" else [1, width, width, 1]
        expected_parameters = count_dense_parameters(layer_widths)
        expected_cached = estimate_cached_scalars(layer_widths, batch_size)
        assert row["parameters"] == expected_parameters
        assert row["cached_scalars"] == expected_cached
        assert row["reverse_mode_sweeps"] == 1
        assert row["forward_mode_sweeps"] == expected_parameters
        assert row["sweep_ratio"] == row["forward_mode_sweeps"] / row["reverse_mode_sweeps"]
    assert seen_pairs == expected_pairs


def print_efficiency_table(rows):
    print(f"{'architecture':>12}  {'width':>5}  {'parameters':>10}  {'cached':>8}  {'forward sweeps':>15}  {'reverse sweeps':>15}  {'ratio':>8}")
    print("-" * 86)
    for row in rows:
        print(
            f"{row['architecture']:>12}  {row['width']:5d}  {row['parameters']:10d}  "
            f"{row['cached_scalars']:8d}  {row['forward_mode_sweeps']:15d}  "
            f"{row['reverse_mode_sweeps']:15d}  {row['sweep_ratio']:8.1f}"
        )
    print()


def plot_efficiency_table(rows):
    fig, ax = plt.subplots(figsize=(6.7, 3.3))
    for architecture, marker in [("shallow", "o"), ("deep", "s")]:
        subset = [row for row in rows if row["architecture"] == architecture]
        widths = [row["width"] for row in subset]
        ratios = [row["sweep_ratio"] for row in subset]
        ax.plot(widths, ratios, marker + "-", label=architecture)
    ax.set_xlabel("hidden width")
    ax.set_ylabel("forward-mode sweeps / reverse-mode sweeps")
    ax.set_title("Full-gradient sweep count for one scalar loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()


def init_dense_mlp(layer_widths, seed=0):
    rng = np.random.default_rng(seed)
    params = {}
    for layer_index, (in_dim, out_dim) in enumerate(zip(layer_widths[:-1], layer_widths[1:]), start=1):
        scale = 1.0 / np.sqrt(max(in_dim, 1))
        params[f"W{layer_index}"] = jnp.array(rng.normal(size=(in_dim, out_dim)) * scale, dtype=jnp.float64)
        params[f"b{layer_index}"] = jnp.zeros((1, out_dim), dtype=jnp.float64)
    return params


def order_for_layer_count(layer_count):
    names = []
    for layer_index in range(1, layer_count + 1):
        names.extend([f"W{layer_index}", f"b{layer_index}"])
    return tuple(names)


def mlp_forward(params, x):
    layer_count = len(params) // 2
    h = x
    for layer_index in range(1, layer_count):
        z = h @ params[f"W{layer_index}"] + params[f"b{layer_index}"]
        h = reference_relu(z)
    return h @ params[f"W{layer_count}"] + params[f"b{layer_count}"]


def mlp_loss(params, x, y):
    return loss_from_predictions(mlp_forward(params, x), y)


def flatten_params(params, order):
    pieces = []
    spec = []
    for name in order:
        value = params[name]
        spec.append((name, value.shape, value.size))
        pieces.append(jnp.ravel(value))
    return jnp.concatenate(pieces), spec


def unflatten_params(flat, spec):
    params = {}
    start = 0
    for name, shape, size in spec:
        params[name] = jnp.reshape(flat[start:start + size], shape)
        start += size
    return params


def loss_from_flat(flat_params, spec, x, y):
    params = unflatten_params(flat_params, spec)
    return mlp_loss(params, x, y)


def reverse_mode_grad_flat(flat_params, spec, x, y):
    def flat_loss(values):
        return loss_from_flat(values, spec, x, y)
    return jax.grad(flat_loss)(flat_params)


def forward_mode_grad_flat(flat_params, spec, x, y):
    def flat_loss(values):
        return loss_from_flat(values, spec, x, y)
    components = []
    basis = jnp.eye(int(flat_params.size), dtype=flat_params.dtype)
    for direction in basis:
        _, directional_derivative = jax.jvp(flat_loss, (flat_params,), (direction,))
        components.append(directional_derivative)
    return jnp.stack(components)


def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def time_gradient_methods(widths, batch_size=6):
    x, y = make_tiny_batch()
    x = x[:batch_size]
    y = y[:batch_size]
    rows = []
    for width in widths:
        layer_widths = [1, width, width, 1]
        params = init_dense_mlp(layer_widths, seed=10 + width)
        order = order_for_layer_count(len(layer_widths) - 1)
        flat_params, spec = flatten_params(params, order)

        start = time.perf_counter()
        reverse_grad = reverse_mode_grad_flat(flat_params, spec, x, y)
        _block_until_ready(reverse_grad)
        reverse_ms = 1000.0 * (time.perf_counter() - start)

        start = time.perf_counter()
        forward_grad = forward_mode_grad_flat(flat_params, spec, x, y)
        _block_until_ready(forward_grad)
        forward_ms = 1000.0 * (time.perf_counter() - start)

        rows.append(
            {
                "width": width,
                "parameters": int(flat_params.size),
                "reverse_ms": reverse_ms,
                "forward_ms": forward_ms,
                "max_abs_difference": float(jnp.max(jnp.abs(reverse_grad - forward_grad))),
            }
        )
    return rows


def print_timing_table(rows):
    print("Illustrative timing: full gradient of a small deep network")
    print("---------------------------------------------------------")
    print(f"{'width':>5}  {'parameters':>10}  {'reverse ms':>12}  {'forward ms':>12}  {'max |diff|':>12}")
    for row in rows:
        print(
            f"{row['width']:5d}  {row['parameters']:10d}  {row['reverse_ms']:12.2f}  "
            f"{row['forward_ms']:12.2f}  {row['max_abs_difference']:12.3e}"
        )
    print()
