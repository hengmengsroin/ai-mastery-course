"""
PyTorch in One Hour — Interactive Playground
Applied AI Mastery Program

Turns the "PyTorch in One Hour" tutorial notebook into a hands-on Streamlit app:
tensors, autograd, network architecture, data loaders, and a live training loop
you can actually train, watch converge, and download.

Run:  streamlit run pytorch_playground_app.py
"""
import warnings; warnings.filterwarnings("ignore")
import io

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import streamlit as st
import plotly.graph_objects as go


def add_grid(fig, x0, y0, rows, cols, cell, color, opacity=1.0):
    for r in range(rows):
        for c in range(cols):
            fig.add_shape(
                type="rect",
                x0=x0 + c * cell, x1=x0 + c * cell + cell * 0.85,
                y0=y0 - r * cell, y1=y0 - r * cell - cell * 0.85,
                line=dict(color="white", width=1), fillcolor=color, opacity=opacity,
            )

st.set_page_config(page_title="PyTorch in One Hour — Playground", page_icon="🔥", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.35rem; }
div[data-testid="stExpander"] > div { padding: 4px 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 PyTorch in One Hour — Interactive Playground")
st.caption("A hands-on companion to the tutorial notebook: tensors → autograd → networks → data loaders → training → devices.")

tabs = st.tabs([
    "1. Tensors", "2. Autograd", "3. Build a Network",
    "4. Data Loaders", "5. Train it Live", "6. Save / Load",
    "7. Devices & DDP",
])

# ────────────────────────────────────────────────────────────────────────────
# 1. Tensors
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Tensors: the basic building block")

    with st.expander("📖 What is a tensor, and why PyTorch?", expanded=True):
        st.markdown(
            "**PyTorch** pairs a NumPy-like array interface with two things NumPy doesn't have "
            "built in: automatic differentiation (autograd) and native GPU acceleration. The "
            "data structure both are built on is the **tensor**.\n\n"
            "A tensor generalizes numbers → vectors → matrices → higher-dimensional arrays. Its "
            "**rank** (number of dimensions) tells you what kind of object it is:\n"
            "- rank 0 → scalar (a single number)\n"
            "- rank 1 → vector\n"
            "- rank 2 → matrix\n"
            "- rank 3+ → just called an *N-D tensor*\n\n"
            "Type a nested list below and see its rank, shape, and dtype."
        )
        fig_rank = go.Figure()
        add_grid(fig_rank, 0.2, 1, 1, 1, 1, "#4C72B0")
        fig_rank.add_annotation(x=0.6, y=1.9, text="Scalar (0D)", showarrow=False, font=dict(size=12))
        add_grid(fig_rank, 2.5, 1, 1, 4, 1, "#55A868")
        fig_rank.add_annotation(x=4.1, y=1.9, text="Vector (1D)", showarrow=False, font=dict(size=12))
        add_grid(fig_rank, 8, 1, 3, 4, 1, "#DD8452")
        fig_rank.add_annotation(x=9.6, y=1.9, text="Matrix (2D)", showarrow=False, font=dict(size=12))
        add_grid(fig_rank, 15.1, 0.4, 3, 4, 1, "#C44E52", opacity=0.85)
        add_grid(fig_rank, 14.5, 1, 3, 4, 1, "#C44E52")
        fig_rank.add_annotation(x=16.1, y=1.9, text="3D tensor", showarrow=False, font=dict(size=12))
        fig_rank.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False, range=[-0.5, 19.5]), yaxis=dict(visible=False, range=[-3, 2.5]),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rank, width='stretch')

    col1, col2 = st.columns([1, 1])
    with col1:
        expr = st.text_area(
            "Python list literal", value="[[1, 2, 3], [4, 5, 6]]", height=100,
            help="e.g. 1, [1, 2, 3], [[1, 2], [3, 4]], or a float list like [1.0, 2.0]",
        )
        try:
            data = eval(expr, {"__builtins__": {}})
            t = torch.tensor(data)
            st.success("Parsed successfully")
        except Exception as e:
            t = None
            st.error(f"Couldn't parse that as a tensor: {e}")

    with col2:
        if t is not None:
            st.metric("Rank (ndim)", t.dim())
            st.metric("Shape", str(list(t.shape)))
            st.metric("Dtype", str(t.dtype))
            st.code(str(t), language="text")

    if t is not None and t.dim() == 2 and t.numel() > 0:
        st.subheader("Reshape / transpose / matmul playground")
        r, c = t.shape
        divisors = [d for d in range(1, r * c + 1) if (r * c) % d == 0]
        c1, c2 = st.columns(2)
        with c1:
            new_rows = st.selectbox("reshape rows", divisors, index=0)
        new_cols = (r * c) // new_rows
        with c2:
            st.write(f"→ reshape({new_rows}, {new_cols})")
        st.code(
            f"tensor.reshape({new_rows}, {new_cols}):\n{t.reshape(new_rows, new_cols)}\n\n"
            f"tensor.T:\n{t.T}\n\n"
            f"tensor @ tensor.T:\n{t.float() @ t.float().T}",
            language="text",
        )

    st.subheader("Why dtype matters")
    st.markdown(
        "The element type trades **precision** against **memory and speed**. Python `int`s "
        "become `int64`; Python `float`s become `float32` — not `float64`. `float32` is the "
        "deep learning workhorse: enough precision for most tasks, half the memory of "
        "`float64`, and what GPUs are fastest at."
    )
    dtype_bytes = {"float64": 8, "float32": 4, "float16": 2, "int64": 8, "int32": 4, "bool": 1}
    fig_dtype = go.Figure(go.Bar(
        x=list(dtype_bytes.keys()), y=list(dtype_bytes.values()),
        marker_color=["#C44E52" if k == "float64" else "#4C72B0" if k == "float32" else "#8C8C8C"
                      for k in dtype_bytes],
    ))
    fig_dtype.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="bytes per element")
    st.plotly_chart(fig_dtype, width='stretch')

    if t is not None:
        target = st.selectbox("Cast to", ["float32", "float64", "int32", "int64", "bool"])
        st.code(f"tensor.to(torch.{target}) -> dtype = {t.to(getattr(torch, target)).dtype}")

# ────────────────────────────────────────────────────────────────────────────
# 2. Autograd / computation graph
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Automatic differentiation")

    with st.expander("📖 What's a computation graph, and why does autograd need it?", expanded=True):
        st.markdown(
            "Training a network means nudging its parameters to reduce a loss — and that "
            "requires the **gradient** of the loss with respect to every parameter. "
            "Backpropagation is just the calculus chain rule, applied automatically across "
            "every operation you ran.\n\n"
            "PyTorch builds a **computation graph** on the fly: every tensor operation "
            "(multiply, add, sigmoid, …) becomes a node that remembers *how* it was computed. "
            "Calling `.backward()` walks that graph from the loss back to the leaves (`w1`, "
            "`b`), applying the chain rule at each node to accumulate `.grad`. You almost never "
            "compute derivatives by hand — `requires_grad=True` + `.backward()` handles it.\n\n"
            "This mirrors the notebook's single-neuron example: "
            r"$z = x_1 w_1 + b$, $a = \sigma(z)$, loss = binary cross-entropy(a, y). "
            "Drag the sliders and watch the forward pass and gradients update live."
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x1_val = st.slider("x1 (input)", -5.0, 5.0, 1.1, 0.1)
    with c2:
        w1_val = st.slider("w1 (weight)", -5.0, 5.0, 2.2, 0.1)
    with c3:
        b_val = st.slider("b (bias)", -5.0, 5.0, 0.0, 0.1)
    with c4:
        y_val = st.selectbox("y (true label)", [0.0, 1.0], index=1)

    y = torch.tensor([y_val])
    x1 = torch.tensor([x1_val])
    w1 = torch.tensor([w1_val], requires_grad=True)
    b = torch.tensor([b_val], requires_grad=True)

    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)
    grads = grad(loss, [w1, b], retain_graph=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("z = x1·w1 + b", f"{z.item():.4f}")
    m2.metric("a = σ(z)", f"{a.item():.4f}")
    m3.metric("loss (BCE)", f"{loss.item():.4f}")
    m4.metric("∂loss/∂w1", f"{grads[0].item():.4f}")
    m5.metric("∂loss/∂b", f"{grads[1].item():.4f}")

    # Computation graph diagram, recomputed with current values
    st.subheader("Computation graph")
    nodes = {
        "x1": (0, 3, f"x1={x1_val:.2f}"),
        "w1": (0, 2, f"w1={w1_val:.2f}"),
        "mul": (1.3, 2.5, f"x1·w1={ (x1_val*w1_val):.2f}"),
        "b": (1.3, 1.0, f"b={b_val:.2f}"),
        "z": (2.6, 2.0, f"z={z.item():.2f}"),
        "a": (3.9, 2.0, f"a=σ(z)={a.item():.2f}"),
        "y": (3.9, 0.7, f"y={y_val:.0f}"),
        "loss": (5.2, 1.4, f"loss={loss.item():.2f}"),
    }
    edges = [("x1", "mul"), ("w1", "mul"), ("mul", "z"), ("b", "z"),
             ("z", "a"), ("a", "loss"), ("y", "loss")]

    fig = go.Figure()
    for src, dst in edges:
        x0, y0, _ = nodes[src]
        x1_, y1_, _ = nodes[dst]
        fig.add_annotation(
            x=x1_, y=y1_, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor="#888",
        )
    xs = [v[0] for v in nodes.values()]
    ys = [v[1] for v in nodes.values()]
    labels = [v[2] for v in nodes.values()]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
        marker=dict(size=34, color="#4C72B0", line=dict(width=2, color="white")),
        textfont=dict(size=12),
    ))
    fig.update_layout(
        height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.5, 6]), yaxis=dict(visible=False, range=[0, 4]),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("`.backward()` walks this graph right-to-left, applying the chain rule at every node.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Where 'a' sits on the sigmoid curve")
        zs = np.linspace(-8, 8, 200)
        sig = 1 / (1 + np.exp(-zs))
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=zs, y=sig, mode="lines", line=dict(color="#4C72B0", width=2), name="σ(z)"))
        fig_sig.add_trace(go.Scatter(x=[z.item()], y=[a.item()], mode="markers",
                                      marker=dict(size=14, color="#DD8452"), name="current (z, a)"))
        fig_sig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="z", yaxis_title="σ(z)", showlegend=False)
        st.plotly_chart(fig_sig, width='stretch')
        st.caption("Sigmoid squashes any real number into (0, 1) — that's what turns a raw logit into a probability.")

    with col_b:
        st.subheader(f"BCE loss shape (y={y_val:.0f})")
        a_range = np.linspace(0.001, 0.999, 200)
        bce = -np.log(a_range) if y_val == 1.0 else -np.log(1 - a_range)
        fig_bce = go.Figure()
        fig_bce.add_trace(go.Scatter(x=a_range, y=bce, mode="lines", line=dict(color="#55A868", width=2)))
        fig_bce.add_trace(go.Scatter(x=[a.item()], y=[loss.item()], mode="markers",
                                      marker=dict(size=14, color="#DD8452")))
        fig_bce.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="a (predicted probability)", yaxis_title="loss", showlegend=False)
        st.plotly_chart(fig_bce, width='stretch')
        st.caption("Loss explodes as the prediction confidently disagrees with the label — that's what drives large gradients early in training.")

# ────────────────────────────────────────────────────────────────────────────
# 3. Build a network
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Build a multilayer network")

    with st.expander("📖 Why hidden layers and activation functions?", expanded=True):
        st.markdown(
            "Stacking `nn.Linear` layers with **no** nonlinearity between them collapses "
            "mathematically to a single linear layer — depth would buy nothing. Inserting a "
            "nonlinear **activation function** (here `ReLU`) after each hidden layer is what "
            "lets the network approximate curved, non-linear decision boundaries.\n\n"
            "`__init__` defines the layers (wrapped in `nn.Sequential` so they run in order); "
            "`forward` describes how input flows through them. The network outputs raw "
            "**logits** (no final softmax) — PyTorch's classification losses (`cross_entropy`) "
            "apply softmax internally for numerical stability, so you feed logits straight in."
        )
        xs = np.linspace(-5, 5, 200)
        fig_relu = go.Figure(go.Scatter(x=xs, y=np.maximum(0, xs), line=dict(color="#4C72B0", width=2)))
        fig_relu.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), title="ReLU(x) = max(0, x)")
        st.plotly_chart(fig_relu, width='stretch')

    st.markdown("Choose hidden layer sizes and see the resulting architecture and parameter count.")

    num_inputs = st.number_input("Input features", 1, 200, 2)
    num_outputs = st.number_input("Output classes", 1, 20, 2)
    n_hidden = st.slider("Number of hidden layers", 0, 4, 2)
    hidden_sizes = []
    if n_hidden:
        cols = st.columns(n_hidden)
        for i, col in enumerate(cols):
            with col:
                hidden_sizes.append(st.number_input(f"Hidden {i+1} size", 1, 512, [30, 20, 16, 8][i] if i < 4 else 16))

    layer_sizes = [int(num_inputs)] + [int(h) for h in hidden_sizes] + [int(num_outputs)]

    class BuiltNetwork(torch.nn.Module):
        def __init__(self, sizes):
            super().__init__()
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    layers.append(torch.nn.ReLU())
            self.layers = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    torch.manual_seed(123)
    built_model = BuiltNetwork(layer_sizes)
    n_params = sum(p.numel() for p in built_model.parameters() if p.requires_grad)

    st.metric("Trainable parameters", f"{n_params:,}")

    st.subheader("Random weight initialization")
    st.markdown(
        "Weights start as small **random** numbers — this breaks symmetry so neurons learn "
        "different things instead of all computing the same update. Seeded here with "
        "`torch.manual_seed(123)` so it's reproducible."
    )
    first_layer_w = built_model.layers[0].weight.detach().flatten().numpy()
    fig_hist = go.Figure(go.Histogram(x=first_layer_w, marker_color="#55A868", nbinsx=30))
    fig_hist.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                           title="Initial weight distribution — first Linear layer", xaxis_title="weight value")
    st.plotly_chart(fig_hist, width='stretch')

    # Diagram
    MAX_SHOW = 8
    fig = go.Figure()
    x_positions = np.linspace(0, 10, len(layer_sizes))
    layer_names = ["Input"] + [f"Hidden {i+1}" for i in range(len(hidden_sizes))] + ["Output"]
    node_xy = []
    for li, (size, xpos, name) in enumerate(zip(layer_sizes, x_positions, layer_names)):
        shown = min(size, MAX_SHOW)
        ys = np.linspace(0, 1, shown) if shown > 1 else np.array([0.5])
        ys = ys * 6
        col_nodes = [(xpos, y) for y in ys]
        node_xy.append(col_nodes)
        fig.add_trace(go.Scatter(
            x=[xpos] * shown, y=ys, mode="markers", marker=dict(size=18, color="#4C72B0"),
            showlegend=False, hovertext=[f"{name} neuron"] * shown,
        ))
        fig.add_annotation(x=xpos, y=6.8, text=f"{name}<br>({size})", showarrow=False, font=dict(size=12))
        if size > MAX_SHOW:
            fig.add_annotation(x=xpos, y=-0.6, text="…", showarrow=False, font=dict(size=16))
    for li in range(len(node_xy) - 1):
        for (x0, y0) in node_xy[li]:
            for (x1_, y1_) in node_xy[li + 1]:
                fig.add_shape(type="line", x0=x0, y0=y0, x1=x1_, y1=y1_,
                              line=dict(color="rgba(150,150,150,0.25)", width=1))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Show the generated nn.Module code"):
        code_lines = ["class NeuralNetwork(torch.nn.Module):",
                      "    def __init__(self):", "        super().__init__()",
                      "        self.layers = torch.nn.Sequential("]
        for i in range(len(layer_sizes) - 1):
            code_lines.append(f"            torch.nn.Linear({layer_sizes[i]}, {layer_sizes[i+1]}),")
            if i < len(layer_sizes) - 2:
                code_lines.append("            torch.nn.ReLU(),")
        code_lines.append("        )")
        code_lines += ["", "    def forward(self, x):", "        return self.layers(x)"]
        st.code("\n".join(code_lines), language="python")

    st.session_state["arch_layer_sizes"] = layer_sizes

# ────────────────────────────────────────────────────────────────────────────
# 4. Data loaders
# ────────────────────────────────────────────────────────────────────────────
TOY_X_TRAIN = torch.tensor([
    [-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [-1.0, 2.8], [-0.7, 3.0],
    [2.3, -1.1], [2.7, -1.5], [2.0, -1.0], [2.5, -1.3], [2.9, -1.7],
])
TOY_Y_TRAIN = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
TOY_X_TEST = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
TOY_Y_TEST = torch.tensor([0, 1])


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


with tabs[3]:
    st.header("DataLoader batching")

    with st.expander("📖 Dataset vs DataLoader, and what's a 'step'?", expanded=True):
        st.markdown(
            "Training iterates over data in **batches**, split into two pieces:\n"
            "- **`Dataset`** — knows how to fetch a single example (`__getitem__`) and how many "
            "there are (`__len__`).\n"
            "- **`DataLoader`** — wraps a `Dataset` to handle batching, shuffling, and "
            "(optionally) parallel loading via `num_workers`.\n\n"
            "One full pass over the dataset is an **epoch**; each `optimizer.step()` on one "
            "batch is a **step**. With N examples and batch size B, an epoch has roughly N/B "
            "steps — fewer if `drop_last=True` discards a short final batch, which avoids "
            "letting a tiny, noisy batch destabilize training."
        )

    st.markdown("Same 10-point toy dataset as the notebook (5 per class). Tune the loader and see the actual batches produced.")

    c1, c2, c3 = st.columns(3)
    with c1:
        batch_size = st.slider("batch_size", 1, 10, 2)
    with c2:
        shuffle = st.checkbox("shuffle", value=True)
    with c3:
        drop_last = st.checkbox("drop_last", value=True)

    seed = st.number_input("torch.manual_seed", 0, 9999, 123)
    torch.manual_seed(int(seed))
    train_ds = ToyDataset(TOY_X_TRAIN, TOY_Y_TRAIN)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)

    batches = list(loader)
    for i, (feats, labels) in enumerate(batches):
        st.write(f"**Batch {i+1}** — features shape `{tuple(feats.shape)}`, labels `{labels.tolist()}`")
        st.dataframe(
            {"x0": feats[:, 0].tolist(), "x1": feats[:, 1].tolist(), "label": labels.tolist()},
            width='stretch', hide_index=True,
        )
    n_batches = len(batches)
    st.info(f"{n_batches} batch(es) per epoch with these settings — "
            f"{'the last partial batch is dropped' if drop_last else 'the last batch may be smaller'}.")

    st.subheader("This epoch, visually")
    strip_x, strip_batch, strip_label = [], [], []
    for bi, (feats, labels) in enumerate(batches):
        for lbl in labels.tolist():
            strip_x.append(len(strip_x))
            strip_batch.append(bi)
            strip_label.append(f"batch {bi + 1}, label {lbl}")
    fig_strip = go.Figure(go.Bar(
        x=strip_x, y=[1] * len(strip_x), marker=dict(color=strip_batch, colorscale="Viridis"),
        hovertext=strip_label, hoverinfo="text",
    ))
    fig_strip.update_layout(
        height=140, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        xaxis=dict(title="example index in epoch"), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig_strip, width='stretch')
    n_used = len(strip_x)
    if n_used < len(train_ds):
        st.caption(f"{len(train_ds) - n_used} example(s) dropped this epoch — `drop_last=True` and they didn't fill a full batch. "
                   f"Each color band above is one batch/step.")
    else:
        st.caption("Each color band above is one batch — i.e. one `optimizer.step()` if this were a real training epoch.")

# ────────────────────────────────────────────────────────────────────────────
# 5. Train it live
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Train a real model, live")

    with st.expander("📖 The training loop, step by step", expanded=True):
        st.markdown(
            "Every mini-batch goes through the same five steps:\n\n"
            "1. `logits = model(features)` — **forward pass**\n"
            "2. `loss = F.cross_entropy(logits, labels)` — compute the loss (cross-entropy "
            "applies softmax internally)\n"
            "3. `optimizer.zero_grad()` — **reset** gradients from the previous step (PyTorch "
            "accumulates `.grad` by default, so this is easy to forget)\n"
            "4. `loss.backward()` — **backward pass**, populates `.grad` on every parameter\n"
            "5. `optimizer.step()` — nudge parameters using their gradients and the "
            "optimizer's update rule\n\n"
            "One pass through all mini-batches is an **epoch**. Repeat for several epochs "
            "until the loss stops improving."
        )

    st.subheader("Bonus: how do SGD and Adam differ?")
    st.markdown(
        "Same idea, zoomed out: an optimizer repeatedly nudges parameters downhill on the "
        "loss surface. Below, a synthetic elongated bowl `L(p1, p2) = 0.1·p1² + 2·p2²` shows "
        "why plain SGD can zig-zag on ill-conditioned losses while Adam — which adapts a "
        "per-parameter step size — tends to move more directly toward the minimum."
    )

    def _run_optimizer_demo(opt_name, steps=25, lr_demo=0.3):
        p = torch.tensor([4.0, 2.0], requires_grad=True)
        opt_demo = (torch.optim.SGD([p], lr=lr_demo) if opt_name == "SGD"
                    else torch.optim.Adam([p], lr=lr_demo))
        path = [p.detach().numpy().copy()]
        for _ in range(steps):
            opt_demo.zero_grad()
            loss_demo = 0.1 * p[0] ** 2 + 2 * p[1] ** 2
            loss_demo.backward()
            opt_demo.step()
            path.append(p.detach().numpy().copy())
        return np.array(path)

    path_sgd = _run_optimizer_demo("SGD")
    path_adam = _run_optimizer_demo("Adam")
    gx, gy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-3, 3, 100))
    gz = 0.1 * gx ** 2 + 2 * gy ** 2
    fig_opt = go.Figure()
    fig_opt.add_trace(go.Contour(x=gx[0], y=gy[:, 0], z=gz, showscale=False, opacity=0.5,
                                  colorscale="Blues", contours=dict(coloring="lines")))
    fig_opt.add_trace(go.Scatter(x=path_sgd[:, 0], y=path_sgd[:, 1], mode="lines+markers",
                                  name="SGD", line=dict(color="#DD8452", width=2)))
    fig_opt.add_trace(go.Scatter(x=path_adam[:, 0], y=path_adam[:, 1], mode="lines+markers",
                                  name="Adam", line=dict(color="#4C72B0", width=2)))
    fig_opt.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="p1", yaxis_title="p2")
    st.plotly_chart(fig_opt, width='stretch')
    st.caption("Both start at (4, 2) with the same learning rate (0.3), fixed here for a clean comparison — independent of the hyperparameters below.")

    st.divider()
    st.markdown("Now trains `NeuralNetwork(2, 2)` on the toy 2-class dataset from the notebook. Adjust hyperparameters and hit **Train**.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lr = st.number_input("Learning rate", 0.001, 5.0, 0.5, 0.01)
    with c2:
        num_epochs = st.slider("Epochs", 1, 100, 20)
    with c3:
        optimizer_name = st.selectbox("Optimizer", ["SGD", "Adam"])
    with c4:
        train_seed = st.number_input("Seed", 0, 9999, 123, key="train_seed")

    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device_options.append("mps")
    device_choice = st.selectbox("Device", device_options)

    class NeuralNetwork(torch.nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(num_inputs, 30), torch.nn.ReLU(),
                torch.nn.Linear(30, 20), torch.nn.ReLU(),
                torch.nn.Linear(20, num_outputs),
            )

        def forward(self, x):
            return self.layers(x)

    def compute_accuracy(model, dataloader, device):
        model.eval()
        correct, total = 0.0, 0
        for feats, labels in dataloader:
            feats, labels = feats.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(feats)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(labels == preds)
            total += len(labels)
        return (correct / total).item()

    if st.button("Train", type="primary"):
        device = torch.device(device_choice)
        torch.manual_seed(int(train_seed))
        model = NeuralNetwork(2, 2).to(device)
        optimizer = (torch.optim.SGD(model.parameters(), lr=lr) if optimizer_name == "SGD"
                     else torch.optim.Adam(model.parameters(), lr=lr))

        train_ds = ToyDataset(TOY_X_TRAIN, TOY_Y_TRAIN)
        test_ds = ToyDataset(TOY_X_TEST, TOY_Y_TEST)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

        loss_history = []
        progress = st.progress(0.0, text="Training…")
        for epoch in range(num_epochs):
            model.train()
            for feats, labels in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
            progress.progress((epoch + 1) / num_epochs, text=f"Epoch {epoch+1}/{num_epochs} — loss {loss.item():.4f}")
        progress.empty()

        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)

        m1, m2, m3 = st.columns(3)
        m1.metric("Final batch loss", f"{loss_history[-1]:.4f}")
        m2.metric("Train accuracy", f"{train_acc:.0%}")
        m3.metric("Test accuracy", f"{test_acc:.0%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss_history, mode="lines+markers", name="batch loss",
                                  line=dict(color="#4C72B0", width=2)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="update step", yaxis_title="loss", title="Loss curve")
        st.plotly_chart(fig, width='stretch')

        x_min, x_max = TOY_X_TRAIN[:, 0].min().item() - 1, TOY_X_TRAIN[:, 0].max().item() + 1
        y_min, y_max = TOY_X_TRAIN[:, 1].min().item() - 1, TOY_X_TRAIN[:, 1].max().item() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            zz = torch.argmax(model(grid), dim=1).cpu().numpy().reshape(xx.shape)

        fig2 = go.Figure()
        fig2.add_trace(go.Contour(x=np.linspace(x_min, x_max, 150), y=np.linspace(y_min, y_max, 150),
                                   z=zz, showscale=False, opacity=0.35,
                                   colorscale=[[0, "#4C72B0"], [1, "#DD8452"]]))
        fig2.add_trace(go.Scatter(x=TOY_X_TRAIN[:, 0], y=TOY_X_TRAIN[:, 1], mode="markers",
                                   marker=dict(color=TOY_Y_TRAIN, colorscale=[[0, "#4C72B0"], [1, "#DD8452"]],
                                               size=12, line=dict(width=1, color="white")),
                                   name="train"))
        fig2.add_trace(go.Scatter(x=TOY_X_TEST[:, 0], y=TOY_X_TEST[:, 1], mode="markers",
                                   marker=dict(symbol="star", color=TOY_Y_TEST,
                                               colorscale=[[0, "#4C72B0"], [1, "#DD8452"]],
                                               size=16, line=dict(width=1, color="black")),
                                   name="test"))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title="Decision boundary")
        st.plotly_chart(fig2, width='stretch')

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        st.session_state["trained_state_dict"] = buf.getvalue()
        st.success("Model trained. Head to the **Save / Load** tab to download it.")
    else:
        st.info("Set your hyperparameters above, then click **Train**.")

# ────────────────────────────────────────────────────────────────────────────
# 6. Save / load
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Save & load a model")

    with st.expander("📖 Why state_dict, not the whole model?", expanded=True):
        st.markdown(
            "`state_dict` is a plain dict mapping each layer name to its learned weights and "
            "biases — nothing else. Saving it, rather than `pickle`-ing the entire model "
            "object, is the recommended approach because:\n\n"
            "- it **decouples weights from code** — you can refactor the class definition "
            "without breaking old checkpoints, as long as the layer shapes still line up\n"
            "- it's portable across PyTorch versions and machines\n"
            "- loading a fully pickled model executes arbitrary code embedded in the file; "
            "loading just a `state_dict` into a class *you* define is safer\n\n"
            "To restore, recreate a model with the **same architecture**, then load the saved "
            "parameters into it."
        )

    if "trained_state_dict" in st.session_state:
        st.download_button(
            "⬇️ Download trained model.pth",
            data=st.session_state["trained_state_dict"],
            file_name="model.pth",
            mime="application/octet-stream",
        )
        st.caption("Reload it with:")
        st.code(
            "model = NeuralNetwork(2, 2)\n"
            "model.load_state_dict(torch.load(\"model.pth\", weights_only=True))\n"
            "model.eval()",
            language="python",
        )
    else:
        st.info("Train a model in the **Train it Live** tab first — this tab will let you download its weights.")

# ────────────────────────────────────────────────────────────────────────────
# 7. Devices & DDP
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("Devices")
    st.markdown(
        "In PyTorch a **device** is where a tensor lives and where its operations run — CPU, "
        "a CUDA GPU, or Apple `mps`. The rule: **all tensors in an operation must be on the "
        "same device**; move them with `.to(device)`. Moving training onto a single GPU takes "
        "only three additions to the loop: pick a `device`, move the **model** onto it, move "
        "each **batch** onto it. On a tiny toy dataset a GPU won't even be faster — transfer "
        "cost dominates — but for large networks and LLMs the speedup is dramatic. "
        "This app auto-detects what's available:"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("CUDA available", str(torch.cuda.is_available()))
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    c2.metric("Apple MPS available", str(mps_ok))
    c3.metric("CPU threads", torch.get_num_threads())

    st.code(
        'if torch.cuda.is_available():\n'
        '    device = torch.device("cuda")\n'
        'elif torch.backends.mps.is_available():\n'
        '    device = torch.device("mps")\n'
        'else:\n'
        '    device = torch.device("cpu")\n\n'
        'model.to(device)\n'
        'features, labels = features.to(device), labels.to(device)',
        language="python",
    )

    st.divider()
    st.header("Multi-GPU training with DDP (read-only)")
    st.markdown(
        "For **multiple** GPUs, `DistributedDataParallel` (DDP) runs one process per GPU, "
        "each holding a full model replica:\n\n"
        "1. each process forward/backward-passes a *different shard* of the batch, "
        "independently and in parallel\n"
        "2. gradients are synchronized across processes with an **all-reduce** (averaged) "
        "after every backward pass\n"
        "3. because every replica now holds the same averaged gradient, every "
        "`optimizer.step()` keeps all replicas in sync\n\n"
        "This is why speedup isn't perfectly linear with GPU count — the all-reduce "
        "communication is overhead that grows with the number of processes."
    )

    fig_ddp = go.Figure()
    for i, x in enumerate([0, 3, 6]):
        fig_ddp.add_shape(type="rect", x0=x, x1=x + 2, y0=2, y1=3.2,
                          line=dict(color="#4C72B0"), fillcolor="rgba(76,114,176,0.15)")
        fig_ddp.add_annotation(x=x + 1, y=2.6, text=f"GPU {i}<br>model replica<br>+ batch shard {i}",
                               showarrow=False, font=dict(size=11))
        fig_ddp.add_shape(type="line", x0=x + 1, y0=2, x1=4, y1=0.8, line=dict(color="#888", width=1.5))
    fig_ddp.add_shape(type="rect", x0=3, x1=5, y0=0, y1=0.8,
                      line=dict(color="#DD8452"), fillcolor="rgba(221,132,82,0.2)")
    fig_ddp.add_annotation(x=4, y=0.4, text="all-reduce<br>(average gradients)", showarrow=False, font=dict(size=11))
    fig_ddp.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.5, 8.5]), yaxis=dict(visible=False, range=[-0.3, 3.6]),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ddp, width='stretch')

    st.markdown(
        "This can't run inside a Streamlit session — it needs `torchrun` launching real "
        "processes across real GPUs — so here's the reference script from the notebook to run "
        "separately:"
    )
    try:
        with open("DDP-script-torchrun.py") as f:
            st.code(f.read(), language="python")
        st.caption("Run with: `torchrun --nproc_per_node=<num_gpus> DDP-script-torchrun.py`")
    except FileNotFoundError:
        st.warning("DDP-script-torchrun.py not found alongside this app.")
