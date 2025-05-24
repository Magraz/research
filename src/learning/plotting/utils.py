import matplotlib.pyplot as plt

import numpy as np

import networkx as nx

from matplotlib.patches import Rectangle

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch

# GAT plots


def plot_diagonal_attention_timeline(
    edge_indices,
    attention_weights_over_time,
    figsize=(12, 10),
    cmap="viridis",
    num_samples=5,
):
    """
    Creates a diagonal overlay visualization of attention heatmaps over time

    Args:
        edge_indices: List of edge_index tensors for each timestep
        attention_weights_over_time: List of attention weight tensors
        figsize: Size of the figure
        cmap: Colormap to use
        num_samples: Number of timesteps to sample
    """
    # Sample timesteps if needed
    total_timesteps = len(edge_indices)
    if total_timesteps > num_samples:
        timestep_indices = np.linspace(0, total_timesteps - 1, num_samples, dtype=int)
    else:
        timestep_indices = np.arange(total_timesteps)

    num_timesteps = len(timestep_indices)

    # Get maximum number of nodes
    max_nodes = max([edge_indices[i].max().item() + 1 for i in timestep_indices])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate alpha values (increasing for newer timesteps)
    alphas = np.linspace(0.3, 0.9, num_timesteps)

    # Calculate overlap percentage
    overlap_percent = 0.7

    # Size of each heatmap (adjusted to fit diagonally)
    heatmap_width = 0.8 / (1 + (num_timesteps - 1) * (1 - overlap_percent))
    heatmap_height = 0.8 / (1 + (num_timesteps - 1) * (1 - overlap_percent))

    # Plot each timestep with increasing alpha and diagonal shift
    for idx, t in enumerate(timestep_indices):
        # Calculate position for this timestep's heatmap
        pos_x = 0.1 + idx * heatmap_width * (1 - overlap_percent)
        pos_y = 0.9 - heatmap_height - idx * heatmap_height * (1 - overlap_percent)

        # Create a new axes for this timestep
        ax_t = fig.add_axes([pos_x, pos_y, heatmap_width, heatmap_height])

        # Get data for this timestep
        edges = edge_indices[t].cpu().numpy()
        weights = attention_weights_over_time[t].detach().cpu().numpy()

        # Average over heads if multiple
        if weights.shape[1] > 1:
            edge_weights = weights.mean(axis=1)
        else:
            edge_weights = weights.squeeze()

        # Create attention matrix
        attention_matrix = np.zeros((max_nodes, max_nodes))
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            if src < max_nodes and dst < max_nodes:  # Safety check
                attention_matrix[src, dst] = edge_weights[i]

        # Plot the heatmap with appropriate alpha
        im = ax_t.imshow(attention_matrix, cmap=cmap, alpha=alphas[idx], vmin=0, vmax=1)

        # Add a timestep label
        ax_t.set_title(f"t={t+1}", fontsize=10)

        # Only show axis labels for the last timestep
        if idx < num_timesteps - 1:
            ax_t.set_xticks([])
            ax_t.set_yticks([])
        else:
            ax_t.set_xlabel("Target Node", fontsize=8)
            ax_t.set_ylabel("Source Node", fontsize=8)

            # Add proper tick labels
            ax_t.set_xticks(np.arange(max_nodes))
            ax_t.set_yticks(np.arange(max_nodes))
            ax_t.set_xticklabels(range(max_nodes), fontsize=8)
            ax_t.set_yticklabels(range(max_nodes), fontsize=8)

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention Weight")

    # Add a legend showing alpha progression
    legend_elements = []
    for idx, t in enumerate(timestep_indices):
        legend_elements.append(
            Rectangle(
                (0, 0),
                1,
                1,
                fc=plt.cm.get_cmap(cmap)(0.7),
                alpha=alphas[idx],
                label=f"Timestep {t+1}",
            )
        )

    # Add legend in an empty part of the figure
    legend_ax = fig.add_axes([0.05, 0.05, 0.2, 0.1])
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_elements, loc="center")

    # Turn off the main axis
    ax.axis("off")

    # Set figure title
    fig.suptitle(
        "Attention Weights Evolution\n(Newer timesteps overlay with higher opacity)",
        fontsize=14,
    )

    return fig


def plot_gat_attention_as_graph(edge_index, attention_weights, figsize=(12, 10)):
    """
    Plot a graph with edges colored according to attention weights

    Args:
        edge_index: Tensor of shape [2, num_edges] containing edge connections
        attention_weights: Tensor of shape [num_edges, num_heads] containing attention scores
        figsize: Size of the figure
    """
    # Convert to numpy for easier handling
    edges = edge_index.cpu().numpy()
    weights = attention_weights.detach().cpu().numpy()

    # If there are multiple attention heads, average them
    if weights.shape[1] > 1:
        edge_weights = weights.mean(axis=1)
        print(f"Averaging {weights.shape[1]} attention heads")
    else:
        edge_weights = weights.squeeze()

    # Create a directed graph
    G = nx.DiGraph()

    # Get number of nodes from edge indices
    num_nodes = max(edges.max() + 1, 8)  # At least 8 nodes for your example

    # Add all nodes (including isolated ones)
    G.add_nodes_from(range(num_nodes))

    # Add edges with weights
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        G.add_edge(src.item(), dst.item(), weight=edge_weights[i])

    # Set up the plot - FIXED: Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=figsize)

    # Use a layout that spreads nodes nicely
    if num_nodes <= 8:  # For small graphs
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Normalize edge weights for coloring
    edge_weights_list = [G[u][v]["weight"] for u, v in G.edges()]
    min_weight = min(edge_weights_list) if edge_weights_list else 0
    max_weight = max(edge_weights_list) if edge_weights_list else 1
    normalized_weights = [
        (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
        for w in edge_weights_list
    ]

    # Draw the nodes - FIXED: Pass ax parameter
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", ax=ax)

    # Draw the edges with a colormap based on weight - FIXED: Pass ax parameter
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color=normalized_weights,
        edge_cmap=plt.cm.Blues,
        width=4,
        ax=ax,
    )

    # Add a colorbar - FIXED: Pass ax parameter to colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_weight, vmax=max_weight)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Attention Weight")

    # Add labels - FIXED: Pass ax parameter
    nx.draw_networkx_labels(G, pos, font_size=14, ax=ax)

    ax.set_title("Graph Attention Visualization", fontsize=16)
    ax.axis("off")
    plt.tight_layout()

    # Return the figure for further customization
    return fig


def plot_attention_heatmap(
    edge_index, attention_weights, figsize=(10, 8), cmap="Blues"
):
    """
    Plot attention weights as a matrix heatmap

    Args:
        edge_index: Tensor of shape [2, num_edges] containing edge connections
        attention_weights: Tensor of shape [num_edges, num_heads] containing attention scores
        figsize: Size of the figure
        cmap: Colormap to use
    """
    # Convert to numpy for easier handling
    edges = edge_index.cpu().numpy()
    weights = attention_weights.detach().cpu().numpy()

    # If there are multiple attention heads, average them
    if weights.shape[1] > 1:
        edge_weights = weights.mean(axis=1)
        print(f"Averaging {weights.shape[1]} attention heads")
    else:
        edge_weights = weights.squeeze()

    # Get number of nodes
    num_nodes = max(edges.max() + 1, 8)

    # Create an empty attention matrix
    attention_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the matrix with attention weights
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        attention_matrix[src, dst] = edge_weights[i]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(attention_matrix, cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(num_nodes))
    ax.set_yticks(np.arange(num_nodes))
    ax.set_xticklabels([f"Node {i}" for i in range(num_nodes)])
    ax.set_yticklabels([f"Node {i}" for i in range(num_nodes)])

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add title and labels
    ax.set_title("Attention Weights Matrix")
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")

    # Loop over data dimensions and create text annotations
    for i in range(num_nodes):
        for j in range(num_nodes):
            if attention_matrix[i, j] > 0:
                text = ax.text(
                    j,
                    i,
                    f"{attention_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if attention_matrix[i, j] < 0.5 else "white",
                )

    # Adjust layout
    fig.tight_layout()

    return fig


def plot_attention_time_series(edge_indices, attention_weights_over_time, top_k=5):
    """
    Plot the attention weights over time for the top-k most important edges

    Args:
        edge_indices: List of edge_index tensors
        attention_weights_over_time: List of attention weight tensors
        top_k: Number of top edges to track
    """
    num_timesteps = len(edge_indices)

    # Find edges that appear in all timesteps
    # For simplicity, we'll assume the graph structure stays constant
    edges = edge_indices[0].cpu().numpy()
    edge_pairs = [(edges[0, i], edges[1, i]) for i in range(edges.shape[1])]

    # Calculate average attention over time for each edge
    edge_avg_attention = np.zeros(len(edge_pairs))
    for t in range(num_timesteps):
        weights = attention_weights_over_time[t].detach().cpu().numpy()
        if weights.shape[1] > 1:  # If multiple heads
            weights = weights.mean(axis=1)
        else:
            weights = weights.squeeze()

        edge_avg_attention += weights

    edge_avg_attention /= num_timesteps

    # Get indices of top-k edges by average attention
    top_indices = np.argsort(edge_avg_attention)[-top_k:]

    # Prepare time series data
    time_steps = np.arange(1, num_timesteps + 1)
    attention_series = np.zeros((top_k, num_timesteps))

    for t in range(num_timesteps):
        weights = attention_weights_over_time[t].detach().cpu().numpy()
        if weights.shape[1] > 1:
            weights = weights.mean(axis=1)
        else:
            weights = weights.squeeze()

        for i, idx in enumerate(top_indices):
            attention_series[i, t] = weights[idx]

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, idx in enumerate(top_indices):
        src, dst = edge_pairs[idx]
        ax.plot(
            time_steps,
            attention_series[i],
            linewidth=2,
            label=f"Edge {src}→{dst}",
        )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title("Attention Evolution for Top Edges", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def plot_3d_attention_surface(edge_indices, attention_weights_over_time):
    """
    Create a 3D surface plot showing attention evolution

    Args:
        edge_indices: List of edge_index tensors
        attention_weights_over_time: List of attention weight tensors
    """
    num_timesteps = len(edge_indices)
    max_nodes = max([edge.max().item() + 1 for edge in edge_indices])

    # Create 3D matrix of attention weights [time, source, target]
    attention_cube = np.zeros((num_timesteps, max_nodes, max_nodes))

    for t in range(num_timesteps):
        edges = edge_indices[t].cpu().numpy()
        weights = attention_weights_over_time[t].detach().cpu().numpy()

        if weights.shape[1] > 1:
            edge_weights = weights.mean(axis=1)
        else:
            edge_weights = weights.squeeze()

        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            attention_cube[t, src, dst] = edge_weights[i]

    # Create mesh grid
    time_steps = np.arange(num_timesteps)
    source_nodes = np.arange(max_nodes)
    target_nodes = np.arange(max_nodes)

    T, S = np.meshgrid(time_steps, source_nodes)

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot multiple surfaces, one for each target node
    for target in target_nodes:
        surf = ax.plot_surface(
            T, S, attention_cube[:, :, target].T, alpha=0.7, label=f"Target {target}"
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Source Node")
    ax.set_zlabel("Attention Weight")
    ax.set_title("3D Attention Evolution")

    return fig


def plot_3d_attention_scatter(
    edge_indices,
    attention_weights_over_time,
    figsize=(12, 10),
    cmap="viridis",
    sample_rate=1,
):
    """
    Create a 3D scatter plot of attention weights with:
    - X-axis: Source nodes
    - Y-axis: Target nodes
    - Z-axis: Time steps
    - Color: Attention magnitude

    Args:
        edge_indices: List of edge_index tensors for each timestep
        attention_weights_over_time: List of attention weight tensors
        figsize: Figure size
        cmap: Colormap for attention weights
        sample_rate: Sample every N timesteps (for reducing visual clutter)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Sample timesteps to reduce visual clutter
    timesteps = list(range(0, len(edge_indices), sample_rate))

    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Lists to store all data points
    all_sources = []
    all_targets = []
    all_times = []
    all_weights = []
    all_sizes = []

    # Process data for all timesteps
    for t_idx, t in enumerate(timesteps):
        # Get edge data
        edges = edge_indices[t].cpu().numpy()
        weights = attention_weights_over_time[t].detach().cpu().numpy()

        # Average over heads if multiple
        if weights.shape[1] > 1:
            edge_weights = weights.mean(axis=1)
        else:
            edge_weights = weights.squeeze()

        # Store source, target, time, and weight for each edge
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            weight = edge_weights[i]

            all_sources.append(src)
            all_targets.append(dst)
            all_times.append(t)
            all_weights.append(weight)

            # Size based on weight (makes important connections more visible)
            all_sizes.append(100 * weight + 20)

    # Convert to numpy arrays
    all_sources = np.array(all_sources)
    all_targets = np.array(all_targets)
    all_times = np.array(all_times)
    all_weights = np.array(all_weights)
    all_sizes = np.array(all_sizes)

    # Normalize weights for colormapping
    norm_weights = (all_weights - all_weights.min()) / (
        all_weights.max() - all_weights.min() + 1e-8
    )

    # Create scatter plot
    scatter = ax.scatter(
        all_sources,
        all_targets,
        all_times,
        c=all_weights,
        s=all_sizes,
        cmap=cmap,
        alpha=0.7,
        edgecolors="w",
        linewidth=0.5,
    )

    # Set labels and title
    ax.set_xlabel("Source Node", fontsize=12)
    ax.set_ylabel("Target Node", fontsize=12)
    ax.set_zlabel("Timestep", fontsize=12)
    ax.set_title("3D Attention Weights Visualization", fontsize=14)

    # Set integer ticks for nodes
    max_node = max(all_sources.max(), all_targets.max())
    ax.set_xticks(np.arange(max_node + 1))
    ax.set_yticks(np.arange(max_node + 1))

    # Set integer ticks for timesteps (but not too many)
    if len(timesteps) > 10:
        z_ticks = np.linspace(0, max(timesteps), 10, dtype=int)
    else:
        z_ticks = timesteps
    ax.set_zticks(z_ticks)

    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Attention Weight", rotation=270, labelpad=20)

    # Improve perspective
    ax.view_init(elev=20, azim=45)

    return fig


def plot_3d_attention_interactive(
    edge_indices, attention_weights_over_time, sample_rate=5
):
    """
    Create an interactive 3D scatter plot of attention weights that you can rotate and zoom

    Args:
        edge_indices: List of edge_index tensors for each timestep
        attention_weights_over_time: List of attention weight tensors
        sample_rate: Sample every N timesteps (for reducing visual clutter)
    """

    # Sample timesteps to reduce visual clutter
    timesteps = list(range(0, len(edge_indices), sample_rate))

    # Lists to store all data points
    all_sources = []
    all_targets = []
    all_times = []
    all_weights = []
    all_texts = []

    # Process data for all timesteps
    for t_idx, t in enumerate(timesteps):
        # Get edge data
        edges = edge_indices[t].cpu().numpy()
        weights = attention_weights_over_time[t].detach().cpu().numpy()

        # Average over heads if multiple
        if weights.shape[1] > 1:
            edge_weights = weights.mean(axis=1)
        else:
            edge_weights = weights.squeeze()

        # Store source, target, time, and weight for each edge
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            weight = edge_weights[i]

            all_sources.append(src)
            all_targets.append(dst)
            all_times.append(t)
            all_weights.append(weight)
            all_texts.append(
                f"Source: {src}<br>Target: {dst}<br>Time: {t}<br>Weight: {weight:.3f}"
            )

    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=all_sources,
                y=all_targets,
                z=all_times,
                mode="markers",
                marker=dict(
                    size=10,
                    color=all_weights,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="Attention Weight"),
                    line=dict(color="white", width=0.5),
                ),
                text=all_texts,
                hoverinfo="text",
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="Interactive 3D Attention Weights Visualization",
        scene=dict(
            xaxis_title="Source Node",
            yaxis_title="Target Node",
            zaxis_title="Timestep",
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis=dict(tickmode="linear", dtick=1),
        ),
        width=900,
        height=800,
    )

    # Save to HTML for interactive viewing
    fig.write_html("interactive_attention_3d.html")

    return fig


def plot_3d_attention_volume(
    edge_indices,
    attention_weights_over_time,
    figsize=(10, 8),
    cmap="viridis",
    sample_rate=1,
):
    """
    Create a 3D volume visualization of attention weights as a cube

    Args:
        edge_indices: List of edge_index tensors
        attention_weights_over_time: List of attention weight tensors
        figsize: Figure size
        cmap: Colormap for attention weights
        sample_rate: Sample every N timesteps
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Sample timesteps
    timesteps = list(range(0, len(edge_indices), sample_rate))
    num_timesteps = len(timesteps)

    # Get max number of nodes
    max_nodes = max([edge_indices[t].max().item() + 1 for t in timesteps])

    # Create 3D attention volume [time, source, target]
    attention_volume = np.zeros((num_timesteps, max_nodes, max_nodes))

    # Fill in attention volume
    for i, t in enumerate(timesteps):
        edges = edge_indices[t].cpu().numpy()
        weights = attention_weights_over_time[t].detach().cpu().numpy()

        if weights.shape[1] > 1:
            edge_weights = weights.mean(axis=1)
        else:
            edge_weights = weights.squeeze()

        for j in range(edges.shape[1]):
            src, dst = edges[0, j], edges[1, j]
            attention_volume[i, src, dst] = edge_weights[j]

    # Create figure for 3D visualization
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create coordinate arrays
    t_indices, s_indices, d_indices = np.indices((num_timesteps, max_nodes, max_nodes))

    # Scale to match the colormap
    vmin = np.min(attention_volume)
    vmax = np.max(attention_volume)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap(cmap)

    # Plot non-zero values as colored voxels
    mask = attention_volume > 0.01  # threshold to reduce noise
    colors = cmap(norm(attention_volume))

    # Create the 3D voxel plot
    ax.voxels(mask, facecolors=colors, edgecolor="k", linewidth=0.1, alpha=0.7)

    # Set labels
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Source Node")
    ax.set_zlabel("Target Node")

    # Set ticks
    ax.set_xticks(np.linspace(0, num_timesteps - 1, min(5, num_timesteps)))
    ax.set_xticklabels(
        [
            timesteps[int(i)]
            for i in np.linspace(0, num_timesteps - 1, min(5, num_timesteps)).astype(
                int
            )
        ]
    )

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Attention Weight")

    ax.set_title("3D Attention Volume Visualization")

    return fig


# Transformer plots


def plot_transformer_attention(attn_matrix, layer_name="", head_idx=0, figsize=(10, 8)):
    """
    Plot a single attention head as a heatmap

    Args:
        attn_matrix: Attention matrix with shape [B, H, L, L]
        layer_name: Name of the layer (for title)
        head_idx: Which attention head to visualize
        figsize: Size of the figure
    """
    # Extract the first batch item and specified head
    attention = attn_matrix[0, head_idx].cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        attention,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        cbar=True,
        ax=ax,
    )

    # Set labels and title
    ax.set_xlabel("Target Token Position")
    ax.set_ylabel("Source Token Position")
    ax.set_title(f"{layer_name} - Attention Head {head_idx}")

    # Custom tick labels for agent positions
    n_agents = attention.shape[0]
    ax.set_xticklabels([f"Agent {i+1}" for i in range(n_agents)])
    ax.set_yticklabels([f"Agent {i+1}" for i in range(n_agents)])

    plt.tight_layout()
    return fig


def plot_all_attention_heads(attn_matrix, layer_name="", figsize=(15, 12)):
    """
    Plot all attention heads in a grid

    Args:
        attn_matrix: Attention matrix with shape [B, H, L, L]
        layer_name: Name of the layer (for title)
        figsize: Size of the figure
    """
    batch_size, n_heads, seq_len, _ = attn_matrix.shape

    # Create grid layout based on number of heads
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(n_heads):
        if h < len(axes):
            attention = attn_matrix[0, h].cpu().numpy()

            sns.heatmap(
                attention,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                linewidths=0.5,
                cbar=True if h == 0 else False,
                ax=axes[h],
            )

            axes[h].set_title(f"Head {h+1}")

            if h % n_cols == 0:  # Leftmost plots
                axes[h].set_ylabel("Source Agent")
            else:
                axes[h].set_ylabel("")

            if h >= n_heads - n_cols:  # Bottom row
                axes[h].set_xlabel("Target Agent")
            else:
                axes[h].set_xlabel("")

            # Custom tick labels
            axes[h].set_xticklabels([f"{i+1}" for i in range(seq_len)], rotation=45)
            axes[h].set_yticklabels([f"{i+1}" for i in range(seq_len)])

    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"{layer_name} - All Attention Heads", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_averaged_attention(attn_matrix, layer_name="", figsize=(10, 8)):
    """
    Plot attention averaged across all heads

    Args:
        attn_matrix: Attention matrix with shape [B, H, L, L]
        layer_name: Name of the layer (for title)
        figsize: Size of the figure
    """
    # Average over heads dimension
    avg_attention = attn_matrix[0].mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        avg_attention,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        cbar=True,
        ax=ax,
    )

    # Set labels and title
    ax.set_xlabel("Target Agent")
    ax.set_ylabel("Source Agent")
    ax.set_title(f"{layer_name} - Average Attention Across All Heads")

    # Custom tick labels
    n_agents = avg_attention.shape[0]
    ax.set_xticklabels([f"{i+1}" for i in range(n_agents)])
    ax.set_yticklabels([f"{i+1}" for i in range(n_agents)])

    plt.tight_layout()
    return fig


def plot_attention_across_layers(attn_scores_dict, figsize=(15, 10)):
    """
    Plot average attention for each layer side by side

    Args:
        attn_scores_dict: Dictionary of attention matrices {layer_name: attn_matrix}
        figsize: Size of the figure
    """
    n_layers = len(attn_scores_dict)

    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]

    for i, (layer_name, attn_matrix) in enumerate(attn_scores_dict.items()):
        # Average over heads dimension
        avg_attention = attn_matrix[0].mean(dim=0).cpu().numpy()

        # Create heatmap
        sns.heatmap(
            avg_attention,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            linewidths=0.5,
            cbar=True if i == n_layers - 1 else False,
            ax=axes[i],
        )

        axes[i].set_title(f"{layer_name}")
        axes[i].set_xlabel("Target Agent")

        if i == 0:  # Only first plot shows y-label
            axes[i].set_ylabel("Source Agent")

        # Custom tick labels
        n_agents = avg_attention.shape[0]
        axes[i].set_xticklabels([f"{i+1}" for i in range(n_agents)])
        axes[i].set_yticklabels([f"{i+1}" for i in range(n_agents)])

    fig.suptitle("Attention Patterns Across Layers", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_attention_over_time_grid(
    attention_over_time, attn_type="Enc_L0", head_idx=0, num_samples=5, figsize=(20, 6)
):
    """
    Plot attention at selected timesteps in a grid

    Args:
        attention_over_time: Dict with lists of attention weights per timestep
        attn_type: Type of attention to visualize
        head_idx: Which attention head to visualize
        num_samples: Number of timesteps to sample and display
        figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    attention_matrices = attention_over_time[attn_type]
    total_timesteps = len(attention_matrices)

    # Sample evenly spaced timesteps
    if total_timesteps > num_samples:
        sample_indices = np.linspace(0, total_timesteps - 1, num_samples, dtype=int)
    else:
        sample_indices = np.arange(total_timesteps)

    # Create figure with subplots
    fig, axes = plt.subplots(1, len(sample_indices), figsize=figsize)
    if len(sample_indices) == 1:
        axes = [axes]

    # Plot each sampled timestep
    for i, t_idx in enumerate(sample_indices):
        attn_matrix = attention_matrices[t_idx][0, head_idx].cpu().numpy()

        sns.heatmap(
            attn_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=(i == len(sample_indices) - 1),  # Only add colorbar on last plot
            ax=axes[i],
        )

        axes[i].set_title(f"Timestep {t_idx+1}")

        if i == 0:  # Only add y-label to leftmost plot
            axes[i].set_ylabel("Source Position")
        else:
            axes[i].set_ylabel("")

        axes[i].set_xlabel("Target Position")

        # Add tick labels
        n_positions = attn_matrix.shape[0]
        axes[i].set_xticklabels([f"{j+1}" for j in range(n_positions)], rotation=45)
        axes[i].set_yticklabels([f"{j+1}" for j in range(n_positions)])

    # Add overall title
    type_label = {
        "Enc_L0": "Encoder Self-Attention",
        "Dec_L0": "Decoder Self-Attention",
        "Cross_L0": "Cross-Attention",
    }
    fig.suptitle(
        f"{type_label.get(attn_type, attn_type)} - Head {head_idx+1} Evolution",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_key_attention_trends(
    attention_over_time, attn_type="Enc_L0", head_idx=0, top_k=5, figsize=(12, 8)
):
    """
    Plot the top-k attention connections over time as a line chart

    Args:
        attention_over_time: Dict with lists of attention weights per timestep
        attn_type: Type of attention to visualize ('Enc_L0', 'Dec_L0', or 'Cross_L0')
        head_idx: Which attention head to visualize
        top_k: Number of top connections to track
        figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get the attention matrices for the specified type
    attention_matrices = attention_over_time[attn_type]

    if not attention_matrices:
        print(f"No attention data found for {attn_type}")
        return plt.figure(figsize=(6, 4))

    num_timesteps = len(attention_matrices)

    # Get matrix dimensions from first timestep
    batch_size, num_heads, seq_len, _ = attention_matrices[0].shape

    # Calculate average attention over time for each source-target pair
    # to determine which ones are most important
    avg_attention = torch.zeros((seq_len, seq_len))

    for t in range(num_timesteps):
        avg_attention += attention_matrices[t][0, head_idx].cpu()

    avg_attention /= num_timesteps

    # Find top-k connections
    flat_indices = torch.topk(avg_attention.view(-1), top_k).indices
    top_src = flat_indices // seq_len
    top_tgt = flat_indices % seq_len

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Line colors
    colors = plt.cm.viridis(np.linspace(0, 1, top_k))

    # Prepare time series data
    time_steps = np.arange(1, num_timesteps + 1)

    # Plot each top connection
    for i in range(top_k):
        src, tgt = top_src[i].item(), top_tgt[i].item()
        values = []

        for t in range(num_timesteps):
            values.append(attention_matrices[t][0, head_idx, src, tgt].item())

        ax.plot(
            time_steps,
            values,
            markersize=4,
            linewidth=2,
            color=colors[i],
            label=f"Src {src+1} → Tgt {tgt+1}",
        )

    # Add labels and title
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)

    type_label = {
        "Enc_L0": "Encoder Self-Attention",
        "Dec_L0": "Decoder Self-Attention",
        "Cross_L0": "Cross-Attention",
    }

    ax.set_title(
        f"{type_label.get(attn_type, attn_type)} - Head {head_idx+1}\nTop {top_k} Attention Connections",
        fontsize=14,
    )

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # Format y-axis ticks as percentages
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Add horizontal lines at key thresholds
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_token_attention_trends(
    attention_over_time, attn_type="Enc_L0", src_idx=0, head_idx=0, figsize=(12, 8)
):
    """
    Plot how one specific token attends to all other tokens over time

    Args:
        attention_over_time: Dict with lists of attention weights per timestep
        attn_type: Type of attention to visualize ('Enc_L0', 'Dec_L0', or 'Cross_L0')
        src_idx: Source token index to focus on (which token is doing the attending)
        head_idx: Which attention head to visualize
        figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get the attention matrices for the specified type
    attention_matrices = attention_over_time[attn_type]

    if not attention_matrices:
        print(f"No attention data found for {attn_type}")
        return plt.figure(figsize=(6, 4))

    num_timesteps = len(attention_matrices)

    # Get matrix dimensions from first timestep
    batch_size, num_heads, seq_len, _ = attention_matrices[0].shape

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Line colors - one for each target token
    colors = plt.cm.viridis(np.linspace(0, 1, seq_len))

    # Prepare time series data
    time_steps = np.arange(1, num_timesteps + 1)

    # Plot attention from src_idx to all target tokens
    for tgt_idx in range(seq_len):
        values = []

        for t in range(num_timesteps):
            values.append(attention_matrices[t][0, head_idx, src_idx, tgt_idx].item())

        label = f"Token {src_idx+1} → Token {tgt_idx+1}"
        if src_idx == tgt_idx:
            label += " (self)"
            line_style = "--"
            line_width = 2.5
        else:
            line_style = "-"
            line_width = 2

        ax.plot(
            time_steps,
            values,
            marker="o" if src_idx == tgt_idx else None,
            markersize=3,
            linewidth=line_width,
            linestyle=line_style,
            color=colors[tgt_idx],
            label=label,
        )

    # Add labels and title
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)

    type_label = {
        "Enc_L0": "Encoder Self-Attention",
        "Dec_L0": "Decoder Self-Attention",
        "Cross_L0": "Cross-Attention",
    }

    ax.set_title(
        f"{type_label.get(attn_type, attn_type)} - Head {head_idx+1}\nToken {src_idx+1} Attending to All Tokens",
        fontsize=14,
    )

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # Format y-axis ticks as percentages
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    plt.tight_layout()
    return fig
