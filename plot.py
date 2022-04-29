import matplotlib.pyplot as plt
import networkx as nx

network_style = dict(
    node_kwargs = dict(
        node_size=3000, 
        node_color='lightblue', 
        node_shape='o', 
        alpha=1.,
        linewidths=2, 
        edgecolors='black'
    ),
    edge_kwargs = dict(
        width=1.5, 
        edge_color='k', 
        style='solid', 
        alpha=None, 
        arrowstyle='->', 
        arrowsize=25, 
        node_size=3000, 
        node_shape='o', 
        connectionstyle='arc3'
    ),
    node_label_kwargs = dict(
        font_size=13, 
        font_color='k', 
        font_family='serif',
        horizontalalignment='center', 
        verticalalignment='center'
    ),
    edge_label_kwargs = dict(
        label_pos=0.5, 
        font_size=13, 
        font_color='tomato', 
        font_family='serif', 
        horizontalalignment='center', 
        verticalalignment='center'
    ),
)

def plot_network(edges):
    G = nx.MultiDiGraph()
    G.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    plt.figure(figsize=(13, len(G.nodes)*2))
    nx.draw_networkx_nodes(
        G, pos, 
        **network_style['node_kwargs']
    )
    nx.draw_networkx_edges(
        G, pos, 
        **network_style['edge_kwargs']
    )
    nx.draw_networkx_labels(
        G, pos, 
        **network_style['node_label_kwargs']
    )
    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels={(u, v):d for u, v, d in G.edges}, 
        **network_style['edge_label_kwargs']
    )
    plt.axis('off')
    plt.show()