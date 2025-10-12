import numpy as np
import cobra
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import networkx as nx
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigs
plt.rcParams['font.family'] = 'Arial'

shared_medium = {
    'EX_mn2(e)': 1000.0, 'EX_cu2(e)': 1000.0, 'EX_pi(e)': 1000.0,
    'EX_cobalt2(e)': 1000.0, 'EX_h(e)': 1000.0, 'EX_mg2(e)': 1000.0,
    'EX_co2(e)': 1000.0, 'EX_o2(e)': 1000.0, 'EX_cl(e)': 1000.0,
    'EX_zn2(e)': 1000.0, 'EX_so4(e)': 1000.0, 'EX_fe3(e)': 1000.0,
    'EX_h2o(e)': 1000.0, 'EX_k(e)': 1000.0, 'EX_nh4(e)': 1000.0,
    'EX_ca2(e)': 1000.0, 'EX_na1(e)': 1000.0, 'EX_fe2(e)': 1000.0
}

def compute_stationary_distribution(P, tol=1e-12):
    """
    Compute the stationary distribution π of the transition matrix P.
    π satisfies: P^T π = π, sum(π) = 1
    """
    n = P.shape[0]
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))  # 找到最接近 1 的特征值
    print(eigvecs)
    pi = np.real(eigvecs[:, idx])
    pi = pi / np.sum(pi)
    pi = np.maximum(pi, 0)
    pi = pi / np.sum(pi)  # Renormalize to make sure sum = 1
    return pi

def chung_laplacian_spectrum_directed(G):
    """
    Construct Chung's symmetric Laplacian for a directed graph G.
    Return the sorted real eigenvalues (spectrum).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Build weighted adjacency matrix A
    A = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        A[i, j] = data.get("weight", 1.0)
    
    # Build transition matrix P (row-normalized)
    row_sums = A.sum(axis=1, keepdims=True)
    P = np.divide(A, row_sums, where=row_sums != 0)
    
    # Compute stationary distribution π
    pi = compute_stationary_distribution(P)
    
    # Φ = diag(pi)
    sqrt_Phi = np.diag(np.sqrt(pi))
    inv_sqrt_Phi = np.diag(1 / np.sqrt(pi))
    
    # Chung symmetric Laplacian
    sym_P = 0.5 * (sqrt_Phi @ P @ inv_sqrt_Phi + inv_sqrt_Phi @ P.T @ sqrt_Phi)
    L = np.eye(n) - sym_P

    # Compute sorted eigenvalues
    spectrum = eigvalsh(L)  # Returns sorted real eigenvalues
    return spectrum



def get_inner_graph(G_coarse, inner_nodes):
    # 创建有向子图
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(inner_nodes)

    for u in inner_nodes:
        for v in inner_nodes:
            if u != v and G_coarse.has_edge(u, v):
                edge_data = G_coarse.get_edge_data(u, v)
                if edge_data is not None:
                    # 把所有属性传过去，比如 weight 等
                    subgraph.add_edge(u, v, **edge_data)

    return subgraph

def calculate_trophic_levels(G_coarse, inner_nodes):
    """
    Calculate the trophic levels of a directed graph.
    
    Parameters:
    G_coarse (networkx.DiGraph): The directed graph.
    inner_nodes (list): List of nodes to consider for trophic level calculation.
    
    Returns:
    dict: A dictionary with nodes as keys and their trophic levels as values.
    """
    # Create a subgraph with only the inner nodes
    subgraph = get_inner_graph(G_coarse, inner_nodes)
    components = list(nx.weakly_connected_components(subgraph))
    largest_wcc = max(components, key=len)
    sg = subgraph.subgraph(largest_wcc).copy()
    F_0, h = ta.trophic_incoherence(sg)

    # 映射：{node: trophic level}
    node_list = list(sg.nodes())
    h_dict = {node: h[i] for i, node in enumerate(node_list)}

    h_final = {}
    for node in subgraph.nodes():
        if str(node) in list(h_dict.keys()):
            h_final[node] = h_dict[node]
        else:
            h_final[node] = -1.0
    return F_0, h_final

def construct_mass_flow_graph(S, met_flux, reaction_list):
    """
    Constructs a mass flow graph from the stoichiometric matrix S.
    """
    
    v = np.zeros(S.shape[1])

    uni = {}
    for met in met_flux:
            if met[1] == 0:
                continue
            if '_reverse' in met[0]:
                uni[met[0].replace('_reverse','')] = met[1]
            elif '_forward' in met[0]:
                uni[met[0].replace('_forward','')] = met[1]
            else:
                uni[met[0]] = met[1]
    fixed_uni = [(key, value) for key, value in uni.items()]
    

    for key, value in fixed_uni:
        if key in reaction_list:
            v[reaction_list.index(key)] = value


    # for key, value in met_flux:
    #     if key in reaction_list:
    #         v[reaction_list.index(key)] = value
    
    v_forward = np.maximum(v, 0)
    v_backward = np.maximum(-v, 0)
    v_2m = np.concatenate([v_forward, v_backward])  # shape (2m,)

    # Step 2: Build S_2m^+ and S_2m^-
    S_plus = np.maximum(S, 0)
    S_minus = np.maximum(-S, 0)

    S2m_plus = np.concatenate([S_plus, S_minus], axis=1)   # shape (n, 2m)
    S2m_minus = np.concatenate([S_minus, S_plus], axis=1)  # flip roles for backward

    # print(S2m_plus.shape)
    # print(v_2m.shape)

    j_v = S2m_plus @ v_2m

    diag_v2m = np.diag(v_2m)
    diag_jv_inv = np.diag(np.where(j_v != 0, 1.0 / j_v, 0))

    M = (S2m_plus @ diag_v2m).T @ diag_jv_inv @ (S2m_minus @ diag_v2m)

    M[np.abs(M) < 1e-2] = 0

    return M

def construct_networkx_graph(M, reaction_list):
    G_mfg = nx.DiGraph()
    rxn_nodes = [f"{name}_fwd" for name in reaction_list] + [f"{name}_bwd" for name in reaction_list]

    for i in range(len(rxn_nodes)):
        for j in range(len(rxn_nodes)):
            if M[i, j] > 1e-3:  # threshold to show
                G_mfg.add_edge(rxn_nodes[i], rxn_nodes[j], weight=round(M[i, j], 3))
    # components = list(nx.weakly_connected_components(G_mfg))
    # largest_wcc = max(components, key=len)
    # G_largest = G_mfg.subgraph(largest_wcc).copy()

    return G_mfg


def walk_single_out_path(G, start_node):
    path = [start_node]
    current = start_node

    while True:
        successors = list(G.successors(current))
        if len(successors) != 1:
            break  # 停止：分支 or 终点
        next_node = successors[0]
        
        if 'bio' in next_node or len(path) > 3:
            break
        path.append(next_node)
        current = next_node

    return path

def clustered_nodes4graph(G, met_flux, cluster_nodes):
    except_nodes = sorted([rec+'_bwd' for rec, value in met_flux if ('EX_' in rec and value < -0.01)])
    except_nodes_final = []
    for node in except_nodes:
        try:
            path = walk_single_out_path(G, node)
        except:
            except_nodes_final.append([node])
            continue
        if len(path) <= 2:
            except_nodes_final.append(path[:1])
        else:
            except_nodes_final.append(path[:3])

    nodes = []
    for cluster_node in cluster_nodes.values():
        temp = []
        for node in cluster_node:
            if any(node in subgroup for subgroup in except_nodes_final) or (node not in list(G.nodes())):
                continue
            temp.append(node)
        nodes.append(temp)
    nodes.extend(except_nodes_final)

    return nodes, len(cluster_nodes)


def merge_nodes(graph, merged_nodes):
    """
    Merge nodes in the graph into clusters and update the edges' weights.
    
    Args:
        graph (nx.DiGraph): The original directed graph.
        merged_nodes (list): A list of lists of nodes that form clusters.
        
    Returns:
        coarse_grained_graph (nx.DiGraph): The new coarse-grained graph with updated edges.
    """
    coarse_grained_graph = nx.DiGraph()
    
    # Add cluster nodes
    cluster_to_nodes = {}
    for i, cluster in enumerate(merged_nodes):
        cluster_node = str(i)  # Use a simple integer as the cluster name
        coarse_grained_graph.add_node(cluster_node)  # Add the cluster as a node
        cluster_to_nodes[cluster_node] = cluster  # Map the cluster node to original nodes
    
    # Add edges between clusters based on the original graph
    for i, clusterA in enumerate(merged_nodes):
        for j, clusterB in enumerate(merged_nodes):
            if i == j:
                continue  # Skip the self-loops between the same clusters

            # Calculate the weight of the edge from clusterA to clusterB
            total_weight = 0
            edge_count = 0
            for nodeA in clusterA:
                for nodeB in clusterB:
                    if graph.has_edge(nodeA, nodeB):
                        total_weight += graph[nodeA][nodeB].get('weight', 1)
                        edge_count += 1
            
            # Standardize the weight by dividing by the number of nodes in clusterB
            if edge_count > 0:
                avg_weight = total_weight / edge_count
            else:
                avg_weight = 0
            
            if avg_weight > 0:
                # Add the edge with the calculated weight
                coarse_grained_graph.add_edge(str(i), str(j), weight=avg_weight)
    
    return coarse_grained_graph

def save_coarse_grained_graph(G, model, rst_index, nodes, filename):
    """
    Save the coarse-grained graph to a file.
    
    Args:
        graph (nx.DiGraph): The coarse-grained graph.
        filename (str): The name of the file to save the graph.
    """
    from collections import Counter
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    import textwrap

    fig, ax = plt.subplots(figsize=(32, 16))

    # ==================== 关键修改1：创建分层布局 ====================
    # 生成两个不同半径的圆形布局
    inner_radius = 0.6  # 内圈半径
    outer_radius = 1.0  # 外圈半径

    # 分离节点到不同层级
    inner_nodes = [n for n in G.nodes if int(n) < rst_index]  # 内圈节点（蓝色）
    outer_nodes = [n for n in G.nodes if int(n) >= rst_index]  # 外圈节点（红色）

    # 生成分层布局坐标
    pos = {}
    pos.update(nx.circular_layout(inner_nodes, scale=inner_radius))  # 内圈缩小
    pos.update(nx.circular_layout(outer_nodes, scale=outer_radius))  # 外圈保持原大小

    # ==================== 关键修改2：保持边的绘制逻辑 ====================
    edge_colors = ['gray' for _ in G.edges()]
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    # normalized_weights = [20*w/max(edge_weights) for w in edge_weights]
    normalized_weights = [w for w in edge_weights]

    nx.draw_networkx_edges(G, pos, 
                        width=normalized_weights,
                        edge_color=edge_colors,
                        arrows=True,
                        arrowsize=20)

    # ==================== 关键修改3：按分层绘制节点 ====================
    # 绘制内圈节点（skyblue）
    nx.draw_networkx_nodes(G, pos,
                        nodelist=inner_nodes,
                        node_color='pink',
                        node_size=1000)

    # 绘制外圈节点（pink）
    nx.draw_networkx_nodes(G, pos,
                        nodelist=outer_nodes,
                        node_color='skyblue',
                        node_size=1000)

    # ==================== 关键修改4：标签同步分层 ====================
    # 修改标签，使之包含inner_weight的值
    labels_with_weights = {node: f"{node}" if node in inner_nodes else node for node in G.nodes}

    nx.draw_networkx_labels(G, pos, labels=labels_with_weights, font_size=20)

    # ==================== 图例保持原有逻辑 ====================
    legend_elements = []
    # for i, pathways in enumerate(nodes):
    #     pathway_str = ", ".join(pathways)
    #     wrapped_pathway_str = textwrap.fill(pathway_str, width=100)
        
    #     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
    #                                      label=f'{i}: {wrapped_pathway_str}',
    #                                      markersize=10,
    #                                      markerfacecolor='skyblue' if i < rst_index else 'pink'))

    for i, group in enumerate(nodes):
        if i >= rst_index:
            pathway_str = ", ".join(group)
            wrapped_pathway_str = textwrap.fill(pathway_str, width=100)
            label_str = f'{i}: {wrapped_pathway_str}'
        else:
            subsystem_counter = Counter()
            for node in group:
                rxn_id = node[:-4]
                subsystem = model.reactions.get_by_id(rxn_id).subsystem or 'Unknown'
                subsystem_counter[subsystem] += 1

            subsystem_desc = ", ".join(f"{k}({v})" for k, v in subsystem_counter.most_common(5))

            label_str = f'{i} - {len(group)} nodes\n{subsystem_desc}'

        # 包装文字换行（适应图例宽度）
        wrapped_label_str = textwrap.fill(label_str, width=100)

        # 添加图例元素
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                    label=wrapped_label_str,
                    markersize=10,
                    markerfacecolor='skyblue' if i >= rst_index else 'pink')
        )
    ax.legend(handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def graph_node_atp_generation(model, G, met_flux, cluster_nodes):
    graph_atp_generation = {}

    for node_id in range(20):
        temp = node_atp_generation(model, G, met_flux, node_id, cluster_nodes)
        graph_atp_generation[node_id] = temp
    
    return graph_atp_generation

def node_atp_generation(model, G, met_flux, node_id, cluster_nodes):
        
    grouped_rec = [rec for rec in list(G.nodes) if rec in cluster_nodes[node_id]]
    grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in grouped_rec or rec+'_bwd' in grouped_rec}

    node_atp_generation = 0
    for rec, flux in grouped_flux.items():
        atp_generation = calculate_atp_generation(model, rec, flux)
        node_atp_generation += atp_generation

    return node_atp_generation

def calculate_atp_generation(model, rec_id, rec_flux):
        rec_temp = model.reactions.get_by_id(rec_id)
        keys = list(rec_temp.metabolites.keys())
        values = list(rec_temp.metabolites.values())

        if rec_flux == 0.0:
            return 0

        flux_generate = [[keys[i].id, values[i]] for i in range(len(keys)) if (values[i] > 0)] # 被指向
        flux_consume = [[keys[i].id, values[i]] for i in range(len(keys)) if (values[i] < 0)] # 指出去

        if any(item[0] == 'atp[c]' for item in flux_generate):
            w = next((item[1] for item in flux_generate if item[0] == 'atp[c]'), None)
        elif any(item[0] == 'atp[c]' for item in flux_consume):
            w = next((item[1] for item in flux_consume if item[0] == 'atp[c]'), None)
        else:
            w = 0

        return w * rec_flux

def node_proteome_cost(G, met_flux, node_id, cluster_nodes):
        
    grouped_rec = [rec for rec in list(G.nodes) if rec in cluster_nodes[node_id]]
    grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in grouped_rec or rec+'_bwd' in grouped_rec}

    node_proteome_cost = 0
    for rec, flux in grouped_flux.items():
        proteome_cost = abs(flux)
        node_proteome_cost += proteome_cost

    return node_proteome_cost

def graph_node_proteome_cost(G, met_flux, cluster_nodes):
    graph_proteome_cost = {}

    for node_id in range(20):
        temp = node_proteome_cost(G, met_flux, node_id, cluster_nodes)
        graph_proteome_cost[node_id] = temp

    return graph_proteome_cost



def graph_edge_proteome_cost(G, met_flux, cluster_nodes):
    edge_proteome_cost_matrix = np.zeros((20, 20))

    for i, j in [(i, j) for i in range(20) for j in range(20) if i!=j]:

        source_nodes = cluster_nodes[i]
        source_grouped_rec = [rec for rec in list(G.nodes) if rec in source_nodes]
        source_grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in source_grouped_rec or rec+'_bwd' in source_grouped_rec}

        target_nodes = cluster_nodes[j]
        target_grouped_rec = [rec for rec in list(G.nodes) if rec in target_nodes]
        target_grouped_flux = {rec: value for rec, value in met_flux if rec+'_fwd' in target_grouped_rec or rec+'_bwd' in target_grouped_rec}

        edge_proteome_cost = 0
        for u in source_nodes:
            if u not in G.nodes:
                continue

            for v in G.successors(u):
                if v in target_nodes:
                    temp = G[u][v].get('weight', 0)
                    edge_proteome_cost += temp
        edge_proteome_cost_matrix[i, j] = edge_proteome_cost

        
    return edge_proteome_cost_matrix