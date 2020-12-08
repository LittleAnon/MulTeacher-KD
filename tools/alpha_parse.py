import torch

def parse(alpha, k, PRIMITIVES = ['conv_3', 'conv_5', 'conv_7', 'skip_connect', 'self_att', 'rnn', 'none']):
    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:,:-1], 1)  # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))
    
        gene.append(node_gene)

    return gene

if __name__ == "__main__":
    a=[[0.0384,0.0267,0.0208,0.4942,0.0251,0.0934,0.3014], [0.0836,0.0442,0.0236,0.0497,0.0308,0.4857,0.2823], [0.0131,0.0103,0.0114,0.2121,0.0932,0.3776,0.2825], [0.0499,0.0199,0.0139,0.145,0.0677,0.2423,0.4613], [0.0117,0.0116,0.0096,0.183,0.1964,0.1843,0.4033], [0.0095,0.0098,0.0085,0.3246,0.1732,0.1591,0.3152]]



    import torch.nn as nn

    b = nn.ParameterList()
    b.append(nn.Parameter(torch.tensor([a[0]],)))
    b.append(nn.Parameter(torch.tensor([a[1], a[2]], )))
    b.append(nn.Parameter(torch.tensor([a[3], a[4], a[5]], )))

    print(parse(b, 1))
    
    # alpha_normal = nn.ParameterList()
    # for i in range(3):
    #     alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+1, 3)))
    # print(alpha_normal)