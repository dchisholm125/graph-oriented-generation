import os
import networkx as nx

def seed_graph_from_prompt(graph, prompt):
    """Maps prompt keywords to actual nodes in the graph to establish entry points."""
    prompt_lower = prompt.lower()
    seeds = []
    
    for node in graph.nodes():
        node_name_lower = os.path.basename(node).lower()
        if "auth" in prompt_lower and "auth" in node_name_lower:
            seeds.append(node)
        elif "header" in prompt_lower and "header" in node_name_lower:
            seeds.append(node)
        elif "settings" in prompt_lower and "settings" in node_name_lower:
            seeds.append(node)
            
    return list(set(seeds))

def isolate_context(graph, prompt):
    """Traverses the graph starting from seed nodes to extract pristine context."""
    seeds = seed_graph_from_prompt(graph, prompt)
    
    if not seeds:
        return list(graph.nodes())
        
    subgraph_nodes = set()
    
    if len(seeds) == 1:
        # Easy Level: 1 Seed. Just return the seed and its immediate descendants.
        subgraph_nodes.add(seeds[0])
        # For localized search, we can just grab everything structurally tied to it.
        subgraph_nodes.update(nx.descendants(graph, seeds[0]))
    else:
        # Medium / Hard Level: We have multiple seeds. Find the shortest paths between them.
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                source, target = seeds[i], seeds[j]
                
                try: # source -> target
                    path = nx.shortest_path(graph, source=source, target=target)
                    subgraph_nodes.update(path)
                    subgraph_nodes.update(nx.descendants(graph, target))
                except nx.NetworkXNoPath:
                    pass
                
                try: # target -> source
                    path = nx.shortest_path(graph, source=target, target=source)
                    subgraph_nodes.update(path)
                    subgraph_nodes.update(nx.descendants(graph, source))
                except nx.NetworkXNoPath:
                    pass
                    
        # Fallback if no paths connect our seeds
        if not subgraph_nodes:
            for seed in seeds:
                subgraph_nodes.add(seed)
                subgraph_nodes.update(nx.descendants(graph, seed))


    return sorted(list(subgraph_nodes))

if __name__ == "__main__":
    # Test on the generated maze
    from srm_engine import ast_parser
    target = os.path.join(os.path.dirname(__file__), "../target_repo")
    if os.path.exists(target):
        G = ast_parser.build_graph(target)
        prompt = "Where is user auth state in HeaderWidget?"
        context_files = isolate_context(G, prompt)
        print(f"Isolated {len(context_files)} relevant files:")
        for file in context_files:
            print(f"  - {os.path.relpath(file, target)}")
