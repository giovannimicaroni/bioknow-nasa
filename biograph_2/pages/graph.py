import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import json
import pickle

# ==================== GRAPH INITIALIZATION ====================
@st.cache_resource
def load_knowledge_graph():
    """
    Initialize your knowledge graph here.
    Replace this sample data with your actual graph construction.
    """
    with open("assets/grafo_keywords.gpickle", "rb") as f:
        G = pickle.load(f)
    
    return G

# ==================== GRAPH UTILITIES ====================
def get_node_types(G):
    """Get all unique node types in the graph"""
    types = set()
    for node in G.nodes():
        if 'type' in G.nodes[node]:
            types.add(G.nodes[node]['type'])
    return sorted(list(types))

def get_relationship_types(G):
    """Get all unique relationship types in the graph"""
    rels = set()
    for u, v, data in G.edges(data=True):
        if 'relationship' in data:
            rels.add(data['relationship'])
    return sorted(list(rels))

def filter_by_node_type(G, node_type):
    """Extract subgraph containing only nodes of specified type and their connections"""
    nodes = [n for n in G.nodes() if G.nodes[n].get('type') == node_type]
    return G.subgraph(nodes + list(set([neighbor for node in nodes for neighbor in G.neighbors(node)])))

def filter_by_relationship_type(G, rel_type):
    """Extract subgraph containing only edges of specified relationship type"""
    edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == rel_type]
    nodes = set([u for u, v in edges] + [v for u, v in edges])
    return G.subgraph(nodes)

def get_node_neighborhood(G, node, depth=1):
    """Get k-hop neighborhood of a node"""
    if node not in G:
        return nx.Graph()
    
    nodes = {node}
    current_layer = {node}
    
    for _ in range(depth):
        next_layer = set()
        for n in current_layer:
            next_layer.update(G.neighbors(n))
        nodes.update(next_layer)
        current_layer = next_layer
    
    return G.subgraph(nodes)

def get_shortest_path_subgraph(G, source, target):
    """Get subgraph containing shortest path between two nodes"""
    try:
        path = nx.shortest_path(G, source, target)
        return G.subgraph(path)
    except nx.NetworkXNoPath:
        return nx.Graph()

# ==================== VISUALIZATION ====================
def visualize_graph(G, title="Knowledge Graph"):
    """Create Network Science style visualization"""
    if len(G.nodes()) == 0:
        return None
    
    # Create PyVis network with clean styling
    net = Network(
        height="650px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50"
    )
    
    # Configure physics for organic layout
    net.barnes_hut(
        gravity=-200,
        central_gravity=0,
        spring_length=150,
        spring_strength=0,
        damping=0
    )
    
    # Color scheme inspired by Network Science book
    type_colors = {
        'Person': '#3498db',      # Blue
        'Company': '#e74c3c',     # Red
        'Technology': '#2ecc71',  # Green
        'Project': '#f39c12',     # Orange
        'Location': '#9b59b6',    # Purple
        'Skill': '#1abc9c',       # Turquoise
    }
    
    # Add nodes with styling
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Create label
        label = str(node)
        
        # Create hover tooltip with all attributes
        title_parts = [f"<b>{label}</b>", f"Type: {node_type}"]
        for key, value in node_data.items():
            if key != 'type':
                title_parts.append(f"{key}: {value}")
        title = "<br>".join(title_parts)
        
        # Get color
        color = type_colors.get(node_type, '#95a5a6')
        
        # Calculate node size based on degree
        degree = G.degree(node)
        size = 15 + (degree * 3)
        
        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=2,
            borderWidthSelected=4,
            font={'size': 14, 'color': '#2c3e50', 'face': 'arial'}
        )
    
    # Add edges with clean styling
    for source, target, data in G.edges(data=True):
        relationship = data.get('relationship', '')
        
        net.add_edge(
            source,
            target,
            title=relationship,
            label=relationship,
            color={'color': '#95a5a6', 'highlight': '#2c3e50'},
            width=2,
            font={'size': 11, 'color': '#7f8c8d', 'align': 'middle'},
            smooth={'type': 'continuous'}
        )
    
    # Set options for better interactivity
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 1
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        },
        "nodes": {
            "font": {
                "size": 14,
                "face": "arial"
            },
            "borderWidth": 2,
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.1)",
                "size": 5,
                "x": 2,
                "y": 2
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "font": {
                "size": 11,
                "align": "middle"
            }
        }
    }
    """)
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as file:
            html_content = file.read()
        os.unlink(f.name)
    
    return html_content

# ==================== MAIN APPLICATION ====================
def main():
    st.set_page_config(
        page_title="Knowledge Graph Explorer",
        layout="wide",
        page_icon="🕸️"
    )
    
    # Custom CSS for clean design
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: white;
            border-radius: 5px;
            padding: 0 20px;
            border: 1px solid #dee2e6;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #3498db;
            color: white;
        }
        h1 {
            color: #2c3e50;
            font-family: 'Arial', sans-serif;
        }
        .metric-card {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load graph
    G = load_knowledge_graph()
    
    # Header
    st.title("🕸️ Knowledge Graph Explorer")
    st.markdown("Explore relationships and patterns in the knowledge network")
    
    # Sidebar with graph statistics and legend
    with st.sidebar:
        st.header("📊 Graph Statistics")
        
        col1, col2 = st.columns(2)
        col1.metric("Nodes", len(G.nodes()))
        col2.metric("Edges", len(G.edges()))
        
        # Calculate additional metrics
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
        density = nx.density(G)
        
        col1.metric("Avg Degree", f"{avg_degree:.1f}")
        col2.metric("Density", f"{density:.3f}")
        
        st.markdown("---")
        
        st.header("🎨 Node Types")
        node_types = get_node_types(G)
        
        type_colors = {
            'Person': '#3498db',
            'Company': '#e74c3c',
            'Technology': '#2ecc71',
            'Project': '#f39c12',
        }
        
        for node_type in node_types:
            count = len([n for n in G.nodes() if G.nodes[n].get('type') == node_type])
            color = type_colors.get(node_type, '#95a5a6')
            st.markdown(
                f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                f'<div style="width: 20px; height: 20px; background-color: {color}; '
                f'border-radius: 50%; margin-right: 10px; border: 2px solid #2c3e50;"></div>'
                f'<span><b>{node_type}</b> ({count})</span></div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        st.header("🔗 Relationships")
        rel_types = get_relationship_types(G)
        for rel in rel_types:
            count = len([(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == rel])
            st.markdown(f"• **{rel}** ({count})")
        
        st.markdown("---")
        st.caption("💡 Hover over nodes and edges for details")
        st.caption("🖱️ Drag to pan, scroll to zoom")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Full Graph",
        "🔍 Node Explorer",
        "🎯 Path Finder",
        "🏷️ Filter by Type",
        "📊 Graph Stats"
    ])
    
    # ==================== TAB 1: FULL GRAPH ====================
    with tab1:
        st.header("Complete Knowledge Graph")
        
        with st.spinner("Rendering graph..."):
            html = visualize_graph(G, "Full Knowledge Graph")
            if html:
                components.html(html, height=670)
            else:
                st.warning("No data to display")
    
    # ==================== TAB 2: NODE EXPLORER ====================
    with tab2:
        st.header("Explore Node Neighborhood")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            node_list = sorted(list(G.nodes()))
            selected_node = st.selectbox("Select a node", node_list)
        
        with col2:
            depth = st.selectbox("Neighborhood depth", [1, 2, 3], index=0)
        
        if st.button("🔍 Explore", use_container_width=True):
            subgraph = get_node_neighborhood(G, selected_node, depth)
            
            if len(subgraph.nodes()) > 0:
                st.info(f"**{selected_node}** has {G.degree(selected_node)} direct connection(s)")
                st.info(f"Showing {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges within {depth} hop(s)")
                
                html = visualize_graph(subgraph, f"Neighborhood of {selected_node}")
                if html:
                    components.html(html, height=670)
            else:
                st.warning(f"Node '{selected_node}' not found in graph")
    
    # ==================== TAB 3: PATH FINDER ====================
    with tab3:
        st.header("Find Shortest Path")
        
        node_list = sorted(list(G.nodes()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_node = st.selectbox("From", node_list, key="source")
        
        with col2:
            target_node = st.selectbox("To", node_list, key="target", index=min(1, len(node_list)-1))
        
        if st.button("🎯 Find Path", use_container_width=True):
            if source_node == target_node:
                st.warning("Please select different nodes")
            else:
                try:
                    path = nx.shortest_path(G, source_node, target_node)
                    subgraph = get_shortest_path_subgraph(G, source_node, target_node)
                    
                    if len(subgraph.nodes()) > 0:
                        st.success(f"✅ Path found with {len(path)-1} step(s)")
                        
                        # Display path
                        path_str = " → ".join(path)
                        st.info(f"**Path:** {path_str}")
                        
                        html = visualize_graph(subgraph, f"Path from {source_node} to {target_node}")
                        if html:
                            components.html(html, height=670)
                    else:
                        st.warning(f"No path exists between {source_node} and {target_node}")
                except nx.NetworkXNoPath:
                    st.warning(f"No path exists between {source_node} and {target_node}")
    
    # ==================== TAB 4: FILTER BY TYPE ====================
    with tab4:
        st.header("Filter Graph by Type")
        
        filter_mode = st.radio(
            "Filter by",
            ["Node Type", "Relationship Type"],
            horizontal=True
        )
        
        if filter_mode == "Node Type":
            node_types = get_node_types(G)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_type = st.selectbox("Select node type", node_types)
            
            with col2:
                st.write("")
                st.write("")
                filter_btn = st.button("📊 Apply Filter", use_container_width=True)
            
            if filter_btn:
                subgraph = filter_by_node_type(G, selected_type)
                
                if len(subgraph.nodes()) > 0:
                    type_nodes = [n for n in subgraph.nodes() if G.nodes[n].get('type') == selected_type]
                    st.info(f"Showing {len(type_nodes)} **{selected_type}** nodes and their {len(subgraph.nodes()) - len(type_nodes)} connected nodes")
                    
                    html = visualize_graph(subgraph, f"{selected_type} Subgraph")
                    if html:
                        components.html(html, height=670)
                else:
                    st.warning(f"No nodes of type '{selected_type}' found")
        
        else:  # Relationship Type
            rel_types = get_relationship_types(G)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_rel = st.selectbox("Select relationship type", rel_types)
            
            with col2:
                st.write("")
                st.write("")
                filter_btn = st.button("📊 Apply Filter", use_container_width=True)
            
            if filter_btn:
                subgraph = filter_by_relationship_type(G, selected_rel)
                
                if len(subgraph.nodes()) > 0:
                    st.info(f"Showing {len(subgraph.edges())} **{selected_rel}** relationships")
                    
                    html = visualize_graph(subgraph, f"{selected_rel} Relationships")
                    if html:
                        components.html(html, height=670)
                else:
                    st.warning(f"No relationships of type '{selected_rel}' found")
    
    # ==================== TAB 5: STATISTICS ====================
    with tab5:
        st.header("Graph Analysis & Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Metrics")
            st.metric("Total Nodes", len(G.nodes()))
            st.metric("Total Edges", len(G.edges()))
            st.metric("Graph Density", f"{nx.density(G):.4f}")
            
            if nx.is_connected(G):
                st.metric("Average Path Length", f"{nx.average_shortest_path_length(G):.2f}")
            else:
                st.metric("Connected Components", nx.number_connected_components(G))
        
        with col2:
            st.subheader("Degree Statistics")
            degrees = dict(G.degree())
            st.metric("Average Degree", f"{sum(degrees.values()) / len(degrees):.2f}")
            st.metric("Max Degree", max(degrees.values()))
            st.metric("Min Degree", min(degrees.values()))
        
        with col3:
            st.subheader("Top Connected Nodes")
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, degree in top_nodes:
                node_type = G.nodes[node].get('type', 'Unknown')
                st.write(f"**{node}** ({node_type}): {degree} connections")
        
        st.markdown("---")
        
        # Degree distribution
        st.subheader("Degree Distribution")
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(degree_sequence)), degree_sequence, color='#3498db', alpha=0.7)
        ax.set_xlabel("Node Index (sorted by degree)")
        ax.set_ylabel("Degree")
        ax.set_title("Node Degree Distribution")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

if __name__ == "__main__":
    main()