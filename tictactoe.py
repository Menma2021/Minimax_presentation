import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import shutil

class TicTacToeVisualizer:
    def __init__(self, m=3, n=3, k=3, depth=3, pruning=False, predefined_board=None):
        """Initializing Tic-Tac-Toe"""
        self.m, self.n, self.k = m, n, k
        self.depth = depth
        self.pruning = pruning
        self.tree = nx.DiGraph()
        self.node_counter = 0

        if predefined_board is not None:
            self.board = np.array(predefined_board)  # Using user's board
        else:
            self.board = np.full((m, n), ".", dtype=str)  # Creating empty board

        # Preventing too deep visualization
        self.allow_board_images = depth <= 4

    def minimax(self, board, depth, maximizing, alpha=float('-inf'), beta=float('inf'), parent=None):
        """Recursive minimax with alpha-beta pruning"""
        node_id = self.node_counter
        self.node_counter += 1
        self.tree.add_node(node_id, board=board.copy(), value=None)

        if parent is not None:
            self.tree.add_edge(parent, node_id, alpha_beta=(alpha, beta))

        board_eval = self.evaluate_board(board)

        # Terminal state check
        if depth == 0 or board_eval != 0 or "." not in board:
            self.tree.nodes[node_id]['value'] = board_eval
            return board_eval
        
        # If current player is maximizer
        if maximizing:
            max_eval = float('-inf')
            for new_board in self.generate_moves(board, "X"):
                eval = self.minimax(new_board, depth - 1, False, alpha, beta, node_id)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                
                # Alpha-beta pruning
                if self.pruning and beta <= alpha:
                        break
            self.tree.nodes[node_id]['value'] = max_eval

        # If current player is minimizer
        else:
            min_eval = float('inf')
            for new_board in self.generate_moves(board, "O"):
                eval = self.minimax(new_board, depth - 1, True, alpha, beta, node_id)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                # Alpha-beta pruning
                if self.pruning and beta <= alpha:
                    break
            self.tree.nodes[node_id]['value'] = min_eval
        
        if parent is not None:
            self.tree[parent][node_id]['alpha_beta'] = (alpha, beta)

        return max_eval if maximizing else min_eval

    def evaluate_board(self, board):
        """Evaluating board state with heuristic"""
        def check_win(player):
            for i in range(self.m):
                for j in range(self.n):
                    if (j + self.k <= self.n and all(board[i, j + d] == player for d in range(self.k))) or \
                       (i + self.k <= self.m and all(board[i + d, j] == player for d in range(self.k))) or \
                       (i + self.k <= self.m and j + self.k <= self.n and all(board[i + d, j + d] == player for d in range(self.k))) or \
                       (i - self.k + 1 >= 0 and j + self.k <= self.n and all(board[i - d, j + d] == player for d in range(self.k))):
                        return True
            return False
        
        if check_win("X"): return 10
        if check_win("O"): return -10
        return 0  # Neutral if no win
    
    def generate_moves(self, board, player):
        """Generating all possible next moves"""
        new_boards = []
        for i in range(self.m):
            for j in range(self.n):
                if board[i, j] == ".":
                    new_board = board.copy()
                    new_board[i, j] = player
                    new_boards.append(new_board)
        return new_boards
    
    def generate_moves_priority(self, board, player):
        """Generating all possible next moves (with prioritizing beneficial boards first))"""
        move_scores = []

        for i in range(self.m):
            for j in range(self.n):
                if board[i, j] == ".":
                    new_board = board.copy()
                    new_board[i, j] = player
                    score = self.evaluate_board(new_board)  # Checking board value after move
                    move_scores.append((score, new_board))

        # Sorting moves by score, prioritizing either +10 if maximizer or -10 boards if minimizer
        if player == "X":
            move_scores.sort(reverse=True, key=lambda x: (x[0] == 10, x[0] == 0, x[0] == -10))
        else:  # minimizer
            move_scores.sort(reverse=True, key=lambda x: (x[0] == -10, x[0] == 0, x[0] == 10))

        return [move for _, move in move_scores]


    def build_tree(self, maximizing = True):
        """Building the decision tree from initial state"""
        self.minimax(self.board, self.depth, maximizing)

    def draw_board(self, board, filename):
        """Generating an image of the board"""
        
        if not self.allow_board_images:
            return
        
        size = 100  # Cell size
        img_size = (self.n * size, self.m * size)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()

        # Grid
        for i in range(1, self.m):
            draw.line([(0, i * size), (img_size[0], i * size)], fill="black", width=3)
        for j in range(1, self.n):
            draw.line([(j * size, 0), (j * size, img_size[1])], fill="black", width=3)

        # Xs and Os
        for i in range(self.m):
            for j in range(self.n):
                if board[i, j] != ".":
                    text = board[i, j]
                    bbox = draw.textbbox((0, 0), text, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((j * size + (size - w) // 2, i * size + (size - h) // 2), text, fill="black", font=font)

        img.save(filename)

    def visualize_tree(self, filename="decision_tree.png"):
        """Visualizing the decision tree"""
        pos = nx.drawing.nx_pydot.pydot_layout(self.tree, prog="twopi")  # Layout can be changed to dot (not recommended for big values)

        node_count = len(self.tree.nodes)
        if node_count > 120:  # Adaptable size and labels
            node_size = 30
            font_size = 6
            if node_count > 250: 
                labels = {node: "" for node in self.tree.nodes}
            else:
                labels = {node: str(self.tree.nodes[node]['value']) for node in self.tree.nodes}
        else:
            node_size = 300
            font_size = 10
            labels = {node: str(self.tree.nodes[node]['value']) for node in self.tree.nodes}

        plt.figure(figsize=(14, 10))
        nx.draw(self.tree, pos, with_labels=False, node_color="lightblue", edge_color="gray", 
                node_size=node_size, font_size=font_size)
        nx.draw_networkx_labels(self.tree, pos, labels=labels, font_size=font_size, font_color="black")

        plt.title("Decision Tree")
        plt.savefig(filename, dpi = 300)
        plt.show()


    def visualize_boards_tree(self, output_filename="board_tree.png"):
        """Visualizing the decision tree with board images instead of nodes. Ended up being irrelevant
        with the next function, kept for reference"""
        
        if len(self.tree.nodes) > 200:
            print("Too many nodes to visualize boards")
            return
        
        img_folder = "board_images"
        if os.path.exists(img_folder):
            shutil.rmtree(img_folder)
        os.makedirs(img_folder, exist_ok=True)

        # Generating and saving board images
        img_paths = {}
        for node in self.tree.nodes:
            filename = f"{img_folder}/node_{node}.png"
            self.draw_board(self.tree.nodes[node]['board'], filename)
            img_paths[node] = filename

        # Graph with GraphViz
        A = pgv.AGraph(strict=True, directed=True)

        max_node_size = 0.8
        dpi = 150

        for node, img_path in img_paths.items():
            img = Image.open(img_path)
            width, height = img.size

            min_size = 200  # The smallest size
            if width < min_size or height < min_size:
                scale_factor = min_size / min(width, height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                img = img.resize(new_size, Image.LANCZOS)
                img.save(img_path)

            # Adding node to the graph
            A.add_node(
                node,
                shape="none",  # Doesn't matter
                image=img_path,  # Attaching board to the node
                label="",  # Doesn't matter
                width=str(max_node_size),  # Scaling
                height=str(max_node_size),
                fixedsize="true"  # Fixed size for nodes
            )

        # Edges with alphs-beta values
        for u, v, data in self.tree.edges(data=True):
            alpha, beta = data.get("alpha_beta", ("-", "-"))
            A.add_edge(u, v, label=f"α={alpha}, β={beta}", fontsize = "12pt")

        A.graph_attr.update(dpi=str(dpi))
        A.layout(prog="twopi", args="-Granksep=2")  # Can be changed to dot for small values
        A.draw(output_filename, format="png")

        print(f"Board tree visualization saved as {output_filename}")


    def draw_board_with_value(self, board, filename, value):
        """Generating image of the board with manimax values"""
        
        size = 100
        img_size = (self.n * size, self.m * size)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 40)
            value_font = ImageFont.truetype("arial.ttf", 50)
        except:
            font = ImageFont.load_default()
            value_font = ImageFont.load_default()

        # Grid
        for i in range(1, self.m):
            draw.line([(0, i * size), (img_size[0], i * size)], fill="black", width=3)
        for j in range(1, self.n):
            draw.line([(j * size, 0), (j * size, img_size[1])], fill="black", width=3)

        # Xs and Os
        for i in range(self.m):
            for j in range(self.n):
                if board[i, j] != ".":
                    text = board[i, j]
                    bbox = draw.textbbox((0, 0), text, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((j * size + (size - w) // 2, i * size + (size - h) // 2), text, fill="black", font=font)

        # Minimax evaluation at the center
        value_text = str(value)
        bbox = draw.textbbox((0, 0), value_text, font=value_font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((img_size[0] - w) // 2, (img_size[1] - h) // 2), value_text, fill="red", font=value_font)

        img.save(filename)

    def get_best_move_chain(self):
        """Sequence of best moves by tracing the minimax evaluation (works properly only for maximizer)"""

        if not self.tree.nodes:
            return []

        # Root node
        root_node = None
        for node in self.tree.nodes:
            if self.tree.in_degree(node) == 0:  # No incoming edges
                root_node = node
                break

        if root_node is None:
            return []

        best_chain = []
        current_node = root_node

        while current_node in self.tree.nodes:
            best_child = None
            best_value = float('-inf') if self.tree.nodes[current_node].get("is_maximizing", True) else float('inf')

            for neighbor in self.tree.neighbors(current_node):  # Children
                node_value = self.tree.nodes[neighbor].get("value", None)

                if node_value is not None:
                    if self.tree.nodes[current_node].get("is_maximizing", True):
                        if node_value > best_value:
                            best_value = node_value
                            best_child = neighbor
                    else:
                        if node_value < best_value:
                            best_value = node_value
                            best_child = neighbor

            if best_child:
                best_chain.append((current_node, best_child))
                current_node = best_child
            else:
                break  # No more best moves

        return best_chain

    def visualize_boards_with_values(self, output_filename="board_values_tree.png"):
        """Visualizing the decision tree with board images instead of nodes, with options for alpha-beta pruning and minimax values"""

        if len(self.tree.nodes) > 200:
            print("Too many nodes to visualize boards clearly. Skipping board tree visualization.")
            return

        img_folder = "board_images"
        if os.path.exists(img_folder):
            shutil.rmtree(img_folder)
        os.makedirs(img_folder, exist_ok=True)

        img_paths = {}
        node_values = {}

        # Generating and saving board images
        for node in self.tree.nodes:
            board = self.tree.nodes[node]['board']
            filename = f"{img_folder}/node_{node}.png"
            value = self.tree.nodes[node].get('value', '?')  # Geting node value
            #self.draw_board_with_value(board, filename, value)  # Drawing boards with minimax values
            self.draw_board(board, filename)  # Drawing standard boards
            img_paths[node] = filename
            node_values[node] = value  # Values for labeling

        # Graph with GraphViz
        A = pgv.AGraph(strict=True, directed=True)

        max_node_size = 0.8
        dpi = 300

        # Optimal path
        best_move_chain = self.get_best_move_chain()

        for node, img_path in img_paths.items():
            img = Image.open(img_path)
            width, height = img.size
            min_size = 300  # The smallest size
            if width < min_size or height < min_size:
                scale_factor = min_size / min(width, height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                img = img.resize(new_size, Image.LANCZOS)
                img.save(img_path)

            # Adding node to the graph
            A.add_node(
                node,
                shape="none",  # Doesn't matter
                image=img_path,  # Attaching board to the node
                label="",  # Doesn't matter
                width=str(max_node_size),  # Scaling
                height=str(max_node_size),
                fixedsize="true"  # Fixed size for nodes
            )

        # Add edges and highlight the best move chain
        for u, v, data in self.tree.edges(data=True):
            alpha, beta = data.get("alpha_beta", ("-", "-"))
            edge_color = "black"  # Default

            if (u, v) in best_move_chain:
                edge_color = "red"  # Highlighting best move chain

            A.add_edge(u, v, label=f"α={alpha}, β={beta}", fontsize="16pt", color=edge_color, penwidth="2.5", fontcolor = "#d07d2f")  # alpha-beta pruning values
            #A.add_edge(u, v, color=edge_color, penwidth="2.5")  # Standard edges

        A.graph_attr.update(dpi=str(dpi))
        A.layout(prog="dot", args="-Granksep=0.8 -Gnodesep=1.3 -Gsplines=false")  # Hierarchical layout (tree-like)
        #A.layout(prog="twopi", args="-Granksep=2")  # Radial layout (circle-like)
        A.draw(output_filename, format="png")

        print(f"Board values visualization saved as {output_filename}")


# Execurion
visualizer = TicTacToeVisualizer(m=3, n=3, k=3, depth=3, pruning=True, predefined_board = [
    ["X", "O", "X"],
    ["O", ".", "X"],
    [".", "O", "."]
])

"""visualizer = TicTacToeVisualizer(m=15, n=15, k=4, depth=1, pruning=False, predefined_board = [
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "O", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "X", "O", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", "X", "X", "X", "X", "O", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "O", "X", "O", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."]
    ])"""

visualizer.build_tree(True)  # Can change maximizing to false for first move by minimizer

# Basic decision tree
visualizer.visualize_tree()

# Board decision tree visualization 
#visualizer.visualize_boards_tree()

# Board decision tree visualization with values
visualizer.visualize_boards_with_values()
