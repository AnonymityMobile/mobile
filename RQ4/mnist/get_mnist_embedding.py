import networkx as nx
import pickle
from GraphEmbedding.ge.models.node2vec import Node2Vec
from GraphEmbedding.ge.models.deepwalk import DeepWalk
from GraphEmbedding.ge.models.line import LINE
from GraphEmbedding.ge.models.sdne import SDNE
from GraphEmbedding.ge.models.struc2vec import Struc2Vec


def run_node2vec():
    save_embedding = 'mnist_lenet5_tflite_node2vec.pkl'
    tree_path = 'mnist_lenet5_tflite_tree.txt'
    G = nx.read_edgelist(tree_path, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)
    model.train(window_size=5, iter=300, embed_size=128)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


def run_line():
    save_embedding = 'mnist_lenet5_tflite_line.pkl'
    tree_path = 'mnist_lenet5_tflite_tree.txt'
    G = nx.read_edgelist(tree_path, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=300, verbose=2)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


def run_deepwalk():
    save_embedding = 'mnist_lenet5_tflite_deepwalk.pkl'
    tree_path = 'mnist_lenet5_tflite_tree.txt'
    G = nx.read_edgelist(tree_path, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=16)
    model.train(window_size=5, iter=300, embed_size=128,)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


def run_struc2vec():
    save_embedding = 'mnist_lenet5_tflite_struc2vec.pkl'
    tree_path = 'mnist_lenet5_tflite_tree.txt'
    G = nx.read_edgelist(tree_path, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Struc2Vec(G, 10, 80, workers=16, verbose=40, )
    model.train(window_size=5, iter=200, embed_size=128)
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open(save_embedding, 'wb'))


run_node2vec()
run_line()
run_deepwalk()
run_struc2vec()


