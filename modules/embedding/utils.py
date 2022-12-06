from .none import NoneEmbedding
from .periodic import PeriodicEmbedding
from .simple import SimpleEmbedding


embedding_name_to_class = {
    'none': NoneEmbedding,
    'simple': SimpleEmbedding,
    'periodic': PeriodicEmbedding,
    # 'some_other_embedding_name': some_other_embedding_class,
}


def get_embedding_class(embedding_name):
    return embedding_name_to_class[embedding_name]


def get_embedding_instace(embedding_config):
    embedding_class = get_embedding_class(embedding_config.embedding_name)
    return embedding_class(embedding_config)
