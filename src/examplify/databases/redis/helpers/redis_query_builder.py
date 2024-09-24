from redis.commands.search.query import Query


def redis_query_builder(mapping: str, field: str, top_k: int) -> Query:
    """
    Summary
    -------
    helper function for creating a Redis query

    Parameters
    ----------
    mapping (str) : the mapping to use
    field (str) : the field to use
    top_k (int) : the number of results to return

    Returns
    -------
    query (Query) : the Redis query object
    """
    return (
        Query(f'(@{mapping}:{{ {field} }})=>[KNN {top_k} @vector $vec as score]')
        .sort_by('score')
        .return_fields('content', 'score')
        .paging(0, top_k)
        .dialect(2)
    )
