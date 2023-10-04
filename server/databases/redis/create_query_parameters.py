from numpy import float64
from numpy.typing import NDArray


def create_query_parameters(embedding: NDArray[float64]) -> dict[str, bytes]:
    """
    Summary
    -------
    helper function for creating the query parameters for a Redis query

    Parameters
    ----------
    embedding (NDArray[float64]) : the embedding to use

    Returns
    -------
    query_parameters (dict[str, bytes]) : the query parameters
    """
    return  {
        'vec': embedding.tobytes()
    }
