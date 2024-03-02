from uvicorn import run

from server import initialise
from server.config import Config


def main():
    """
    Summary
    -------
    programmatically run the server with Uvicorn
    """
    run(
        initialise(),
        host="0.0.0.0",
        port=Config.server_port,
        loop='uvloop',
        use_colors=True,
    )


if __name__ == '__main__':
    main()
