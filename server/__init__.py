from importlib import import_module
from os import sep, walk
from os.path import join

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api import debug, v1
from server.config import Config
from server.lifespans import lifespans
from server.middlewares import LoggingMiddleware


class Framework(FastAPI):
    """
    Summary
    -------
    the FastAPI framework class

    Methods
    -------
    convert_delimiters(string: str, old: str, new: str) -> str
        convert delimiters in a string

    initialise_routes(api_directory: str)
        dynamically initialise all routes
    """
    def convert_delimiters(self, string: str, old: str, new: str) -> str:
        """
        Summary
        -------
        convert delimiters in a string

        Parameters
        ----------
        string (str) : the string to convert
        old (str) : the old delimiter
        new (str) : the new delimiter

        Returns
        -------
        string (str) : the converted string
        """
        return new.join(string.split(old))


    def initialise_routes(self, api_directory: str):
        """
        Summary
        -------
        initialise all routes

        Parameters
        ----------
        api_directory (str) : the directory where the API routes are located
        """
        module_file_names = [
            join(root, file)
            for root, _, files in walk(api_directory)
            for file in files
            if not file.startswith('_') and file.endswith('.py')
        ]

        module_names = [
            import_module(self.convert_delimiters(file_name[:-3], sep, '.')).__name__
            for file_name in module_file_names
        ]

        for module_name in module_names:
            print(f" * {self.convert_delimiters(module_name[len(api_directory):], '.', sep)} route found!")


def initialise() -> Framework:
    """
    Summary
    -------
    initialises everything

    Returns
    ------
    app (Framework) : an extended FastAPI instance
    """
    app = Framework(lifespan=lifespans, root_path=Config.server_root_path)
    app.initialise_routes(join('server', 'api'))
    app.include_router(v1)
    app.include_router(debug)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    return app
