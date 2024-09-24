from server.features.extraction import extract_text_from_image


def file_to_text(file: bytes, file_name: str) -> str:
    """
    Summary
    -------
    extract the text from a file

    Parameters
    ----------
    file (bytes): the file
    file_name (str): the name of the file

    Returns
    -------
    text (str): the text in the file
    """
    file_name_lowered = file_name.lower()

    if file_name_lowered.endswith('.pdf'):
        raise NotImplementedError

    elif file_name_lowered.endswith(
        (
            '.blp',
            '.bmp',
            '.dds',
            '.dib',
            '.eps',
            '.gif',
            '.icns',
            '.ico',
            '.im',
            '.jpeg',
            '.jpg',
            '.msp',
            '.pcx',
            '.pfm',
            '.png',
            '.ppm',
            '.sgi',
            '.spider',
            '.tga',
            '.tiff',
            '.webp',
            '.xbm',
            '.cur',
            '.dcx',
            '.fits',
            '.fli',
            '.flc',
            '.fpx',
            '.ftex',
            '.gbr',
            '.gd',
            '.imt',
            '.iptc',
            '.naa',
            '.mcidas',
            '.mic',
            '.mpo',
            '.pcd',
            '.pixar',
            '.psd',
            '.qoi',
            '.sun',
            '.wal',
            '.wmf',
            '.emf',
            '.xpm',
        )
    ):
        return extract_text_from_image(file)

    else:
        raise NotImplementedError
