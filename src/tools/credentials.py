import os
from getpass import getpass

from quantuminspire.credentials import load_account, get_token_authentication, get_basic_authentication

QI_EMAIL = os.getenv('QI_EMAIL')
QI_PASSWORD = os.getenv('QI_PASSWORD')


def get_authentication():
    """ Gets the authentication for connecting to the Quantum Inspire API.

        First it tries to load a token, saved earlier (see README.md). When a token is not found it tries to login
        with basic authentication read from the environment variables QI_EMAIL and QI_PASSWORD. When the environment
        variables are not set, credentials are asked interactively.
    """
    token = load_account()
    if token is not None:
        return get_token_authentication(token)
    else:
        if QI_EMAIL is None or QI_PASSWORD is None:
            print('Enter email:')
            email = input()
            print('Enter password')
            password = getpass()
        else:
            email, password = QI_EMAIL, QI_PASSWORD
        return get_basic_authentication(email, password)
