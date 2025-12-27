"""WSGI entrypoint for production hosts (e.g., PythonAnywhere).

PythonAnywhere WSGI config can use:
    from wsgi import application
"""

from app_flask import app

# WSGI servers (uWSGI/gunicorn/mod_wsgi) look for `application` by convention.
application = app
