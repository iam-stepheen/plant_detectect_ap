from app import app as application

# If you are using gunicorn or another WSGI server, you might not need this conditional
if __name__ == "__main__":
    application.run(debug=False)
