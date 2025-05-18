# Logging

### Info
- Logging in this project is currently handled using loguru.
- This is defined in the Utils/Utils.py file.
- Every logging instance in this project should import and use the `logger` instance from Utils/Utils.py.
- This is to ensure that all logging is handled in a consistent manner.
- It is also done this way so that if for whatever reason you would prefer to use the built-in logging module, you can do so by simply removing the `logger` instance from Utils/Utils.py and replacing it with a standard logging configuration.
- It also makes it easier to drop-in the standard logging lib if you're using one of the libraries in this project in your own project.