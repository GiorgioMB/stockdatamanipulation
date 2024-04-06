class MethodError(Exception):
    """
    Internal class to raise an error when an invalid method is passed, for internal use only (or not, but I don't see the point)
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
