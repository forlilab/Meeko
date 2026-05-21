"""Polymer-specific exceptions."""

from sys import exc_info
import traceback

eol = "\n"


class PolymerCreationError(RuntimeError):
    def __init__(self, error: str, recommendations: str = None):
        super().__init__(error)
        self.error = error
        self.recommendations = recommendations
        exc_type, exc_value, exc_traceback = exc_info()
        if exc_value is not None:
            self.traceback = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
        else:
            self.traceback = None

    def __str__(self):
        msg = "" + eol
        msg += "Error: Creation of data structure for receptor failed." + eol
        msg += "" + eol
        msg += "Details:" + eol
        msg += self.error + eol

        if self.traceback:
            msg += self.traceback + eol

        if self.recommendations:
            msg += "Recommendations:" + eol
            msg += self.recommendations + eol
            msg += "" + eol

        return msg
