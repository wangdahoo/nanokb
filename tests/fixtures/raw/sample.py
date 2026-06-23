"""Sample Python source for CodeTrack integration tests (Feature s1-feat-010).

This fixture models a tiny calculator module: a top-level function that calls
another, and a class with a method that calls a helper. CodeTrack must extract
``defines`` / ``calls`` / ``contains`` relations from it deterministically.
"""


def greet(name):
    return format_message(name)


def format_message(value):
    return "hello " + value


def helper(x, y):
    return x + y


class Calculator:
    def add(self, x, y):
        return helper(x, y)

    def sub(self, x, y):
        return helper(x, -y)


result = greet("world")
