from typing import List


class Stream:

    verbose = True

    @staticmethod
    def write(*msg: List[str], verbose: bool = None) -> None:
        if verbose is not None:
            if verbose:
                print(*msg)
        else:
            if Stream.verbose:
                print(*msg)

stream = Stream