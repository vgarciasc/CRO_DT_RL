from rich.console import Console

console = Console()

def printv(str, verbose=False):
    if verbose:
        console.log(str)