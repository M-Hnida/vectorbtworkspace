# Add this at the end of plotter.py after all other functions

def plot_path_mc_results(mc_results: Dict[str, Any]) -> None:
    """
    Public API for plotting path randomization Monte Carlo results.
    Wrapper for backward compatibility with existing test code.
    
    Args:
        mc_results: Results dict from monte_carlo_path.run_path_randomization_mc()
    """
    fig = _plot_path_mc_results(mc_results)
    fig.show()
