import numpy as np
from scipy.integrate import solve_ivp

def cr3bp_eom(t, state, mu):
    """
    Equations of motion for the Circular Restricted Three-Body Problem in the 
    rotating frame.
    
    The rotating frame rotates with angular velocity ω = 1 (normalized) such that
    the two primaries remain fixed on the x-axis:
    - Primary 1 (larger, mass = 1-μ) at position (-μ, 0, 0)
    - Primary 2 (smaller, mass = μ) at position (1-μ, 0, 0)
    
    Inputs
    t : float
        Time (not used explicitly, system is autonomous in rotating frame)
    state : array_like, shape (6,)
        State vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter μ = m2/(m1+m2)
    
    Outputs
    dstate : ndarray, shape (6,)
        Time derivative of state [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    
    # Distances to primaries
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    
    # Accelerations in rotating frame
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-(1-mu))/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3
    
    return np.array([vx, vy, vz, ax, ay, az])


def solve_CR3BP(state0, t_span, mu, t_eval=None, rtol=1e-12, atol=1e-12, 
                dense_output=False, events=None):
    """
    Solve the EoM using DOP853.
    
    Inputs
    state0 : array_like, shape (6,)
        Initial state vector [x, y, z, vx, vy, vz] in normalized coordinates
    t_span : tuple
        Integration time span (t0, tf) in normalized time units
    mu : float
        Mass parameter (mu = m2/(m1+m2))

    Optional Inputs    
    t_eval : array_like, optional
        Times at which to store the solution. If None, solver chooses times.
    rtol : float, optional
        Relative tolerance for integration (default: 1e-12)
    atol : float, optional
        Absolute tolerance for integration (default: 1e-12)
    dense_output : bool, optional
        Whether to compute a continuous solution (default: False)
    events : callable or list of callables, optional
        Event functions for detection during integration
    
    Outputs
    states : ndarray, shape (6, N)
        State vectors at each time point
    times : ndarray, shape (N,)
        Time points corresponding to each state
    sol : OdeResult
        Full solution object from solve_ivp (includes dense_output if requested)
    """
    sol = solve_ivp(
        fun=lambda t, y: cr3bp_eom(t, y, mu),
        t_span=t_span,
        y0=state0,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        events=events
    )
    
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    
    return sol.y, sol.t, sol