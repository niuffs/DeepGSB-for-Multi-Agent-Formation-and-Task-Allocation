"""Multi-agent metrics for Gauss2Gauss_Obs3 (xy-plane only).

Collision test:  d_ij < 2r  or  d_i < r
    d_ij  centre distance between agents i, j (xy)
    d_i   distance from agent i to the nearest obstacle surface
    r     agent radius

Simulation: an agent leaves the sim as soon as it collides (destroyed) or enters
the target eps-ball (arrived); a removed agent cannot take part in any later
collision.  All three models are open-loop in x (control depends on (x,t) only),
so removal never changes another agent's path -- the counts are exact.
"""
import numpy as np
from scipy.spatial import cKDTree

DT, STEPS = 0.01, 100
TGT = np.array([4.0, 4.0])                                    # target mean, xy
OBS = [((2.0, 2.0), 1.0), ((-3.0, 1.0), 0.5), ((0.0, -2.0), 1.5)]   # (centre, radius)
R, EPS = 0.002, 0.5                                           # agent radius / target ball


def obs_dist(z):
    """Signed distance to the nearest obstacle surface (negative = inside)."""
    return np.min([np.linalg.norm(z[:, :2] - np.array(c), axis=1) - rr for c, rr in OBS], axis=0)


def nn_field(t2):
    """Nearest-neighbour distance of every agent at every step. (N, T+1)"""
    return np.stack([cKDTree(t2[:, k, :2]).query(t2[:, k, :2], k=2)[0][:, 1]
                     for k in range(t2.shape[1])], axis=1)


def od_field(t2):
    return np.stack([obs_dist(t2[:, k, :]) for k in range(t2.shape[1])], axis=1)


def simulate(t2, od, r=R, eps=EPS):
    """Destruction + arrival simulation. Returns totals, rates and time series."""
    n, t1, _ = t2.shape
    alive = np.ones(n, bool)
    dead_ag = np.zeros(n, bool)          # killed by another agent
    dead_ob = np.zeros(n, bool)          # killed by an obstacle
    arrived = np.zeros(n, bool)
    per_step = np.zeros(t1 - 1)
    dstep = np.full(n, -1)

    for k in range(t1 - 1):
        idx = np.flatnonzero(alive)
        if not idx.size:
            break
        hit = np.linalg.norm(t2[idx, k, :2] - TGT, axis=1) < eps      # reach target -> out
        arrived[idx[hit]] = True
        alive[idx[hit]] = False
        idx = idx[~hit]
        if not idx.size:
            continue
        z = t2[idx, k, :2]                                            # collide among survivors
        nn = cKDTree(z).query(z, k=2)[0][:, 1] if idx.size > 1 else np.full(1, np.inf)
        by_ag, by_ob = nn < 2 * r, od[idx, k] < r
        coll = by_ag | by_ob
        per_step[k] = coll.sum()
        dstep[idx[coll]] = k
        alive[idx[coll]] = False
        dead_ag[idx[by_ag]] = True
        dead_ob[idx[by_ob]] = True

    idx = np.flatnonzero(alive)                                        # arrival at final step
    if idx.size:
        arrived[idx[np.linalg.norm(t2[idx, -1, :2] - TGT, axis=1) < eps]] = True

    dead = dead_ag | dead_ob
    return dict(destroyed=int(dead.sum()), dead_by_agent=int(dead_ag.sum()),
                dead_by_obstacle=int(dead_ob.sum()), arrived=int(arrived.sum()),
                survival=1.0 - dead.sum() / n, success=arrived.sum() / n,
                per_step=per_step, death_step=dstep[dstep >= 0])


def path_len(t2):
    """Mean travelled distance per agent (xy)."""
    return float(np.linalg.norm(np.diff(t2[:, :, :2], axis=1), axis=-1).sum(1).mean())


def cache(path):
    """Fields that do not depend on r, so a sweep over r loads each rollout once."""
    z = np.load(path)
    t2 = z["traj"].astype(np.float32)[:, :, :2]
    return t2, nn_field(t2), od_field(t2), int(z["dim"]), int(z["n"])


def evaluate(path, r=R, eps=EPS, cached=None):
    """All reported metrics for one rollout."""
    t2, nn, od, dim, n = cached if cached is not None else cache(path)
    sim = simulate(t2, od, r, eps)
    return dict(dim=dim, n=n, r=r, eps=eps,
                min_dist=float(nn.min()), path_len=path_len(t2),
                destroyed=sim["destroyed"], dead_by_agent=sim["dead_by_agent"],
                dead_by_obstacle=sim["dead_by_obstacle"], arrived=sim["arrived"],
                survival=sim["survival"], success=sim["success"])
