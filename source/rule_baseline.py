"""Known device rules with gains estimated only from observed context.

This is a privileged-rule reference, not a learned model or an oracle gain.
It assumes future press commands execute, exposing physical execution errors.
"""
import numpy as np


def forecast(row):
    values = np.array(row['context'], dtype=float)
    family = row['family']
    gain = np.full(3, 2.0)
    modes = np.zeros(3, dtype=int)
    estimates = [[] for _ in range(3)]
    for t in range(1, len(values)):
        target, dt = row['context_actions'][t], row['context_dt'][t]
        observed = values[t]-values[t-1]
        if family in (0, 1):
            if target >= 0 and abs(observed[target]) > .5:
                estimates[target].append(abs(observed[target]))
                if family == 1:
                    modes[target] = int(observed[target] > 0)
        else:
            if target >= 0:
                modes[target] = 1-modes[target]
            for i in range(3):
                if modes[i] and observed[i] > .5:
                    estimates[i].append(observed[i]/dt)
    for i in range(3):
        if estimates[i]:
            gain[i] = np.median(estimates[i])
    current = values[-1].copy()
    outputs = []
    for target, dt in zip(row['actions'], row['durations']):
        if target >= 0:
            if family == 0:
                current[target] += gain[target]
            elif family == 1:
                modes[target] = 1-modes[target]
                current[target] += gain[target]*(1 if modes[target] else -1)
            else:
                modes[target] = 1-modes[target]
        if family == 2:
            current += modes*gain*dt
        current = np.clip(current, 0, 999)
        outputs.append(current.copy())
    return np.array(outputs)
