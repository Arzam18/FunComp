"""SPSA tuning with fastchess: tools/spsa.ps1 in Python, for Linux (2026-10-19:
the 3.0 tune on the 7945HX; Python 3 standard library only).

The same method and files as spsa.ps1 (read its description): each iteration
perturbs every parameter at once by +-c_k (signs at random), plays game pairs
between the plus and the minus engine, and moves the parameters towards the
side that scored better; c_k = c / k^0.101, a_k = a / (A + k)^0.602 with A a
tenth of the iterations, scaled to end at c_end and r_end * c_end^2; values
rounded stochastically, so small integer parameters keep a gradient. Several
fastchess processes run at once (--workers), each one iteration; the state CSV
gets a line per finished iteration and running again resumes from its last line
(the same format as spsa.ps1's, so either can resume the other's tune).

    python3 tools/spsa.py --engine <TUNE build> --params tools/spsa/<set>.csv --iterations 2000 \\
        --book <book.epd> --state matches/spsa-<name>.csv --fastchess <fastchess> \\
        [--tc 8+0.08] [--pairs 8] [--workers 8] [--concurrency 4] [--option Name=Value ...]
"""
import argparse
import csv
import math
import os
import random
import re
import signal
import subprocess
import tempfile
import time

ALPHA, GAMMA = 0.602, 0.101
RESULT = re.compile(r"Wins: (\d+), Losses: (\d+), Draws: (\d+)")


def advertised(engine):
    out = subprocess.run([engine], input="uci\nquit\n", capture_output=True, text=True).stdout
    return {m.group(1) for m in re.finditer(r"^option name (\S+)", out, re.M)}


def round_stochastic(v):
    """One of the two nearest integers, the upper with probability the fractional part."""
    f = math.floor(v)
    return f + 1 if random.random() < v - f else f


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--engine", required=True)
    a.add_argument("--params", required=True)
    a.add_argument("--iterations", type=int, required=True)
    a.add_argument("--book", required=True)
    a.add_argument("--state", required=True)
    a.add_argument("--fastchess", required=True)
    a.add_argument("--tc", default="8+0.08")
    a.add_argument("--hash", type=int, default=16)
    a.add_argument("--pairs", type=int, default=8, help="game pairs an iteration (each opening with both colours)")
    a.add_argument("--workers", type=int, default=8, help="fastchess processes at once")
    a.add_argument("--concurrency", type=int, default=4, help="games at once in each")
    a.add_argument("--option", action="append", default=[], help="Name=Value, fixed in both engines")
    a.add_argument("--pgn-dir", default="", help="keep each iteration's games as <dir>/<iteration>.pgn")
    o = a.parse_args()

    for f in (o.engine, o.params, o.book, o.fastchess):
        if not os.path.exists(f):
            raise SystemExit(f"Not found: {f}")
    engine, book, fastchess = (os.path.abspath(f) for f in (o.engine, o.book, o.fastchess))
    fmt = "pgn" if book.endswith(".pgn") else "epd"
    spec = list(csv.DictReader(open(o.params)))
    if not spec:
        raise SystemExit(f"No parameters in {o.params}")
    names = [s["name"] for s in spec]
    # fastchess drops options the engine doesn't advertise: refuse to start.
    have = advertised(engine)
    for n in names + [x.split("=", 1)[0] for x in o.option]:
        if n not in have:
            raise SystemExit(f"{o.engine} doesn't advertise option '{n}': use a build configured with -DTUNE=ON")
    if o.pgn_dir:
        os.makedirs(o.pgn_dir, exist_ok=True)

    big_a = 0.1 * o.iterations
    p = {}
    for s in spec:
        c_end = float(s["c_end"])
        a_end = float(s["r_end"]) * c_end * c_end
        p[s["name"]] = dict(lo=float(s["min"]), hi=float(s["max"]), c0=c_end * o.iterations ** GAMMA,
                            a0=a_end * (big_a + o.iterations) ** ALPHA)

    def clamp(v, n):
        return min(max(v, p[n]["lo"]), p[n]["hi"])

    theta, done = {}, 0
    if os.path.exists(o.state):
        rows = list(csv.DictReader(open(o.state)))
        if rows:
            theta = {n: float(rows[-1][n]) for n in names}
            done = len(rows)  # iterations finish out of order: count them
            print(f"Resuming after iteration {done} of {o.iterations}", flush=True)
    if not theta:
        theta = {s["name"]: float(s["value"]) for s in spec}
        with open(o.state, "w", newline="") as f:
            f.write("iter,wins,losses,draws," + ",".join(names) + "\n")

    tmp = tempfile.mkdtemp(prefix="spsa-")

    def start(k):
        ck, delta, plus, minus = {}, {}, [], []
        for n in names:
            ck[n] = p[n]["c0"] / k ** GAMMA
            delta[n] = random.choice((-1, 1))
            plus.append(f"option.{n}={round_stochastic(clamp(theta[n] + ck[n] * delta[n], n))}")
            minus.append(f"option.{n}={round_stochastic(clamp(theta[n] - ck[n] * delta[n], n))}")
        args = ([fastchess, "-engine", f"cmd={engine}", "name=plus"] + plus +
                ["-engine", f"cmd={engine}", "name=minus"] + minus +
                ["-each", f"tc={o.tc}", f"option.Hash={o.hash}"] + [f"option.{x}" for x in o.option] +
                ["-openings", f"file={book}", f"format={fmt}", "order=random",
                 "-srand", str(random.randrange(2000000000)),
                 "-rounds", str(o.pairs), "-games", "2", "-repeat",
                 "-concurrency", str(o.concurrency),
                 "-draw", "movenumber=40", "movecount=8", "score=10",
                 "-resign", "movecount=3", "score=600",
                 "-ratinginterval", "0",
                 # fastchess keeps its state in config.json by default: one file a worker.
                 "-config", f"outname={os.path.join(tmp, f'{k}.json')}"])
        if o.pgn_dir:
            args += ["-pgnout", f"file={os.path.join(o.pgn_dir, f'{k}.pgn')}"]
        out = os.path.join(tmp, f"{k}.out")
        with open(out, "w") as fo, open(os.path.join(tmp, f"{k}.err"), "w") as fe:
            proc = subprocess.Popen(args, stdout=fo, stderr=fe, start_new_session=True)
        return dict(k=k, proc=proc, out=out, ck=ck, delta=delta)

    nxt, running, failures, finished = done + 1, [], 0, False
    try:
        while done < o.iterations:
            while len(running) < o.workers and nxt <= o.iterations:
                running.append(start(nxt))
                nxt += 1
            time.sleep(0.5)
            for job in [j for j in running if j["proc"].poll() is not None]:
                running.remove(job)
                # fastchess reports the first engine's (plus's) results last.
                found = RESULT.findall(open(job["out"]).read())
                if not found:
                    failures += 1
                    if failures >= 3:
                        raise SystemExit(f"Three iterations in a row gave no result; see {job['out']}")
                    print(f"Iteration {job['k']} gave no result (see {job['out']}); replaying it", flush=True)
                    running.append(start(job["k"]))
                    continue
                failures = 0
                w, l, d = (int(x) for x in found[-1])
                # theta += R_k * c_k * (W - L) * delta, with R_k = a_k / c_k^2.
                for n in names:
                    ak = p[n]["a0"] / (big_a + job["k"]) ** ALPHA
                    rk = ak / (job["ck"][n] ** 2)
                    theta[n] = clamp(theta[n] + rk * job["ck"][n] * (w - l) * job["delta"][n], n)
                done += 1
                with open(o.state, "a", newline="") as f:
                    f.write(f"{job['k']},{w},{l},{d}," + ",".join(f"{theta[n]:.3f}" for n in names) + "\n")
                for ext in ("out", "err", "json"):
                    try:
                        os.remove(os.path.join(tmp, f"{job['k']}.{ext}"))
                    except OSError:
                        pass
                shown = " ".join(f"{n}={theta[n]:.1f}" for n in names)
                print(f"[{done}/{o.iterations}] +{w} -{l} ={d}  {shown}", flush=True)
        finished = True
    finally:
        # Interrupted: don't leave fastchess (and its engines) running.
        for job in running:
            if job["proc"].poll() is None:
                try:
                    os.killpg(job["proc"].pid, signal.SIGTERM)
                except (OSError, AttributeError):
                    job["proc"].terminate()
        if finished:
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            os.rmdir(tmp)

    print("Final values (rounded):")
    for n in names:
        print(f"  {n}={round(theta[n])}")


if __name__ == "__main__":
    main()
