#!/usr/bin/env python3
"""Launcher for the options pricing workbench.

    python3 run.py                 # serve on 127.0.0.1:8770 and open a browser
    python3 run.py --port 9000     # pick a port
    python3 run.py --no-browser    # just serve
    python3 run.py --verbose       # log every request
"""

from __future__ import annotations

import argparse
import errno
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Options pricing & Greeks workbench")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default: loopback only)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        import numpy  # noqa: F401
    except ImportError:
        print("NumPy is required.  Install it with:  python3 -m pip install numpy", file=sys.stderr)
        return 1

    from server.app import serve, serve_ipv6_loopback

    port = args.port
    httpd = None
    for attempt in range(12):
        try:
            httpd = serve(args.host, port, args.verbose)
            break
        except OSError as e:
            if e.errno in (errno.EADDRINUSE, errno.EACCES):
                print(f"  port {port} unavailable, trying {port + 1}")
                port += 1
                continue
            raise
    if httpd is None:
        print(f"Could not bind a port in {args.port}..{port}.", file=sys.stderr)
        return 1

    v6 = serve_ipv6_loopback(port, args.verbose) if args.host in ("127.0.0.1", "localhost", "::1") else None

    url = f"http://{args.host}:{port}/"
    bar = "=" * 62
    print(f"\n{bar}\n  Options Pricing & Greeks Workbench\n{bar}")
    print(f"  Serving   {url}")
    print(f"  Engines   Black-Scholes | binomial x4 | trinomial | MC | QMC | LSMC")
    print(f"  Surfaces  raw SVI | SSVI | cubic spline | thin-plate RBF")
    if v6:
        print(f"  Also on   http://[::1]:{port}/  (localhost resolves here first on many clients)")
    print(f"  Stop      Ctrl-C\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down")
    finally:
        httpd.shutdown()
        httpd.server_close()
        if v6:
            v6.shutdown()
            v6.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
