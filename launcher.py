"""PyInstaller entry script. freeze_support() must run first so SpikeInterface's
worker processes (n_jobs > 1) start correctly inside the frozen app."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from hdmea.app import main
    sys.exit(main())
