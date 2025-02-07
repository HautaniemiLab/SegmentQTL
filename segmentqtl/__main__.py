#!/usr/bin/env python3

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    import segmentqtl

    segmentqtl.main()
