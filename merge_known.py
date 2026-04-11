#!/usr/bin/env python3
"""Merge multiple known_espada.bin (or known_broquel.bin) files into one."""

import struct
import sys

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} output.bin input1.bin input2.bin ...")
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    entries = {}
    rec = struct.Struct("<IIIBb")

    for fname in input_files:
        data = open(fname, "rb").read()
        (count,) = struct.unpack_from("<Q", data)
        off = 8
        for _ in range(count):
            w, b, k, btm, v = rec.unpack_from(data, off)
            entries[(w, b, k, btm)] = v
            off += rec.size
        print(f"  {fname}: {count} positions")

    with open(output_file, "wb") as out:
        out.write(struct.pack("<Q", len(entries)))
        for (w, b, k, btm), v in entries.items():
            out.write(rec.pack(w, b, k, btm, v))

    print(f"Wrote {len(entries)} unique positions to {output_file}")

if __name__ == "__main__":
    main()
