#!/usr/bin/env bash
make clean build

make run-memory-allocation ARGS="-p ZafI7 -o add -n 128 -t 128 -o add"