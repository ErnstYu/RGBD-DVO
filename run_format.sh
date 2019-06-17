#!/usr/bin/env bash

set -e

find ./include ./src -iname "*.h" -or -iname "*.cpp" | xargs clang-format -i
