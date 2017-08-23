#!/bin/bash

export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
export MACOSX_DEPLOYMENT_TARGET=10.10

cmake -G"Unix Makefiles" -H. -Bbuild
if ! cmake --build build --config Debug; then
    echo 'Error building project'
    exit 1
fi