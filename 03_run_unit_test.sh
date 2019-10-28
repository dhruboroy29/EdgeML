#!/usr/bin/env bash

# Run Python pipeline
echo "Running Python inference pipeline on test data..."
python3 "C++/emi_fastgrnn.py"

# Run C++ pipeline
echo "Done! Running C++ inference pipeline on test data..."
cd "C++"
g++ "emi_fastgrnn.cpp"
./a.out

# Show diffs of outputs
echo "Test complete. Printing diff of outputs..."
diff out_py.csv out_c++.csv
