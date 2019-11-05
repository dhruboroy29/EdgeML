#!/usr/bin/env bash

# Remove output files
rm "C++/out_py.csv"
rm "C++/out_c++.csv"

# Run Python pipeline
echo "Running Python inference pipeline on test data..."
python3 "C++/emi_fastgrnn.py"

# Run C++ pipeline
echo "Done! Running C++ inference pipeline on test data..."
cd "C++"
g++ -O3 "emi_fastgrnn.cpp"
./a.out

# Show diffs of outputs
echo "Test complete. Printing diff of outputs..."
diff out_py.csv out_c++.csv
