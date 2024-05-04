#compile program
clang++ -O2 -std=c++11 -o program main.cpp

# call program: ./program seed num gamma length t_max
./program 3 500 0.25 1000 500