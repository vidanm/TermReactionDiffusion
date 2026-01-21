g++ -std=c++20 main.cpp -o app \
  $(pkg-config --cflags --libs notcurses)

