#ifndef UTILS_H
#define UTILS_H

#include <numeric>
#include <vector>

#ifdef __SIZEOF_INT128__ // Check if __int128 is supported
using int_type = __int128;
#else
using int_type = int64_t; // Fallback to int64_t
#endif

inline int_type binomialCoeffTemplated(int n, int k) {
  std::vector<int_type> C(k + 1, 0);
  C[0] = 1;
  for (int i = 1; i <= n; i++) {
    for (int j = std::min(i, k); j > 0; j--)
      C[j] = C[j] + C[j - 1];
  }
  return C[k];
}

inline int_type binomialCoeffInt128(int n, int k) {
  return binomialCoeffTemplated(n, k);
}

template <typename T> int sum(std::vector<T> vec) {
  return std::accumulate(vec.begin(), vec.end(), 0);
}

#endif