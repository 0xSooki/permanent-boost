#ifndef UTILS_H
#define UTILS_H

#include <numeric>
#include <vector>

template <typename int_type>
int_type binomialCoeffTemplated(int n, int k)
{
  int_type C[k + 1];
  memset(C, 0, sizeof(C));
  C[0] = 1;
  for (int i = 1; i <= n; i++)
  {
    for (int j = std::min(i, k); j > 0; j--)
      C[j] = C[j] + C[j - 1];
  }
  return C[k];
}

inline __int128 binomialCoeffInt128(int n, int k)
{
  return binomialCoeffTemplated<__int128>(n, k);
}

template <typename T>
int sum(std::vector<T> vec)
{
  return std::accumulate(vec.begin(), vec.end(), 0);
}

#endif