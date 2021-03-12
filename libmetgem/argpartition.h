#ifndef ARGPARTITION_H
#define ARGPARTITION_H

#include <algorithm>
#include <numeric>
#include <vector>

template <typename T>
std::vector<T> argpartition(std::vector<float> const &vec, const int &kth)
{
    std::vector<T> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0); // fill with 0,1,2,...

    std::partial_sort(indices.begin(), indices.begin() + kth + 1, indices.end(),
                      [&vec](T i, T j) {return vec[i]<vec[j];});

    return indices;
}
#endif /* ARGPARTITION_H */