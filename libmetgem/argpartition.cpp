#include "argpartition.h"

// template <typename T>
std::vector<int> argpartition(std::vector<float> const &vec, const int &kth)
{
    std::vector<int> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0); // fill with 0,1,2,...

    std::partial_sort(indices.begin(), indices.begin() + kth + 1, indices.end(),
                      [&vec](int i, int j) {return vec[i]<vec[j];});
    // std::partition(indices.begin(), indices.begin() + kth,
                      // [&vec](int i, int j) {return vec[i]<vec[j];});

    return indices;
}