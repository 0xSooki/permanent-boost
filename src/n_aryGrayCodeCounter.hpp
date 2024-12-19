#ifndef n_aryGrayCodeCounter_H
#define n_aryGrayCodeCounter_H

#include <cstdint>
#include <vector>

/**
@brief Class iterate over n-ary reflected gray codes
*/
class n_aryGrayCodeCounter {

protected:
  /// the current gray code associated to the offset value
  std::vector<int> gray_code;
  /// The maximal value of the individual gray code elements
  std::vector<int> n_ary_limits;
  /// The incremental counter chain associated to the gray code
  std::vector<int> counter_chain;
  /// the maximal offset in the counter offset = prod( n_ary_limits[i] )
  int64_t offset_max;
  /// the current offset in the counter 0<= offset <= offset_max
  int64_t offset;

public:
  /**
  @brief Default constructor of the class.
  @return Returns with the instance of the class.
  */
  n_aryGrayCodeCounter() {
    offset_max = 0;
    offset = 0;
  };

  /**
  @brief Constructor of the class.
  @param n_ary_limits_in The maximal value of the individual gray code elements
  */
  n_aryGrayCodeCounter(std::vector<int> &n_ary_limits_in) {
    n_ary_limits = n_ary_limits_in;

    if (n_ary_limits.size() == 0) {
      offset_max = 0;
      offset = 0;
      return;
    }

    offset_max = n_ary_limits[0];
    for (size_t idx = 1; idx < n_ary_limits.size(); idx++) {
      offset_max *= n_ary_limits[idx];
    }

    offset_max--;
    offset = 0;

    // initialize the counter
    initialize(0);
  }

  /**
  @brief Constructor of the class.
  @param n_ary_limits_in The maximal value of the individual gray code elements
  @param initial_offset The initial offset of the counter
  0<=initial_offset<=offset_max
  */
  n_aryGrayCodeCounter(std::vector<int> &n_ary_limits_in,
                       int64_t initial_offset) {

    n_ary_limits = n_ary_limits_in;

    if (n_ary_limits.size() == 0) {
      offset_max = 0;
      offset = 0;
      return;
    }

    offset_max = n_ary_limits[0];
    for (size_t idx = 1; idx < n_ary_limits.size(); idx++) {
      offset_max *= n_ary_limits[idx];
    }

    offset_max--;
    offset = initial_offset;

    // initialize the counter
    initialize(initial_offset);
  }

  /**
  @brief Initialize the gray counter by zero offset
  */
  void initialize() { initialize(0); }

  /**
  @brief Initialize the gray counter to specific initial offset
  @param initial_offset The initial offset of the counter 0<= initial_offset <=
  offset_max
  */
  void initialize(int64_t initial_offset) {
    if (initial_offset < 0 || initial_offset > offset_max) {
      std::string error(
          "n_aryGrayCodeCounter::initialize:  Wrong value of initial_offset");
      throw error;
    }

    // generate counter chain
    counter_chain = std::vector<int>(n_ary_limits.size());

    for (size_t idx = 0; idx < n_ary_limits.size(); idx++) {
      counter_chain[idx] = initial_offset % n_ary_limits[idx];
      initial_offset /= n_ary_limits[idx];
    }

    // determine the initial gray code corresponding to the given offset
    gray_code = std::vector<int>(n_ary_limits.size());
    int parity = 0;
    for (unsigned long long jdx = n_ary_limits.size() - 1; jdx != ~0ULL;
         jdx--) {
      gray_code[jdx] = parity ? n_ary_limits[jdx] - 1 - counter_chain[jdx]
                              : counter_chain[jdx];
      parity = parity ^ (gray_code[jdx] & 1);
    }
  }

  /**
  @brief Get the current gray code counter value
  @return Returns with the current gray code associated with the current
  offset.
  */
  std::vector<int> get() { return gray_code; }

  /**
  @brief Iterate the counter to the next value
  */
  int next() {
    int changed_index;

    int &&ret = next(changed_index);
    return ret;
  }

  /**
  @brief Iterate the counter to the next value
  @param changed_index The index of the gray code element where change
  occured.
  */
  int next(int &changed_index) {
    int value_prev, value;
    int &&ret = next(changed_index, value_prev, value);
    return ret;
  }

  /**
  @brief Iterate the counter to the next value
  @param changed_index The index of the gray code element where change
  occured.
  @param value_prev The previous value of the gray code element that changed.
  @param value The new value of the gray code element.
  */
  int next(int &changed_index, int &value_prev, int &value) {
    // determine the index which is about to modify
    changed_index = 0;

    if (offset >= offset_max) {
      return 1;
    }

    bool update_counter = true;
    int counter_chain_idx = 0;
    while (update_counter) {

      if (counter_chain[counter_chain_idx] <
          n_ary_limits[counter_chain_idx] - 1) {
        counter_chain[counter_chain_idx]++;
        update_counter = false;
      } else if (counter_chain[counter_chain_idx] ==
                 n_ary_limits[counter_chain_idx] - 1) {
        counter_chain[counter_chain_idx] = 0;
        update_counter = true;
      }

      counter_chain_idx++;
    }

    // determine the updated gray code
    int parity = 0;
    for (size_t jdx = n_ary_limits.size() - 1; jdx != ~0ULL; jdx--) {
      int gray_code_new_val = parity
                                  ? n_ary_limits[jdx] - 1 - counter_chain[jdx]
                                  : counter_chain[jdx];
      parity = parity ^ (gray_code_new_val & 1);

      if (gray_code_new_val != gray_code[jdx]) {
        value_prev = gray_code[jdx];
        value = gray_code_new_val;
        gray_code[jdx] = gray_code_new_val;
        changed_index = jdx;
        break;
      }
    }

    offset++;

    return 0;
  }

  void set_offset_max(const int64_t &value) { offset_max = value; }

  int64_t get_offset_max() { return offset_max; }

  int64_t get_offset() { return offset; }

}; // n_aryGrayCodeCounter

#endif