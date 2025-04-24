#ifndef n_aryGrayCodeCounter_H
#define n_aryGrayCodeCounter_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

/**
@brief Class to iterate over n-ary reflected gray codes using pointers.
*/
class n_aryGrayCodeCounter
{

public:
  // number of digits in the counter
  size_t num_digits;
  // the current gray code associated to the offset value
  int *gray_code;
  // The maximal value of the individual gray code elements
  int *n_ary_limits;
  // The incremental counter chain associated to the gray code
  int *counter_chain;
  // the maximal offset in the counter (offset = prod( n_ary_limits[i] ))
  int64_t offset_max;
  // the current offset in the counter (0<= offset <= offset_max)
  int64_t offset;

  HOST_DEVICE n_aryGrayCodeCounter(const n_aryGrayCodeCounter &) = delete;
  HOST_DEVICE n_aryGrayCodeCounter &operator=(const n_aryGrayCodeCounter &) = delete;

  /**
  @brief Helper function to allocate arrays given num_digits.
  */
  HOST_DEVICE void allocate_arrays(size_t n)
  {
    num_digits = n;
    if (num_digits > 0)
    {
      n_ary_limits = new int[num_digits];
      gray_code = new int[num_digits];
      counter_chain = new int[num_digits];
    }
    else
    {
      n_ary_limits = nullptr;
      gray_code = nullptr;
      counter_chain = nullptr;
    }
  }

  /**
  @brief Default constructor of the class.
  */
  HOST_DEVICE n_aryGrayCodeCounter()
  {
    num_digits = 0;
    n_ary_limits = nullptr;
    gray_code = nullptr;
    counter_chain = nullptr;
    offset_max = 0;
    offset = 0;
  }

  /**
  @brief Constructor of the class.
  @param limits Array of maximal values of the individual gray code elements.
  @param n Length of the limits array.
  */
  HOST_DEVICE n_aryGrayCodeCounter(int limits[], size_t n)
  {
    allocate_arrays(n);
    if (num_digits == 0)
    {
      offset_max = 0;
      offset = 0;
      return;
    }

    // copy limits
    for (size_t idx = 0; idx < num_digits; idx++)
    {
      n_ary_limits[idx] = limits[idx];
    }

    offset_max = n_ary_limits[0];
    for (size_t idx = 1; idx < num_digits; idx++)
    {
      offset_max *= n_ary_limits[idx];
    }
    offset_max--;
    offset = 0;

    // initialize the counter with offset 0
    initialize(0);
  }

  /**
  @brief Constructor of the class.
  @param limits Array of maximal values of the individual gray code elements.
  @param n Length of the limits array.
  @param initial_offset The initial offset of the counter (0<=initial_offset<=offset_max).
  */
  HOST_DEVICE n_aryGrayCodeCounter(int limits[], size_t n, int64_t initial_offset)
  {
    allocate_arrays(n);
    if (num_digits == 0)
    {
      offset_max = 0;
      offset = 0;
      return;
    }

    for (size_t idx = 0; idx < num_digits; idx++)
    {
      n_ary_limits[idx] = limits[idx];
    }

    offset_max = n_ary_limits[0];
    for (size_t idx = 1; idx < num_digits; idx++)
    {
      offset_max *= n_ary_limits[idx];
    }
    offset_max--;
    offset = initial_offset;

    // initialize the counter with a given offset
    initialize(initial_offset);
  }

  /**
  @brief Destructor cleans up allocated dynamic arrays.
  */
  HOST_DEVICE ~n_aryGrayCodeCounter()
  {
    delete[] n_ary_limits;
    delete[] gray_code;
    delete[] counter_chain;
  }

  /**
  @brief Initialize the gray counter by zero offset.
  */
  HOST_DEVICE void initialize() { initialize(0); }

  /**
  @brief Initialize the gray counter to a specific initial offset.
  @param initial_offset The initial offset of the counter (0<= initial_offset <= offset_max).
  */
  HOST_DEVICE void initialize(int64_t initial_offset)
  {
    if (initial_offset < 0 || initial_offset > offset_max)
    {
      printf("n_aryGrayCodeCounter::initialize: Wrong value of initial_offset");
      return;
    }

    // generate counter chain
    int temp_offset = static_cast<int>(initial_offset);
    for (size_t idx = 0; idx < num_digits; idx++)
    {
      counter_chain[idx] = temp_offset % n_ary_limits[idx];
      temp_offset /= n_ary_limits[idx];
    }

    // determine the initial gray code corresponding to the given offset
    int parity = 0;
    for (int jdx = static_cast<int>(num_digits) - 1; jdx >= 0; jdx--)
    {
      int code = parity ? n_ary_limits[jdx] - 1 - counter_chain[jdx] : counter_chain[jdx];
      gray_code[jdx] = code;
      parity ^= (code & 1);
    }
  }

  /**
  @brief Get the current gray code counter value.
  @return Pointer to an array of ints representing the current gray code.
  */
  HOST_DEVICE int *get() { return gray_code; }

  /**
  @brief Iterate the counter to the next value.
  @return Returns 0 on success, or 1 if the counter has reached the maximal offset.
  */
  HOST_DEVICE int next()
  {
    int changed_index;
    int ret = next(changed_index);
    return ret;
  }

  /**
  @brief Iterate the counter to the next value.
  @param changed_index The index of the gray code element where change occurred.
  @return Returns 0 on success, or 1 if the counter has reached the maximal offset.
  */
  HOST_DEVICE int next(int &changed_index)
  {
    int value_prev, value;
    int ret = next(changed_index, value_prev, value);
    return ret;
  }

  /**
  @brief Iterate the counter to the next value.
  @param changed_index The index of the gray code element that changed.
  @param value_prev The previous value of the gray code element that changed.
  @param value The new value of the gray code element.
  @return Returns 0 on success, or 1 if the counter has reached the maximal offset.
  */
  HOST_DEVICE int next(int &changed_index, int &value_prev, int &value)
  {
    changed_index = 0;

    if (offset >= offset_max)
    {
      return 1;
    }

    bool update_counter = true;
    size_t counter_chain_idx = 0;
    while (update_counter)
    {
      if (counter_chain[counter_chain_idx] < n_ary_limits[counter_chain_idx] - 1)
      {
        counter_chain[counter_chain_idx]++;
        update_counter = false;
      }
      else if (counter_chain[counter_chain_idx] == n_ary_limits[counter_chain_idx] - 1)
      {
        counter_chain[counter_chain_idx] = 0;
        update_counter = true;
      }
      counter_chain_idx++;
    }

    int parity = 0;
    // iterate over gray code in reverse order to determine the updated value
    for (int jdx = static_cast<int>(num_digits) - 1; jdx >= 0; jdx--)
    {
      int gray_code_new_val = parity ? n_ary_limits[jdx] - 1 - counter_chain[jdx]
                                     : counter_chain[jdx];
      parity ^= (gray_code_new_val & 1);
      if (gray_code_new_val != gray_code[jdx])
      {
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

  HOST_DEVICE void set_offset_max(const int64_t &value) { offset_max = value; }
  HOST_DEVICE int64_t get_offset_max() { return offset_max; }
  HOST_DEVICE int64_t get_offset() { return offset; }

}; // n_aryGrayCodeCounter

#endif