def subarry(nums, k):
    length = len(nums)

    window_sum = sum(nums[:k]) / k
    max_window_sum = window_sum
    max_window_sum_index = 0

    for i in range(length-k):
        window_sum = window_sum - nums[i] + nums[i + k] / k

        if(max_window_sum < window_sum):
            max_window_sum = window_sum
    return max_window_sum


# Example usage:
nums = [5]
print(subarry(nums, 1))